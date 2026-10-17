# SPDX-License-Identifier: Apache-2.0
"""One rank's state the launches read, and the per-width kernel that reads it.

:class:`Packed` holds what every width shares -- per-layer weight views, the
bound caches, scratch, the peer buffer and the ``(ar, h_mid)`` buffer pairs --
and :class:`M3Kernel` is one built width. The split matters because FlyDSL
compiles per width and a build takes minutes, so everything that does not depend
on the token count is done once.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    SPARSE_BLOCK,
    Dims,
)
from sgl_amd_mono.minimax_m3.gfx950.config import HIDDEN as HIDDEN_SIZE

logger = logging.getLogger(__name__)

FP8_CACHE = (torch.float8_e4m3fn, torch.uint8)
SWIGLU_ALPHA = 1.702  # aiter's swiglu_mul_batch bakes this in
# The fused shared expert's routing weight. SGLang's TopK gives it
# sum(routed) / routed_scaling_factor and the MoE output is scaled back by
# routed_scaling_factor, so with renormalized routed weights it contributes 1.0.
SHARED_WEIGHT = 1.0

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/mem_cache/memory_pool.py": ("MiniMaxSparseKVPool", "use_aiter_sparse_pa"),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "MiniMaxHybridAttnBackend",
        "index_topk_freq",
    ),
}


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(what)


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


class Packed:
    """Per-rank state shared by every built width."""

    def __init__(self, model, hf_config, dims: Dims, ctx) -> None:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        self.dims = dims
        self.rank = par.tp_rank
        self.cpu_group = par.tp_group.cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.deadline_s = ctx.deadline_s

        layers = model.model.layers
        self.sparse_ids = [i for i, l in enumerate(layers) if l.is_layer_sparse]
        self.first_layer = self.sparse_ids[0]
        config = getattr(hf_config, "text_config", None) or hf_config
        self._pack_layers(layers, config)
        self._bind_caches()
        self._alloc_buffers()
        logger.info(
            "[mono][rank %d] minimax_m3 packed: %d sparse layers (%d..%d), %d "
            "block-table columns, %d KV slots",
            self.rank, len(self.sparse), self.first_layer, self.sparse_ids[-1],
            self.bt_width, self.num_slots,
        )  # fmt: skip

    # ------------------------------------------------------------- weights
    def _pack_layers(self, layers, config) -> None:
        """Each sparse layer's weight views and the build constants every width
        shares."""
        from sgl_amd_mono.minimax_m3.gfx950.weights import (
            SparseMoeLayer,
            sparse_selection,
        )

        heads = self.dims.index_heads
        cos_sin_tables: dict = {}
        self.sparse = [
            SparseMoeLayer.from_layer(layers[i], i, self.dims, heads, cos_sin_tables)
            for i in self.sparse_ids
        ]
        _, init_blocks, local_blocks = sparse_selection(config)
        first = layers[self.first_layer]
        sm_scale = float(first.self_attn.scaling)
        _need(
            abs(sm_scale - HEAD_DIM**-0.5) < 1e-12,
            f"softmax scale {sm_scale}, the kernels bake in {HEAD_DIM**-0.5}",
        )
        self.k4_build = dict(
            npes=self.dims.tp,
            sm_scale=sm_scale,
            eps=float(config.rms_norm_eps),
            route_scale=float(first.mlp.routed_scaling_factor),
            shared_weight=SHARED_WEIGHT,
            swiglu_limit=float(config.swiglu_limit),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            heads=heads,
        )

    # -------------------------------------------------------------- caches
    def _bind_caches(self) -> None:
        """Bind FP8 K / V / index-K, ``local_kv_heads`` heads a rank, page-128
        allocation.

        Logical views stay NHD. Sparse K / V bytes use page-16 SHUFFLE storage;
        index-K and HiCache transfer rows remain NHD.
        """
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxHybridAttnBackend,
        )
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if isinstance(backend, MiniMaxHybridAttnBackend):
            backend = backend.sparse  # the sparse backend owns the MiniMax pool
        _need(
            backend.index_topk_freq == 1,
            "mono selects blocks per layer; set SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1",
        )
        pool = backend.kv_pool
        _need(
            isinstance(pool, MiniMaxSparseKVPool),
            f"{type(pool).__name__}: not the MiniMax sparse KV pool",
        )
        # The byte order, not the allocation: the buffer is the right shape and
        # dtype either way, but only the AITER sparse PA path writes it in
        # page-16 SHUFFLE order (sglang#41998). Reading NHD bytes as SHUFFLE
        # would not fault, it would produce plausible garbage.
        _need(pool.use_aiter_sparse_pa, "the kernels need page-16 SHUFFLE KV storage")
        main = pool.main_pool
        _need(
            pool.page_size == SPARSE_BLOCK,
            f"page size {pool.page_size}, kernels index {SPARSE_BLOCK}-token pages",
        )
        _need(
            main.kv_cache_layout == "nhd",
            f"KV cache logical layout {main.kv_cache_layout!r}, expected NHD views",
        )
        _need(
            main.head_num == self.dims.local_kv_heads,
            f"{main.head_num} kv heads a rank, kernels write "
            f"{self.dims.local_kv_heads}",
        )
        ids = self.sparse_ids
        k0, v0 = pool.get_key_buffer(ids[0]), pool.get_value_buffer(ids[0])
        i0 = pool.get_index_k_buffer(ids[0])
        _need(
            k0.dtype in FP8_CACHE and v0.dtype in FP8_CACHE,
            f"KV cache {k0.dtype} / {v0.dtype}, kernels write e4m3",
        )
        _need(i0.dtype in FP8_CACHE, f"index cache {i0.dtype}, kernels write e4m3")
        _need(
            k0.shape[0] == i0.shape[0]
            and k0.shape[-1] == HEAD_DIM
            and i0.shape[-1] == HEAD_DIM,
            f"KV cache {tuple(k0.shape)} / index cache {tuple(i0.shape)}",
        )
        self.num_slots = k0.shape[0]
        self.k_ptrs, self.v_ptrs, self.index_ptrs = [], [], []
        self.layer_wait = {}
        for lid in ids:
            k = pool.get_key_buffer(lid)
            v = pool.get_value_buffer(lid)
            ik = pool.get_index_k_buffer(lid)
            _need(
                k.is_contiguous() and v.is_contiguous() and ik.is_contiguous()
                and k.shape == k0.shape and v.shape == v0.shape and ik.shape == i0.shape,
                f"layer {lid}: cache views differ from layer {ids[0]}'s",
            )  # fmt: skip
            self.k_ptrs.append(_ptr(k))
            self.v_ptrs.append(_ptr(v))
            self.index_ptrs.append(_ptr(ik))
            # Cached pointers must honour HiCache's per-layer H2D loads, as the
            # pool's own get_key_buffer does on the module path: the bytes may
            # still be in flight when the launch would read them.
            self.layer_wait[lid] = pool._wait_for_layer

        req_to_token = backend.req_to_token
        _need(
            req_to_token.dtype == torch.int32 and req_to_token.dim() == 2,
            f"req_to_token {req_to_token.dtype} {tuple(req_to_token.shape)}",
        )
        self.req_to_token = req_to_token
        # EAGLE reserves an extra allocator page for draft lookahead. It is not
        # an increase to the configured model context or score capacity.
        self.max_context_len = int(backend.max_context_len)
        self.bt_width = (self.max_context_len + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        _need(
            0 < self.bt_width <= MAX_INDEX_BLOCKS,
            f"{self.bt_width} blocks a request, scores hold {MAX_INDEX_BLOCKS}",
        )
        self.cols = torch.arange(
            0, self.bt_width * SPARSE_BLOCK, SPARSE_BLOCK, dtype=torch.int64,
            device=req_to_token.device,
        )  # fmt: skip

    # ------------------------------------------------------------- buffers
    def _alloc_buffers(self) -> None:
        """Scratch, the peer buffer, the step counter, the ``(ar, h_mid)`` pairs,
        each layer's fixed K1 arguments and the per-width row tables."""
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.minimax_m3.gfx950.layout import Layout, sym_layout
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import K1_ARGS
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import (
            SCRATCH_BYTES as K1_SCRATCH_BYTES,
        )

        dims, dev = self.dims, self.device
        self.layout = Layout(dims=dims)
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(
            self.layout.scratch_bytes, dtype=torch.uint8, device=dev
        )
        # The peer buffer carries the symmetric regions and, as its tail, the
        # guard's per-rank error flags -- the same shape the other model in this
        # package uses, so that flags_offset is simply the symmetric extent.
        self.sym_bytes = sym_layout(dims.tp)["_bytes"]
        self.peers = PeerBuffer(
            self.sym_bytes + guard.FLAG_BYTES, self.cpu_group, self.rank, dims.tp, dev
        )
        self.flags_offset = self.sym_bytes
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
            for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
            for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, dims.local_q_heads * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in),
                "w_qkv": _ptr(lw.w_qkv), "s_qkv": _ptr(lw.s_qkv),
                "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik),
                "cos_sin": _ptr(lw.cos_sin), "iq_out": _ptr(self.iq),
                "index_cache": self.index_ptrs[i],
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )
        # per token count: the step's block table and sequence lengths
        self.tables = {
            n: (
                torch.zeros(n, self.bt_width, dtype=torch.int32, device=dev),
                torch.ones(n, dtype=torch.int32, device=dev),
            )
            for n in range(1, MAX_TOKENS + 1)
        }

    # --------------------------------------------------------------- build
    def build_kernel(self, rows: int) -> M3Kernel:
        return M3Kernel(self, rows)

    def prepare_step(self, forward_batch, rows: int, query_len: int, is_verify: bool):
        """This step's causal rows from the replay-fresh decode / verify buffers."""
        from sgl_amd_mono.minimax_m3.gfx950.metadata import prepare_token_rows

        bt, seq_lens = self.tables[rows]
        return prepare_token_rows(
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.cols,
            bt,
            seq_lens,
            forward_batch.out_cache_loc[:rows],
            query_len,
            is_verify,
            self.max_context_len,
        )


class M3Kernel:
    """The ``rows``-token launch every sparse layer shares."""

    def __init__(self, packed: Packed, rows: int) -> None:
        from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel

        self.packed = packed
        self.rows = rows
        # FlyDSL compiles on the first launch: the eager warm-up forward that
        # reaches this width before its graph capture
        self.launcher = build_post_attn_kernel(
            tokens=rows,
            dims=packed.dims,
            deadline_s=packed.deadline_s,
            flags_offset=packed.flags_offset,
            **packed.k4_build,
        )

    def launch(self, layer_id, positions, hidden_states, residual, forward_batch, memo):
        """Sparse layer ``layer_id`` in a decoder layer's own terms:
        ``(hidden_states, residual)`` in, ``(hidden_states, residual)`` out.

        ``ar`` has to be in ``ars[i % 2]``, which it already is when the
        preceding launch left it there; the copy is for the first sparse layer,
        which the modules hand it. ``memo``: the step's row tables, derived once
        by its first routed layer.
        """
        p = self.packed
        i = layer_id - p.first_layer
        n = self.rows
        wait = p.layer_wait.get(layer_id)
        if wait is not None:
            wait(layer_id)  # HiCache's per-layer H2D load may still be in flight
        rows = memo.get("rows")
        if rows is None:
            rows = memo["rows"] = p.prepare_step(
                forward_batch, n, memo["query_len"], memo["is_verify"]
            )
        block_table, seq_lens = rows
        ar = p.ars[i % 2]
        if hidden_states.data_ptr() != ar.data_ptr():
            ar[:n].copy_(hidden_states.view(n, HIDDEN_SIZE))
        lw = p.sparse[i]
        self.launcher(
            _ptr(p.h), _ptr(p.q), _ptr(block_table), _ptr(seq_lens),
            p.k_ptrs[i], p.v_ptrs[i], _ptr(lw.k_scale), _ptr(lw.v_scale),
            _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post), _ptr(lw.gate),
            _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13), _ptr(lw.w2), _ptr(lw.s2),
            _ptr(p.h_mids[(i + 1) % 2]), _ptr(p.ars[(i + 1) % 2]),
            _ptr(p.scratch4), p.peers.local, _ptr(p.peers.addresses),
            _ptr(p.step), p.rank, lw.layer_id,
            block_table.shape[1], memo["query_len"],
            _ptr(p.k1_args[i]), _ptr(positions),
            _ptr(forward_batch.out_cache_loc), _ptr(residual.view(n, HIDDEN_SIZE)),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        if i == len(p.sparse) - 1:
            # Opens the next step's epoch, which is what keeps a layer's mailbox
            # tags distinct from the same layer's tags one step ago. Enqueued
            # after the last launch of this step, so no kernel can see it early.
            p.step.add_(1)
        return p.ars[(i + 1) % 2][:n], p.h_mids[(i + 1) % 2][:n]

    def capture_input(self) -> torch.Tensor:
        """K1's BF16 residual after the input add, before the norm: the hidden
        state an EAGLE3 draft captures at this layer's attention boundary (a
        reused buffer: the collector copies it)."""
        return self.packed.h[: self.rows]

    def health(self) -> Optional[torch.Tensor]:
        """The guard's error words, for the framework's fail-stop check."""
        return self.packed.peers.bytes

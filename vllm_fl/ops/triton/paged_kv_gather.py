# Copyright (c) 2025 BAAI. All rights reserved.
#
# Triton kernels for gathering paged KV cache into dense formats required by
# FlagGems DSA Triton kernels (triton_lighting_indexer_k_tiled, triton_sparse_mla_fwd).
#
# Background:
#   vLLM stores KV cache in paged blocks: (num_blocks, block_size, num_heads, head_dim).
#   Tokens for a sequence are scattered across non-contiguous blocks referenced by a
#   block_table. The FlagGems Triton attention kernels expect dense contiguous layout.
#   These kernels bridge that gap.
#
# Exported functions:
#   paged_kv_gather_flat(kv_cache, block_table, seq_lens)
#       -> (total_tokens, num_heads, head_dim), cu_seqlens
#   paged_kv_gather_batched(kv_cache, block_table, seq_lens)
#       -> (num_seqs, max_seq_len, num_heads, head_dim)
#   flat_to_local_indices(flat_indices, cu_q_seqlens, cu_kv_seqlens)
#       -> local_indices of same shape

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: gather paged KV to flat (varlen) layout
# ---------------------------------------------------------------------------
# Grid: (num_seqs, cdiv(max_seq_len, BT), num_heads)
# Each program handles BT consecutive token positions for one (seq, head).
# ---------------------------------------------------------------------------

@triton.jit
def _paged_kv_gather_flat_kernel(
    kv_cache_ptr,      # [NB, BS, NH, HD]
    block_table_ptr,   # [NS, MBK]  int32
    seq_lens_ptr,      # [NS]       int32
    cu_seqlens_ptr,    # [NS+1]     int64 — cumulative output offsets
    output_ptr,        # [TT, NH, HD]
    # strides for kv_cache
    stride_nb,
    stride_bs,
    stride_nh,
    stride_hd,
    # strides for output
    stride_ot,
    stride_oh,
    stride_od,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,
    HD: tl.constexpr,   # triton.next_power_of_2(head_dim)
    BT: tl.constexpr,   # tokens per CTA
):
    seq_id  = tl.program_id(0)
    tile_id = tl.program_id(1)
    head_id = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_id)
    token_start = tile_id * BT
    if token_start >= seq_len:
        return

    cu_offset = tl.load(cu_seqlens_ptr + seq_id)

    offs_t = token_start + tl.arange(0, BT)    # [BT]
    mask_t = offs_t < seq_len

    block_idx    = offs_t // block_size          # [BT]
    tok_in_block = offs_t % block_size           # [BT]

    # Load cache-block IDs from block_table
    cache_block = tl.load(
        block_table_ptr + seq_id * max_blocks_per_seq + block_idx,
        mask=mask_t, other=0,
    )  # [BT]

    offs_d = tl.arange(0, HD)
    mask_d = offs_d < head_dim

    # Source address per token: kv_cache[cache_block, tok_in_block, head_id, :]
    src_base = (
        cache_block * stride_nb
        + tok_in_block * stride_bs
        + head_id * stride_nh
    )  # [BT]

    data = tl.load(
        kv_cache_ptr + src_base[:, None] + offs_d[None, :] * stride_hd,
        mask=mask_t[:, None] & mask_d[None, :],
        other=0.0,
    )  # [BT, HD]

    # Destination: output[cu_offset + offs_t, head_id, :]
    out_base = (cu_offset + offs_t) * stride_ot + head_id * stride_oh
    tl.store(
        output_ptr + out_base[:, None] + offs_d[None, :] * stride_od,
        data,
        mask=mask_t[:, None] & mask_d[None, :],
    )


def paged_kv_gather_flat(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV cache into flat (varlen) layout.

    Args:
        kv_cache:    (num_blocks, block_size, num_heads, head_dim)
        block_table: (num_seqs, max_blocks_per_seq) — int32
        seq_lens:    (num_seqs,) — int32

    Returns:
        output:     (total_tokens, num_heads, head_dim)
        cu_seqlens: (num_seqs + 1,) int64 — cumulative token counts per seq
    """
    num_blocks, block_size, num_heads, head_dim = kv_cache.shape
    num_seqs = seq_lens.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=kv_cache.device)
    cu_seqlens[1:] = torch.cumsum(seq_lens.to(torch.int64), dim=0)
    total_tokens = int(cu_seqlens[-1].item())

    output = torch.empty(
        (total_tokens, num_heads, head_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )

    if total_tokens == 0:
        return output, cu_seqlens

    max_seq_len = int(seq_lens.max().item())
    BT = 16
    HD = triton.next_power_of_2(head_dim)

    grid = (num_seqs, triton.cdiv(max_seq_len, BT), num_heads)
    _paged_kv_gather_flat_kernel[grid](
        kv_cache,
        block_table,
        seq_lens,
        cu_seqlens,
        output,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        head_dim=head_dim,
        HD=HD,
        BT=BT,
    )
    return output, cu_seqlens


# ---------------------------------------------------------------------------
# Kernel: gather paged KV to batched padded layout
# ---------------------------------------------------------------------------
# Grid: (num_seqs, cdiv(max_seq_len, BT), num_heads)
# Output: (num_seqs, max_seq_len, num_heads, head_dim) — padded with zeros
# ---------------------------------------------------------------------------

@triton.jit
def _paged_kv_gather_batched_kernel(
    kv_cache_ptr,      # [NB, BS, NH, HD]
    block_table_ptr,   # [NS, MBK]  int32
    seq_lens_ptr,      # [NS]       int32
    output_ptr,        # [NS, MSL, NH, HD]
    # strides for kv_cache
    stride_nb,
    stride_bs,
    stride_nh,
    stride_hd,
    # strides for output
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,
    HD: tl.constexpr,
    BT: tl.constexpr,
):
    seq_id  = tl.program_id(0)
    tile_id = tl.program_id(1)
    head_id = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_id)
    token_start = tile_id * BT
    if token_start >= seq_len:
        return

    offs_t = token_start + tl.arange(0, BT)
    mask_t = offs_t < seq_len

    block_idx    = offs_t // block_size
    tok_in_block = offs_t % block_size

    cache_block = tl.load(
        block_table_ptr + seq_id * max_blocks_per_seq + block_idx,
        mask=mask_t, other=0,
    )

    offs_d = tl.arange(0, HD)
    mask_d = offs_d < head_dim

    src_base = (
        cache_block * stride_nb
        + tok_in_block * stride_bs
        + head_id * stride_nh
    )

    data = tl.load(
        kv_cache_ptr + src_base[:, None] + offs_d[None, :] * stride_hd,
        mask=mask_t[:, None] & mask_d[None, :],
        other=0.0,
    )

    out_base = (
        seq_id * stride_ob
        + offs_t * stride_os
        + head_id * stride_oh
    )  # [BT]
    tl.store(
        output_ptr + out_base[:, None] + offs_d[None, :] * stride_od,
        data,
        mask=mask_t[:, None] & mask_d[None, :],
    )


def paged_kv_gather_batched(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Gather paged KV cache into batched padded layout.

    Args:
        kv_cache:    (num_blocks, block_size, num_heads, head_dim)
        block_table: (num_seqs, max_blocks_per_seq) — int32
        seq_lens:    (num_seqs,) — int32

    Returns:
        output: (num_seqs, max_seq_len, num_heads, head_dim) — zero-padded
    """
    num_blocks, block_size, num_heads, head_dim = kv_cache.shape
    num_seqs = seq_lens.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    max_seq_len = int(seq_lens.max().item())

    output = torch.zeros(
        (num_seqs, max_seq_len, num_heads, head_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )

    if max_seq_len == 0:
        return output

    BT = 16
    HD = triton.next_power_of_2(head_dim)
    grid = (num_seqs, triton.cdiv(max_seq_len, BT), num_heads)

    _paged_kv_gather_batched_kernel[grid](
        kv_cache,
        block_table,
        seq_lens,
        output,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        head_dim=head_dim,
        HD=HD,
        BT=BT,
    )
    return output


# ---------------------------------------------------------------------------
# Kernel: convert flat global KV indices → per-sequence local indices
# ---------------------------------------------------------------------------
# The indexer (triton_lighting_indexer_k_tiled + bin_topk) returns indices
# into the *flat* concatenated KV buffer [total_kv_tokens].
# triton_sparse_mla_fwd expects indices into the per-sequence KV dimension.
# This kernel subtracts the per-sequence KV start offset for each token.
# ---------------------------------------------------------------------------

@triton.jit
def _flat_to_local_indices_kernel(
    indices_ptr,         # [Q, topk]  int32 or int64
    token_kv_offset_ptr, # [Q]        int64 — kv offset for the seq owning token q
    stride_q,            # stride along Q dimension
    topk: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Grid: (Q, cdiv(topk, BK))
    Subtracts token_kv_offset[q] from all indices[q, :].
    """
    q_id   = tl.program_id(0)
    tile_k = tl.program_id(1)

    offset = tl.load(token_kv_offset_ptr + q_id)

    offs_k = tile_k * BK + tl.arange(0, BK)
    mask_k = offs_k < topk

    ptr = indices_ptr + q_id * stride_q + offs_k
    vals = tl.load(ptr, mask=mask_k, other=0)
    # Indices that are -1 (invalid) must stay -1
    valid = vals >= 0
    vals = tl.where(valid, vals - offset, vals)
    tl.store(ptr, vals, mask=mask_k)


def flat_to_local_indices(
    flat_indices: torch.Tensor,     # (Q, topk) — flat global KV indices
    cu_q_seqlens: torch.Tensor,     # (num_seqs + 1,) int64 query cumulative lens
    cu_kv_seqlens: torch.Tensor,    # (num_seqs + 1,) int64 KV cumulative lens
) -> torch.Tensor:
    """Convert flat global KV indices to per-sequence local KV indices.

    The indexer returns indices into the flat concatenated KV buffer.
    triton_sparse_mla_fwd expects indices into each sequence's local KV buffer.

    Args:
        flat_indices:   (Q, topk) — output of bin_topk over flat KV logits
        cu_q_seqlens:   (num_seqs+1,) — cumulative query token counts
        cu_kv_seqlens:  (num_seqs+1,) — cumulative KV token counts

    Returns:
        local_indices: (Q, topk) — indices into per-sequence KV (in-place modify)
    """
    Q, topk = flat_indices.shape
    num_seqs = cu_q_seqlens.shape[0] - 1

    # Build per-query-token KV offset tensor
    token_kv_offset = torch.empty(Q, dtype=torch.int64, device=flat_indices.device)
    for i in range(num_seqs):
        q_start = int(cu_q_seqlens[i].item())
        q_end   = int(cu_q_seqlens[i + 1].item())
        if q_end > q_start:
            token_kv_offset[q_start:q_end] = cu_kv_seqlens[i]

    BK = 64
    grid = (Q, triton.cdiv(topk, BK))
    _flat_to_local_indices_kernel[grid](
        flat_indices,
        token_kv_offset,
        flat_indices.stride(0),
        topk=topk,
        BK=BK,
    )
    return flat_indices

# Copyright (c) 2025 BAAI. All rights reserved.
#
# Paged DSA Triton kernels — DSA is MQA, kv_cache dim 2 is always 1.
# Aligned with FlagGems DSA kernel style (indexer_k_tiled, sparse_mla).
#
# triton_paged_indexer_k_tiled_interface  — replaces npu_lightning_indexer
# triton_sparse_mla_fwd_paged_interface   — replaces npu_sparse_flash_attention

import torch
import triton
import triton.language as tl


# ===========================================================================
# Kernel 1: Paged Lightning Indexer
# Aligned with FlagGems indexer_k_tiled.py
# Grid: (NQ, NK)  NQ=cdiv(Q,BQ)=Q (BQ=1), NK=cdiv(max_kv_len,TK)
# Each program handles one query token × one KV tile (TK tokens).
# Inner loop iterates BK-wide chunks with warp_specialize, doing block_table
# lookup to resolve paged KV addresses.
# Output logits[q, kv_pos] is in LOCAL per-sequence KV coordinate space so
# bucket_sort_topk can directly use starts=0 / ends=kv_seq_len.
# ===========================================================================

indexer_fwd_configs = [
    triton.Config({"num_stages": 2, "num_warps": 4}),
    triton.Config({"num_stages": 4, "num_warps": 8}),
]


@triton.autotune(configs=indexer_fwd_configs, key=["Q", "K", "H", "D"])
@triton.jit
def triton_paged_indexer_k_tiled(
    q_index,            # [Q*H, D] — same row-per-(query,head) layout as FlagGems
    kv_cache_ptr,       # [NB, BS, 1, D] — MQA: third dim is always 1
    block_table_ptr,    # [NS, MBK] int32
    seq_ids_ptr,        # [Q] int32 — sequence id per query token
    kv_ends_ptr,        # [Q] int32 — KV seq length for that token's sequence
    weights,            # [Q*H] — flat, same row ordering as q_index
    logits,             # [Q, K] float32 — output; K = max_kv_len
    stride_qh,          # q_index: stride over Q*H dim
    stride_qd,          # q_index: stride over D dim
    stride_nb,          # kv_cache: stride over num_blocks dim
    stride_bs,          # kv_cache: stride over block_size dim
    stride_hd,          # kv_cache: stride over head_dim dim
    stride_bt,          # block_table: stride over max_blocks_per_seq dim
    stride_wh,          # weights stride
    stride_lm,          # logits: stride over Q dim
    stride_ln,          # logits: stride over K dim
    Q:  tl.constexpr,
    H:  tl.constexpr,
    K:  tl.constexpr,   # max_kv_len (logits output width)
    TK: tl.constexpr,   # KV tokens per tile
    D:  tl.constexpr,
    BQ: tl.constexpr,   # queries per tile (= 1, matching FlagGems)
    BK: tl.constexpr,   # KV tokens per inner chunk
    block_size:         tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    i_sh = tl.program_id(0)   # query tile (BQ=1 → one token per program)
    i_k  = tl.program_id(1)   # KV tile

    # Query token indices in this tile [BQ]
    offs_bq = tl.arange(0, BQ) + i_sh * BQ
    mask_bq = offs_bq < Q

    # KV seq lengths for this tile's queries [BQ]
    kv_ends = tl.load(kv_ends_ptr + offs_bq, mask_bq, other=0)

    # KV tile bounds in LOCAL per-seq coordinates
    bos_base = i_k * TK
    eos_vec  = tl.minimum(kv_ends, bos_base + TK)   # [BQ] per-query tile end
    bos = max(bos_base, 0)
    eos = min(eos_vec.max(0), K)
    CK = eos - bos
    if CK <= 0:
        return

    offs_d  = tl.arange(0, D)
    mask_d  = offs_d < D

    # (Q*H) indices for this tile [BQ*H]  — same pattern as FlagGems
    offs_bqh = tl.arange(0, BQ * H) + i_sh * (BQ * H)
    mask_bqh = offs_bqh < Q * H

    # Load Q [BQ*H, D]
    q_ptr = q_index + offs_bqh[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blk = tl.load(q_ptr, mask_bqh[:, None] & mask_d[None, :], other=0.0).to(tl.float16)

    # Load weights [BQ*H]
    w_blk = tl.load(weights + offs_bqh * stride_wh, mask_bqh, other=0.0).to(tl.float16)

    # seq_id for block_table: scalar since BQ=1
    seq_id = tl.load(seq_ids_ptr + i_sh * BQ, i_sh * BQ < Q, other=0)

    offs_boq = tl.arange(0, BQ) + i_sh * BQ   # query row indices for logits output

    CK_tiles = tl.cdiv(CK, BK)
    for ck in range(CK_tiles, warp_specialize=True):
        offs_bk    = bos + ck * BK + tl.arange(0, BK)   # local KV positions [BK]
        mask_bk    = offs_bk < eos

        # Clamp for safe block_table access on masked lanes
        offs_bk_c  = tl.where(mask_bk, offs_bk, 0)
        block_idx  = offs_bk_c // block_size              # [BK]
        tok_in_blk = offs_bk_c % block_size               # [BK]

        # Block-table lookup: cache_blk[b] = block_table[seq_id, block_idx[b]]
        cache_blk = tl.load(
            block_table_ptr + seq_id * stride_bt + block_idx,
            mask_bk, other=0,
        )  # [BK]

        # kv_cache[cache_blk, tok_in_blk, 0, :] → shape [D, BK]
        # (dim 2 = 0 is implicit; stride_ng drops out for MQA)
        k_ptr = (
            kv_cache_ptr
            + cache_blk[None, :] * stride_nb
            + tok_in_blk[None, :] * stride_bs
            + offs_d[:, None] * stride_hd
        )
        k_blk = tl.load(k_ptr, mask_d[:, None] & mask_bk[None, :], other=0.0).to(tl.float16)

        # [BQ*H, BK] = relu(q @ k) * w
        acc = tl.dot(q_blk, k_blk, out_dtype=tl.float16)
        acc = tl.maximum(acc, 0.0) * w_blk[:, None]

        # Sum over H heads → [BQ, BK]  (same reshape trick as FlagGems)
        out_blk = acc.trans().reshape(BK, BQ, H).sum(-1).trans()

        # Store to logits[q, local_kv_pos], with per-query boundary mask
        out_ptr = (
            logits
            + offs_boq[:, None] * stride_lm
            + offs_bk[None, :] * stride_ln
        )
        out_msk = (
            mask_bq[:, None]
            & mask_bk[None, :]
            & (offs_bk[None, :] < eos_vec[:, None])
        )
        tl.store(out_ptr, out_blk.to(tl.float16), out_msk)


def triton_paged_indexer_k_tiled_interface(
    q,              # (Q, H, D)
    kv_cache,       # (NB, BS, 1, D) — MQA
    block_table,    # (NS, MBK) int32
    weights,        # (Q, H)
    cu_q_seqlens,   # (NS+1,) int64 — cumulative query counts, cu_q_seqlens[0]=0
    seq_kv_lens,    # (NS,) int32 — KV seq length per sequence
):
    """Compute indexer logits over paged KV cache.

    Returns:
        logits:   (Q, max_kv_len) float32 — LOCAL per-seq KV scores
        kv_ends:  (Q,) int32 — KV length for each query's sequence
    """
    Q, H, D = q.shape
    _, BS, _, _ = kv_cache.shape
    NS = seq_kv_lens.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    max_kv_len = int(seq_kv_lens.max().item()) if NS > 0 else 0

    # Per-query seq_id and kv_end, built via repeat_interleave (no Python loop)
    q_lens  = cu_q_seqlens[1:] - cu_q_seqlens[:-1]          # (NS,) int64
    seq_ids = torch.repeat_interleave(
        torch.arange(NS, dtype=torch.int32, device=q.device),
        q_lens,
    )  # (Q,)
    kv_ends = seq_kv_lens[seq_ids].to(torch.int32)           # (Q,)

    K = max_kv_len
    logits = torch.full((Q, K), float("-inf"), dtype=torch.float32, device=q.device)

    if K == 0 or Q == 0:
        return logits, kv_ends

    BQ = 1
    BK = 64
    TK = 2048
    grid = (triton.cdiv(Q, BQ), triton.cdiv(K, TK))

    q_flat = q.reshape(Q * H, D)
    w_flat  = weights.reshape(Q * H)

    triton_paged_indexer_k_tiled[grid](
        q_flat,
        kv_cache,
        block_table,
        seq_ids,
        kv_ends,
        w_flat,
        logits,
        q_flat.stride(0),      # stride_qh
        q_flat.stride(1),      # stride_qd
        kv_cache.stride(0),    # stride_nb
        kv_cache.stride(1),    # stride_bs
        kv_cache.stride(3),    # stride_hd
        block_table.stride(1), # stride_bt
        w_flat.stride(0),      # stride_wh
        logits.stride(0),      # stride_lm
        logits.stride(1),      # stride_ln
        Q=Q, H=H, K=K, TK=TK, D=D, BQ=BQ, BK=BK,
        block_size=BS,
        max_blocks_per_seq=max_blocks_per_seq,
    )
    return logits, kv_ends


# ===========================================================================
# Kernel 2: Paged Sparse MLA Forward
# Aligned with FlagGems sparse_mla.py
# Grid: (Q, NH)  NH=cdiv(H,BH)  — flat query tokens, no batch dim
# Each program: one query token × BH heads.
# KV fetched by block_table lookup using local per-seq topk indices.
# kv_cache dim 2 = 1 (MQA); nope and rope caches are separate tensors.
# ===========================================================================

spar_mla_fwd_configs = [
    triton.Config({"num_stages": 4, "num_warps": 8}),
    triton.Config({"num_stages": 2, "num_warps": 4}),
]


@triton.autotune(configs=spar_mla_fwd_configs, key=["K", "D"])
@triton.jit
def triton_sparse_mla_fwd_paged(
    q,                  # [Q, H, DT]  DT = D + TD
    kv_nope_ptr,        # [NB, BS, 1, D]
    kv_rope_ptr,        # [NB, BS, 1, TD]
    block_table_ptr,    # [NS, MBK] int32
    seq_ids_ptr,        # [Q] int32
    indices,            # [Q, K] int32 — local per-seq topk indices (-1 = invalid)
    sm_scale: tl.constexpr,
    output,             # [Q, H, D]
    lse,                # [Q, H]
    stride_qm,          # q: stride over Q dim
    stride_qh,          # q: stride over H dim
    stride_qd,          # q: stride over D dim (also used for rope offset)
    stride_nb,          # kv_nope: stride over num_blocks dim
    stride_bs,          # kv_nope: stride over block_size dim
    stride_hd,          # kv_nope: stride over head_dim dim
    stride_rnb,         # kv_rope: stride over num_blocks dim
    stride_rbs,         # kv_rope: stride over block_size dim
    stride_rhd,         # kv_rope: stride over head_dim dim
    stride_bt,          # block_table: stride over MBK dim
    stride_tm,          # indices: stride over Q dim
    stride_tt,          # indices: stride over K dim
    stride_om,          # output: stride over Q dim
    stride_oh,          # output: stride over H dim
    stride_od,          # output: stride over D dim
    stride_lm,          # lse: stride over Q dim
    stride_lh,          # lse: stride over H dim
    K:   tl.constexpr,  # topk
    D:   tl.constexpr,  # nope dim (kv_lora_rank)
    TD:  tl.constexpr,  # rope dim (qk_rope_head_dim)
    DP:  tl.constexpr,  # triton.next_power_of_2(D)
    TDP: tl.constexpr,  # triton.next_power_of_2(TD)
    G:   tl.constexpr,  # H (all heads; VG=1 for MQA so G=H)
    BK:  tl.constexpr,
    BH:  tl.constexpr,
    block_size:         tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    i_q  = tl.program_id(0)   # query token index
    i_bh = tl.program_id(1)   # head-group index

    seq_id = tl.load(seq_ids_ptr + i_q)

    offs_h  = tl.arange(0, BH)
    offs_d  = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_t  = tl.arange(0, BK)

    mask_h  = i_bh * BH + offs_h < G
    mask_d  = offs_d < D
    mask_td = offs_td < TD

    # Load Q nope part [BH, D]
    q_base = q + i_q * stride_qm + i_bh * BH * stride_qh
    q_ptr  = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blk  = tl.load(q_ptr, mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float16)

    # Load Q rope part [BH, TD]  — starts at offset D along the last dim
    tq_ptr = q_base + D * stride_qd + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_blk = tl.load(tq_ptr, mask_h[:, None] & mask_td[None, :], other=0.0).to(tl.float16)

    max_log = tl.full([BH], float("-inf"), dtype=tl.float16)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float16)
    acc     = tl.zeros([BH, DP], dtype=tl.float16)

    log_scale: tl.constexpr = sm_scale * 1.44269504

    NK = tl.cdiv(K, BK)
    for ck in range(NK):
        # Load topk indices [BK]
        t_ptr  = indices + i_q * stride_tm + (ck * BK + offs_t) * stride_tt
        t_msk  = (ck * BK + offs_t) < K
        kv_ids = tl.load(t_ptr, t_msk, other=-1)   # [BK]
        mask_ids = kv_ids >= 0

        # Clamp invalid indices for safe block_table access
        kv_ids_c   = tl.where(mask_ids, kv_ids, 0)
        block_idx  = kv_ids_c // block_size          # [BK]
        tok_in_blk = kv_ids_c % block_size           # [BK]

        cache_blk = tl.load(
            block_table_ptr + seq_id * stride_bt + block_idx,
            mask_ids, other=0,
        )  # [BK]

        # Load nope KV: kv_nope[cache_blk, tok_in_blk, 0, :] → [D, BK]
        kv_ptr = (
            kv_nope_ptr
            + cache_blk[None, :] * stride_nb
            + tok_in_blk[None, :] * stride_bs
            + offs_d[:, None] * stride_hd
        )
        kv_blk = tl.load(kv_ptr, mask_d[:, None] & mask_ids[None, :], other=0.0).to(tl.float16)

        # Load rope KV: kv_rope[cache_blk, tok_in_blk, 0, :] → [TD, BK]
        tkv_ptr = (
            kv_rope_ptr
            + cache_blk[None, :] * stride_rnb
            + tok_in_blk[None, :] * stride_rbs
            + offs_td[:, None] * stride_rhd
        )
        tkv_blk = tl.load(tkv_ptr, mask_td[:, None] & mask_ids[None, :], other=0.0).to(tl.float16)

        # QK = (q_nope @ k_nope.T + q_rope @ k_rope.T) * log_scale  [BH, BK]
        qk = tl.dot(q_blk, kv_blk, out_dtype=tl.float16)
        qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float16) * log_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))

        # Online softmax (exp2-based, same as FlagGems)
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk  = tl.math.exp2(qk - new_max[:, None]).to(tl.float16)
        sum_qk  = tl.sum(exp_qk, axis=1)
        alpha   = tl.math.exp2(max_log - new_max).to(tl.float16)
        sum_exp = sum_exp * alpha + sum_qk
        acc     = acc * alpha[:, None]
        acc     = tl.dot(exp_qk, kv_blk.trans(), acc, out_dtype=tl.float16)  # [BH, DP]
        max_log = new_max.to(tl.float16)

    # Write output [BH, D]
    out_vals = acc / sum_exp[:, None]
    o_base   = output + i_q * stride_om + i_bh * BH * stride_oh
    o_ptr    = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptr, out_vals.to(q_blk.dtype), mask_h[:, None] & mask_d[None, :])

    # Write LSE [BH]
    fin_log = max_log + tl.math.log2(sum_exp.to(tl.float32))
    l_base  = lse + i_q * stride_lm + i_bh * BH * stride_lh
    tl.store(l_base + offs_h * stride_lh, fin_log.to(q_blk.dtype), mask_h)


def triton_sparse_mla_fwd_paged_interface(
    q_nope,        # (Q, H, D)   — projected nope query (ql_nope)
    q_rope,        # (Q, 1, TD)  — rope query (broadcast to H heads)
    kv_nope,       # (NB, BS, 1, D)
    kv_rope,       # (NB, BS, 1, TD)
    block_table,   # (NS, MBK) int32
    indices,       # (Q, K) int32 — local per-seq topk indices
    sm_scale,      # float
    cu_q_seqlens,  # (NS+1,) int64 — cu_q_seqlens[0]=0
    d_v: int = 512,
):
    """Paged sparse MLA forward pass.

    Returns:
        output: (Q, H, D)  — attention output in latent space
        lse:    (Q, H)     — log-sum-exp (for potential combine with other shards)
    """
    Q, H, D = q_nope.shape
    assert D == d_v, f"q_nope last dim {D} != d_v {d_v}"
    TD = q_rope.shape[-1]
    _, BS, _, _ = kv_nope.shape
    K  = indices.shape[1]
    NS = cu_q_seqlens.shape[0] - 1
    max_blocks_per_seq = block_table.shape[1]

    DP  = triton.next_power_of_2(D)
    TDP = triton.next_power_of_2(TD)

    # Per-query seq_id
    q_lens  = cu_q_seqlens[1:] - cu_q_seqlens[:-1]   # (NS,)
    seq_ids = torch.repeat_interleave(
        torch.arange(NS, dtype=torch.int32, device=q_nope.device),
        q_lens,
    )  # (Q,)

    # Expand rope query from (Q,1,TD) to (Q,H,TD) and concat: (Q, H, DT)
    q_rope_exp = q_rope.expand(Q, H, TD)
    q_combined = torch.cat([q_nope, q_rope_exp], dim=-1).contiguous()

    output = torch.zeros((Q, H, D), device=q_nope.device, dtype=q_nope.dtype)
    lse    = torch.full((Q, H), float("-inf"), device=q_nope.device, dtype=q_nope.dtype)

    if Q == 0 or K == 0:
        return output, lse

    BH = max(16, min(64, triton.next_power_of_2(H)))
    NH = triton.cdiv(H, BH)
    BK = 32
    grid = (Q, NH)

    triton_sparse_mla_fwd_paged[grid](
        q_combined,
        kv_nope,
        kv_rope,
        block_table,
        seq_ids,
        indices,
        sm_scale,
        output,
        lse,
        q_combined.stride(0),   # stride_qm
        q_combined.stride(1),   # stride_qh
        q_combined.stride(2),   # stride_qd
        kv_nope.stride(0),      # stride_nb
        kv_nope.stride(1),      # stride_bs
        kv_nope.stride(3),      # stride_hd
        kv_rope.stride(0),      # stride_rnb
        kv_rope.stride(1),      # stride_rbs
        kv_rope.stride(3),      # stride_rhd
        block_table.stride(1),  # stride_bt
        indices.stride(0),      # stride_tm
        indices.stride(1),      # stride_tt
        output.stride(0),       # stride_om
        output.stride(1),       # stride_oh
        output.stride(2),       # stride_od
        lse.stride(0),          # stride_lm
        lse.stride(1),          # stride_lh
        K=K, D=D, TD=TD, DP=DP, TDP=TDP, G=H,
        BK=BK, BH=BH,
        block_size=BS,
        max_blocks_per_seq=max_blocks_per_seq,
    )
    return output, lse

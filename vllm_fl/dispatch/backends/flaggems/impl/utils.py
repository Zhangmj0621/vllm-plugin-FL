from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_fl import envs

@dataclass
class FLCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """

    # CPU tensor of sequence lengths for host-side operations.
    # E.g., tensor([128, 256, 64]) for 3 requests with different seq lengths.
    seq_lens_cpu: torch.Tensor = None

    # Number of decode tokens per request, used for speculative decoding.
    # E.g., 1 for normal decoding, >1 for speculative decoding.
    decode_token_per_req: int = 1

    # Actual query sequence lengths for each token in the batch (CPU list).
    # E.g., [1, 1, 1, 128] for 3 decode tokens and 1 prefill with 128 tokens.
    actual_seq_lengths_q: list[int] = field(default_factory=list)

    # NPU tensor of position indices for rotary embeddings computation.
    # E.g., tensor([0, 1, 2, ...]) indicating token positions in sequence.
    positions: torch.Tensor = None

    # Current attention state (e.g., ChunkedPrefill, DecodeOnly).
    attn_state: Any = None

    # Optional metadata for prefill context parallel path.
    prefill_context_parallel_metadata: Any = None

    # Padding size for graph capture, -1 means not in graph mode.
    graph_pad_size: int = -1

    # Total number of tokens including padding, used for padding operations.
    num_input_tokens: int = 0

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int, num_actual_reqs: int) -> "FLCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return FLCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            max_seq_len=self.max_seq_len,
        )

# Used to offload connector to CPU
def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert FLMetadata
    connector.wait_for_layer_load(layer_name)

def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: list[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert FLMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)


def split_decodes_and_prefills(
    common_attn_metadata: FLCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.
    While pcp > 1, query_lens is split across pcp ranks, so we pass in the
    original query_lens and max_query_len to distinguish prefills and decodes.

    Args:
        common_attn_metadata: FLCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = (
        long_seq_metadata.query_lens_pcp_full_cpu if long_seq_metadata else None
    )
    max_query_len_pcp_full = (
        long_seq_metadata.max_query_len_pcp_full if long_seq_metadata else 0
    )
    max_query_len = (
        common_attn_metadata.max_query_len
        if max_query_len_pcp_full == 0
        else max_query_len_pcp_full
    )
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (
        (query_start_loc[1:] - query_start_loc[:-1])
        if query_lens_pcp_full is None
        else query_lens_pcp_full
    )
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)

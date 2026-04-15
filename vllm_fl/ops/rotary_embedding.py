# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional
import torch
import math
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding,
    DeepseekScalingRotaryEmbedding,
)
from .common import (
    rotate_gptj,
    rotate_neox,
)
from vllm_fl.platform import PlatformFL
from vllm_fl.dispatch import call_op
from vllm_fl.utils import is_vl_model, has_rope

_cos_mla: torch.Tensor = None
_sin_mla: torch.Tensor = None
_cos_cache: torch.Tensor = None
_sin_cache: torch.Tensor = None
_cos_sin_cache: torch.Tensor = None
_cos: torch.Tensor = None
_sin: torch.Tensor = None
_cos_slice: torch.Tensor = None
_sin_slice: torch.Tensor = None


def set_cos_and_sin(vllm_config, max_num_reqs, decode_token_per_req, dtype, device):
    global _cos_mla
    global _sin_mla
    global _cos
    global _sin

    if _cos_mla is not None or _sin_mla is not None or _cos is not None or _sin is not None:
        return

    model_config = vllm_config.model_config
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    if model_config.use_mla:
        rope_dim = model_config.hf_text_config.qk_rope_head_dim
        _cos_mla = torch.ones(max_num_batched_tokens, 1, 1, rope_dim, dtype=dtype, device=device)
        _sin_mla = torch.zeros(max_num_batched_tokens, 1, 1, rope_dim, dtype=dtype, device=device)
    elif not is_vl_model(vllm_config) and has_rope(vllm_config):
        rope_dim = model_config.get_head_size()
        # For models using partial rope like Qwen3-Next.
        if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
            rope_dim = int(rope_dim * model_config.hf_text_config.partial_rotary_factor)
        elif hasattr(model_config.hf_text_config, "rotary_dim"):
            rope_dim = int(model_config.hf_text_config.rotary_dim)
        _cos = torch.ones(1, max_num_batched_tokens, 1, rope_dim, dtype=dtype, device=device)
        _sin = torch.zeros(1, max_num_batched_tokens, 1, rope_dim, dtype=dtype, device=device)


def get_cos_and_sin_mla(positions, use_cache=False):
    global _cos_cache
    global _sin_cache
    cos = _cos_cache[positions].unsqueeze(1).unsqueeze(2)
    sin = _sin_cache[positions].unsqueeze(1).unsqueeze(2)
    if not use_cache:
        return cos, sin
    global _cos_mla
    global _sin_mla
    num_tokens = positions.size(0)
    _cos_mla[:num_tokens, ...] = cos
    _sin_mla[:num_tokens, ...] = sin
    return _cos_mla[:num_tokens, ...], _sin_mla[:num_tokens, ...]


def _record_cos_sin_cache(cos_sin_cache):
    global _cos_sin_cache
    if _cos_sin_cache is not None:
        return
    _cos_sin_cache = cos_sin_cache


def _record_cos_and_sin_cache(cos_cache, sin_cache):
    global _cos_cache
    global _sin_cache
    _cos_cache = cos_cache
    _sin_cache = sin_cache


def _record_cos_and_sin_cache_interleaved(cos_sin_cache):
    global _cos_cache
    global _sin_cache
    if _cos_cache is not None or _sin_cache is not None:
        return
    hidden_dim = cos_sin_cache.shape[-1] // 2
    cos_cache, sin_cache = cos_sin_cache.view(-1, 2, hidden_dim).repeat(1, 1, 2).chunk(2, dim=1)
    _cos_cache = cos_cache.squeeze(1)
    _sin_cache = sin_cache.squeeze(1)


def update_cos_sin(positions):
    global _cos
    global _sin
    global _cos_slice
    global _sin_slice

    if _cos_sin_cache is None or _cos is None or _sin is None:
        return

    num_tokens = positions.size(0)
    _cos[:, :num_tokens] = (
        _cos_sin_cache.index_select(0, positions).view(num_tokens, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)[0]
    )
    _sin[:, :num_tokens] = (
        _cos_sin_cache.index_select(0, positions).view(num_tokens, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)[1]
    )
    _cos_slice = _cos[:, :num_tokens]
    _sin_slice = _sin[:, :num_tokens]


def get_cos_and_sin_slice():
    return _cos_slice, _sin_slice

class RotaryEmbeddingFL(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base,
            is_neox_style, dtype
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = call_op(
            "rotary_embedding",
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions,
            not self.is_neox_style,  # rotary_interleaved
            True,  # inplace
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key


# We no longer use DeepseekScalingRotaryEmbedding.forward to calculate q_pe and k_pe togather
# for computation and communication overlap
# We directly get cos and sin from the cache.
class FLDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            self._yarn_get_mscale(self.scaling_factor, float(mscale))
            / self._yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, scaling_factor, dtype
        )

        # NOTE: Reorder sin and cos cache
        self.max_seq_len = math.ceil(max_position_embeddings * scaling_factor)
        self._set_cos_sin_cache(self.max_seq_len, device=PlatformFL.device_type, dtype=dtype)

    def _yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_linear_ramp_mask(self, min_value, max_value, dim):
        # Note: The if conditional branch is not used here
        # to solve MTP compilation error.
        max_value += (min_value == max_value).float() * 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Inverse dim formula to find dim based on number of rotations
    def _yarn_find_correction_dim(self, num_rotations, dim, base=10000, max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        return (dim * torch.log(torch.tensor(max_position_embeddings) / (num_rotations * 2 * torch.pi))) / (
            2 * torch.log(torch.tensor(base))
        )

    # Find dim range bounds based on rotations
    def _yarn_find_correction_range(self, low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        low = torch.floor(self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = torch.ceil(self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        # Note: use torch instead of max/min to solve MTP compilation error.
        return torch.clamp(low, min=0), torch.clamp(high, max=dim - 1)

    def _set_cos_sin_cache(self, max_seq_len, device, dtype):
        dim = self.rotary_dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)
        cos_cached = torch.cat([freqs, freqs], dim=-1).cos() * self.mscale
        sin_cached = torch.cat([freqs, freqs], dim=-1).sin() * self.mscale
        cos_cached = cos_cached.to(dtype)
        sin_cached = sin_cached.to(dtype)
        cache = torch.cat([freqs.cos() * self.mscale, freqs.sin() * self.mscale], dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        _record_cos_sin_cache(cache)
        _record_cos_and_sin_cache(cos_cached, sin_cached)

    def rope_single_query(
        self,
        query: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        query_rot = query[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]

        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
        else:
            query = query_rot
        return query
    
    def rope_single_key(
        self,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            key_pass = key[..., self.rotary_dim :]

        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            key = key_rot
        return key


__all__ = ["RotaryEmbeddingFL"]

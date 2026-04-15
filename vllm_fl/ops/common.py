from __future__ import annotations

import torch


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims (NeoX-style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved dims (GPT-J style)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


__all__ = [
    "rotate_gptj",
    "rotate_neox",
]
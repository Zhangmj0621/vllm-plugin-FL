"""SP (Sequence Parallelism) state management for DSA-CP.

Per-forward-pass state controlling whether reduce_scatter/all_gather
are used instead of all_reduce in the linear op dispatch system.
"""

from vllm.distributed import tensor_model_parallel_all_gather

_sp_enabled: bool = False
_pad_size: int = 0


def set_sp_state(enabled: bool, pad_size: int = 0) -> None:
    global _sp_enabled, _pad_size
    _sp_enabled = enabled
    _pad_size = pad_size


def is_sp_enabled() -> bool:
    return _sp_enabled


def get_sp_pad_size() -> int:
    return _pad_size


def maybe_all_gather_and_unpad(x):
    """All_gather input along dim 0, then remove SP padding.

    Used by SequenceColumnParallelOp before matmul to restore
    the full token dimension that was reduce_scattered by the
    previous SequenceRowParallelOp.
    """
    if not _sp_enabled:
        return x
    x = tensor_model_parallel_all_gather(x, 0)
    if _pad_size > 0:
        x = x[:-_pad_size]
    return x

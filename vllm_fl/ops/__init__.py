"""FL plugin operator overrides.

This subpackage intentionally provides out-of-tree (OOT) replacements for
selected vLLM ops. Keep imports lightweight to avoid importing torch / CUDA
extensions at module import time.
"""

__all__ = []
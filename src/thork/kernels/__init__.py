"""
thork.kernels — a library of pre-built CUDA compute kernels.

Each function in this module takes the relevant compile-time dimensions
and parameters, generates a kernel specialized for those settings,
computes suitable launch parameters, and returns a ``BoundKernel`` that
can be called directly with the numpy array arguments — no need to
specify a grid or block size at the call site.

    matmul = tk.kernels.matmul(1024, 1024, 1024)
    matmul(C, A, B)

    add = tk.kernels.matrix_add((1024, 1024))
    add(C, A, B)
"""

from .matmul import matmul
from .matrix_add import matrix_add

__all__ = [
    "matmul",
    "matrix_add",
]

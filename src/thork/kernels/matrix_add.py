import math
from typing import Sequence

from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..types import (
    BlockDim,
    BlockIdx,
    DevicePointer,
    ThreadIdx,
    Uint,
)


def matrix_add(
    shape : Sequence[int],
    dtype : dt.Dtype = dt.float32,
) -> BoundKernel:
    """
    Elementwise add: ``out = A + B`` over arrays of the given shape.

    The shape is only used to pick a sensible launch geometry; the kernel
    itself treats inputs as flat 1-D buffers.

    Block size defaults to 128 but is rounded down to the largest power
    of two that divides ``prod(shape)``, so any shape with a power-of-two
    element count works.
    """
    n = int(math.prod(shape))
    if n <= 0:
        raise ValueError(f"matrix_add: shape {tuple(shape)!r} has zero elements")

    block = 128
    while n % block != 0 and block > 1:
        block //= 2

    @jit
    def matrix_add_kernel(
        out : DevicePointer[dtype],
        A   : DevicePointer[dtype],
        B   : DevicePointer[dtype],
        bid : Uint[BlockIdx],
        tid : Uint[ThreadIdx],
        bdm : Uint[BlockDim],
    ):
        i = bid * bdm + tid
        out[i] = A[i] + B[i]

    return matrix_add_kernel.bind(
        grid=(n // block, 1, 1),
        block=(block, 1, 1),
    )

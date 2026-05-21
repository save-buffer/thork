from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..tracer import (
    local,
    range,
    shared,
    syncthreads,
)
from ..types import (
    BlockIdx,
    DevicePointer,
    ThreadIdx,
    Uint2,
)


_TILE = 16


def matmul(
    M     : int,
    N     : int,
    K     : int,
    dtype : dt.Dtype = dt.float32,
) -> BoundKernel:
    """
    Tiled ``out = A @ B`` using shared-memory tiles.

    Shapes:
      - A   : (M, K)
      - B   : (K, N)
      - out : (M, N)

    ``M``, ``N``, ``K`` must all be multiples of 16 (the shared tile size).
    """
    if M % _TILE != 0 or N % _TILE != 0 or K % _TILE != 0:
        raise ValueError(
            f"matmul: M, N, K must all be multiples of {_TILE}; got "
            f"M={M}, N={N}, K={K}"
        )

    @jit
    def matmul_kernel(
        out : DevicePointer[dtype],
        A   : DevicePointer[dtype],
        B   : DevicePointer[dtype],
        bid : Uint2[BlockIdx],
        tid : Uint2[ThreadIdx],
    ):
        A_tile = shared(dtype, (_TILE, _TILE))
        B_tile = shared(dtype, (_TILE, _TILE))

        row = bid.y * _TILE + tid.y
        col = bid.x * _TILE + tid.x

        acc = local(dtype, 0.0)
        for k_tile in range(0, K, _TILE):
            A_tile[tid.y, tid.x] = A[row * K + (k_tile + tid.x)]
            B_tile[tid.y, tid.x] = B[(k_tile + tid.y) * N + col]
            syncthreads()

            for k in range(_TILE):
                acc += A_tile[tid.y, k] * B_tile[k, tid.x]
            syncthreads()

        out[row * N + col] = acc

    return matmul_kernel.bind(
        grid=(N // _TILE, M // _TILE, 1),
        block=(_TILE, _TILE, 1),
    )

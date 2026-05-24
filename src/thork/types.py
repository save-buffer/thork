from dataclasses import dataclass
from typing import Optional, Type

from . import dtypes as dt


class ThreadAttribute:
    """
    Marker for a CUDA kernel attribute that gets synthesized at kernel entry.

    Subclasses set ``cuda_expr`` to the CUDA expression (or expression
    template using ``{dim}``) that produces the value.
    """

    cuda_expr : str = ""


class BlockIdx(ThreadAttribute):
    cuda_expr = "blockIdx"


class ThreadIdx(ThreadAttribute):
    cuda_expr = "threadIdx"


class BlockDim(ThreadAttribute):
    cuda_expr = "blockDim"


class GridDim(ThreadAttribute):
    cuda_expr = "gridDim"


class WarpSize(ThreadAttribute):
    cuda_expr = "warpSize"


class LaneId(ThreadAttribute):
    """
    Lane index within the warp (``threadIdx.x & 0x1f`` for 1-D blocks).
    """

    cuda_expr = "(threadIdx.x & 0x1f)"


class WarpId(ThreadAttribute):
    """
    Warp index within the block (``threadIdx.x >> 5`` for 1-D blocks).
    """

    cuda_expr = "(threadIdx.x >> 5)"


@dataclass(frozen=True)
class DevicePointerSpec:
    """
    A parameter spec: ``T *name`` (or ``const T *name``).
    """

    dtype : dt.Dtype


@dataclass(frozen=True)
class KittensGlobalSpec:
    """
    A parameter spec for a ThunderKittens 2-D global layout.

    Lowers to a ``const __grid_constant__ kittens::gl<T, 1, 1, -1, -1[, ST]>``
    kernel parameter. ``tile_shape`` (when set) controls the TMA tile type
    bundled into the ``gl<>`` — required for ``tma::load_async`` / etc.
    """

    dtype      : dt.Dtype
    tile_shape : Optional[tuple] = None


@dataclass(frozen=True)
class ScalarParamSpec:
    """
    A scalar (or vector) parameter spec.

    If ``attribute`` is set, the parameter is synthesized from a CUDA
    built-in (e.g. ``blockIdx.x * blockDim.x + threadIdx.x``). Otherwise
    it is a host-supplied by-value scalar.
    """

    dtype     : dt.Dtype
    cuda_name : str
    vec_size  : int = 1
    attribute : Optional[Type[ThreadAttribute]] = None


class _ScalarTypeBase:
    """
    Marker base for the bare scalar type classes (Uint, Uint3, Int, ...).

    Subscripting a subclass with a ThreadAttribute produces an attribute
    ScalarParamSpec; using the class bare in an annotation produces a
    by-value constant parameter (conversion happens in jit.py).
    """

    _dtype     : dt.Dtype
    _cuda_name : str
    _vec_size  : int = 1


class _DevicePointer:
    def __class_getitem__(cls, dtype : dt.Dtype) -> DevicePointerSpec:
        if not isinstance(dtype, dt.Dtype):
            raise TypeError(
                f"DevicePointer[...] expects a thork dtype, got {dtype!r}"
            )
        return DevicePointerSpec(dtype=dtype)


DevicePointer = _DevicePointer


class _KittensGlobal:
    """
    Annotation marker for a ThunderKittens 2-D global tensor parameter.

    Without a tile shape::

        A : tk.kittens.Global[tk.dt.bfloat16]

    With a TMA tile shape (required by ``tma::load_async`` etc.)::

        A : tk.kittens.Global[tk.dt.bfloat16, (TILE_M, TILE_K)]
    """

    def __class_getitem__(cls, args) -> KittensGlobalSpec:
        if isinstance(args, tuple):
            if len(args) < 1:
                raise TypeError("kittens.Global[...] requires at least a dtype")
            dtype = args[0]
            extras = args[1:]
        else:
            dtype = args
            extras = ()
        if not isinstance(dtype, dt.Dtype):
            raise TypeError(
                f"kittens.Global[...] expects a thork dtype as the first "
                f"argument, got {dtype!r}"
            )
        tile_shape : Optional[tuple] = None
        if extras:
            if len(extras) != 1:
                raise TypeError(
                    "kittens.Global[dtype, (R, C)] accepts at most one tile "
                    f"shape; got {len(extras)} extras"
                )
            shape = extras[0]
            if not (isinstance(shape, tuple) and len(shape) == 2
                    and all(isinstance(x, int) and x > 0 for x in shape)):
                raise TypeError(
                    "kittens.Global tile shape must be a (rows, cols) tuple "
                    f"of positive ints, got {shape!r}"
                )
            tile_shape = shape
        return KittensGlobalSpec(dtype=dtype, tile_shape=tile_shape)


KittensGlobal = _KittensGlobal


def _scalar_factory(dtype : dt.Dtype, cuda_name : str, vec_size : int = 1):
    class _ScalarType(_ScalarTypeBase):
        _dtype     = dtype
        _cuda_name = cuda_name
        _vec_size  = vec_size

        def __class_getitem__(cls, attribute):
            if not (isinstance(attribute, type) and issubclass(attribute, ThreadAttribute)):
                raise TypeError(
                    f"{cuda_name}[...] expects a ThreadAttribute subclass, "
                    f"got {attribute!r}"
                )
            return ScalarParamSpec(
                dtype=cls._dtype,
                cuda_name=cls._cuda_name,
                vec_size=cls._vec_size,
                attribute=attribute,
            )

    _ScalarType.__name__ = cuda_name.capitalize()
    return _ScalarType


Uint  = _scalar_factory(dt.uint32, "unsigned int", 1)
Uint2 = _scalar_factory(dt.uint32, "uint2",        2)
Uint3 = _scalar_factory(dt.uint32, "uint3",        3)
Int   = _scalar_factory(dt.int32,  "int",          1)
Int2  = _scalar_factory(dt.int32,  "int2",         2)
Int3  = _scalar_factory(dt.int32,  "int3",         3)

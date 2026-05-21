from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dtype:
    name      : str
    cuda      : str
    nbytes    : int
    is_float  : bool = False
    is_signed : bool = True

    def __repr__(self) -> str:
        return f"dt.{self.name}"


float32  = Dtype("float32",  "float",          4, is_float=True)
float16  = Dtype("float16",  "__half",         2, is_float=True)
bfloat16 = Dtype("bfloat16", "__nv_bfloat16",  2, is_float=True)
int32    = Dtype("int32",    "int",            4)
uint32   = Dtype("uint32",   "unsigned int",   4, is_signed=False)
int64    = Dtype("int64",    "long long",      8)
uint64   = Dtype("uint64",   "unsigned long long", 8, is_signed=False)
bool_    = Dtype("bool",     "bool",           1, is_signed=False)


_NUMPY_TO_DTYPE = {
    np.dtype(np.float32) : float32,
    np.dtype(np.float16) : float16,
    np.dtype(np.int32)   : int32,
    np.dtype(np.uint32)  : uint32,
    np.dtype(np.int64)   : int64,
    np.dtype(np.uint64)  : uint64,
    np.dtype(np.bool_)   : bool_,
}


def from_numpy(np_dtype) -> Dtype:
    """
    Map a numpy dtype to its thork Dtype counterpart.
    """
    key = np.dtype(np_dtype)
    if key not in _NUMPY_TO_DTYPE:
        raise TypeError(f"No thork dtype for numpy dtype {np_dtype}")
    return _NUMPY_TO_DTYPE[key]

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from cuda.bindings import driver, nvrtc

from . import dtypes as dt
from . import ir


_NVRTC_ERROR_LINE_RE = re.compile(r"(?:default_program|[A-Za-z0-9_./-]+\.cu)\((\d+)\)")


def _rewrite_compile_error(
    nvrtc_log    : str,
    source_map   : Optional[Dict[int, Tuple[str, int]]],
) -> str:
    """
    Find ``<file>.cu(LINE)`` references in an nvrtc compile log and prepend
    a header that translates each referenced line to the Python source
    location where the offending IR was emitted.
    """
    if not source_map:
        return nvrtc_log

    seen : Dict[int, Tuple[str, int]] = {}
    for match in _NVRTC_ERROR_LINE_RE.finditer(nvrtc_log):
        line = int(match.group(1))
        if line in source_map and line not in seen:
            seen[line] = source_map[line]

    if not seen:
        return nvrtc_log

    mapping_lines = [
        f"  generated line {ln} -> {fname}:{lineno}"
        for ln, (fname, lineno) in sorted(seen.items())
    ]
    header = (
        "Python source locations for reported generated lines:\n"
        + "\n".join(mapping_lines)
        + "\n\n"
    )
    return header + nvrtc_log


def _check(result, message : str = ""):
    """
    Unwrap a ``(status, *rest)`` tuple returned by a cuda-python binding.

    Raises ``RuntimeError`` on a non-success status; otherwise returns the
    payload (or the single payload element, if there's exactly one).
    """
    status, *rest = result
    if isinstance(status, driver.CUresult):
        if status != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(status)
            _, msg  = driver.cuGetErrorString(status)
            name_s = name.decode() if isinstance(name, bytes) else str(name)
            msg_s  = msg.decode()  if isinstance(msg, bytes)  else str(msg)
            ctx = f" ({message})" if message else ""
            raise RuntimeError(f"CUDA error{ctx}: {name_s}: {msg_s}")
    elif isinstance(status, nvrtc.nvrtcResult):
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            ctx = f" ({message})" if message else ""
            raise RuntimeError(f"NVRTC error{ctx}: {status}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


_device           = None
_context          = None
_compute_cap_str  = None


def get_device():
    """
    Initialize CUDA, create a primary context on device 0, and return the
    device handle. Subsequent calls return the cached handle.
    """
    global _device, _context, _compute_cap_str
    if _device is not None:
        return _device
    _check(driver.cuInit(0))
    _device = _check(driver.cuDeviceGet(0))
    _context = _check(driver.cuCtxCreate(None, 0, _device))
    major = _check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        _device,
    ))
    minor = _check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        _device,
    ))
    _compute_cap_str = f"{major}{minor}"
    return _device


def get_compute_capability() -> str:
    """
    Return the compute-capability string of device 0 as ``"<major><minor>"``
    (e.g. ``"80"`` for A100, ``"90"`` for H100, ``"120"`` for GB10).
    """
    if _compute_cap_str is None:
        get_device()
    return _compute_cap_str


def _default_include_dirs() -> List[str]:
    """
    Best-effort discovery of the CUDA include directory.

    Honors ``$CUDA_HOME`` / ``$CUDA_PATH`` first; falls back to
    ``/usr/local/cuda/include``.
    """
    dirs : List[str] = []
    for envvar in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(envvar)
        if root:
            candidate = os.path.join(root, "include")
            if os.path.isdir(candidate):
                dirs.append(candidate)
    if not dirs:
        for candidate in ("/usr/local/cuda/include", "/opt/cuda/include"):
            if os.path.isdir(candidate):
                dirs.append(candidate)
                break
    return dirs


def compile_source(
    source       : str,
    *,
    source_map   : Optional[Dict[int, Tuple[str, int]]] = None,
    extra_opts   : Optional[Sequence[str]] = None,
    include_dirs : Optional[Sequence[str]] = None,
):
    """
    Compile ``source`` with nvrtc and load the resulting PTX as a CUDA
    module. Returns the loaded module handle.
    """
    get_device()
    prog = _check(nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"), b"thork_kernel.cu", 0, [], [],
    ))

    cap = get_compute_capability()
    opts : List[bytes] = [
        f"--gpu-architecture=compute_{cap}".encode("utf-8"),
        b"-std=c++17",
        b"--use_fast_math",
    ]
    for d in _default_include_dirs():
        opts.append(f"-I{d}".encode("utf-8"))
    if include_dirs:
        for d in include_dirs:
            opts.append(f"-I{d}".encode("utf-8"))
    if extra_opts:
        for o in extra_opts:
            opts.append(o.encode("utf-8"))

    status, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    log_text = log.rstrip(b"\x00 ").decode("utf-8", errors="replace")

    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        nvrtc.nvrtcDestroyProgram(prog)
        rewritten = _rewrite_compile_error(log_text, source_map)
        raise RuntimeError(f"NVRTC compile failed:\n{rewritten}\n\nSource:\n{source}")

    ptx_size = _check(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    _check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)

    module = _check(driver.cuModuleLoadData(ptx))
    return module


def get_function(module, name : str):
    """
    Look up a kernel function by its (mangled or extern "C") name.
    """
    return _check(driver.cuModuleGetFunction(module, name.encode("utf-8")))


def _expect_pointer_arg(arr, param : ir.Param) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Argument for pointer parameter '{param.name}' must be a numpy array, "
            f"got {type(arr).__name__}"
        )
    expected = param.dtype
    actual = dt.from_numpy(arr.dtype)
    if actual != expected:
        raise TypeError(
            f"Argument for pointer parameter '{param.name}' has dtype {actual.name}, "
            f"expected {expected.name}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"Argument for pointer parameter '{param.name}' must be C-contiguous"
        )
    return arr


_SCALAR_NUMPY = {
    dt.float32 : np.float32,
    dt.float16 : np.float16,
    dt.int32   : np.int32,
    dt.uint32  : np.uint32,
    dt.int64   : np.int64,
    dt.uint64  : np.uint64,
    dt.bool_   : np.int32,  # promote bool to int32 for ABI sanity
}


def _scalar_dtype(d : dt.Dtype):
    if d not in _SCALAR_NUMPY:
        raise TypeError(f"No numpy dtype for thork scalar dtype {d.name}")
    return _SCALAR_NUMPY[d]


def dispatch(
    func,
    params           : Sequence[ir.Param],
    user_args        : Sequence,
    grid_size        : tuple,
    block_size       : tuple,
    shared_mem_bytes : int = 0,
    stream           : int = 0,
) -> None:
    """
    Allocate device buffers for pointer arguments, copy inputs, launch
    the kernel, synchronize, and copy back any pointer args that the
    traced kernel wrote.
    """
    get_device()

    bindable = [p for p in params if p.kind in ("pointer", "constant")]
    if len(user_args) != len(bindable):
        raise TypeError(
            f"Kernel expected {len(bindable)} arguments "
            f"(pointer + constant params), got {len(user_args)}"
        )

    # bindings: list of (dev_ptr_or_None, numpy_array_or_None, param)
    bindings : List[tuple] = []
    arg_iter = iter(user_args)
    # ``param_storage`` holds the per-parameter scalar/pointer value as a
    # one-element numpy array so its address stays stable across the
    # launch.
    param_storage : List[np.ndarray] = []

    for p in params:
        if p.kind == "pointer":
            arr = _expect_pointer_arg(next(arg_iter), p)
            d_ptr = _check(driver.cuMemAlloc(arr.nbytes))
            _check(driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes))
            param_storage.append(np.array([int(d_ptr)], dtype=np.uint64))
            bindings.append((d_ptr, arr, p))
        elif p.kind == "constant":
            value = next(arg_iter)
            np_dtype = np.dtype(_scalar_dtype(p.dtype))
            param_storage.append(np.asarray(value).astype(np_dtype, copy=True).reshape(1))
            bindings.append((None, None, p))

    arg_ptrs = np.array(
        [param_storage[i].ctypes.data for i in range(len(param_storage))],
        dtype=np.uint64,
    )

    try:
        _check(driver.cuLaunchKernel(
            func,
            int(grid_size[0]), int(grid_size[1]), int(grid_size[2]),
            int(block_size[0]), int(block_size[1]), int(block_size[2]),
            int(shared_mem_bytes),
            stream,
            arg_ptrs.ctypes.data,
            0,
        ))
        _check(driver.cuCtxSynchronize())

        for d_ptr, arr, p in bindings:
            if arr is None or not p.written:
                continue
            _check(driver.cuMemcpyDtoH(arr.ctypes.data, d_ptr, arr.nbytes))
    finally:
        for d_ptr, _arr, _p in bindings:
            if d_ptr is not None:
                driver.cuMemFree(d_ptr)

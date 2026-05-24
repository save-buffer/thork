import ctypes
import os
import re
import shutil
import struct
import subprocess
import tempfile
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


_THORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def kittens_include_dirs() -> List[str]:
    """
    Return the include directories needed to compile a ThunderKittens kernel.

    Checks ``$THORK_KITTENS_PATH`` first, then the vendored submodule at
    ``third_party/ThunderKittens/include`` relative to the thork repo.
    """
    dirs : List[str] = []
    env = os.environ.get("THORK_KITTENS_PATH")
    if env and os.path.isdir(env):
        dirs.append(env)
        return dirs
    vendored = os.path.join(_THORK_ROOT, "third_party", "ThunderKittens", "include")
    if os.path.isdir(vendored):
        dirs.append(vendored)
    return dirs


def kittens_sm_define_for_cap(cap : str) -> str:
    """
    Map a compute-capability string (e.g. ``"100"``, ``"100a"``, ``"120"``)
    to the matching ``KITTENS_SMxxx`` preprocessor define. The trailing
    ``a`` (acceleration variant) is stripped for the purpose of picking
    a TK branch.
    """
    digits = cap.rstrip("a")
    major = int(digits[0]) if len(digits) <= 2 else int(digits[:-1])
    if major >= 12:
        return "KITTENS_SM120"
    if major >= 10:
        return "KITTENS_SM100"
    if major == 9:
        return "KITTENS_SM90"
    return "KITTENS_SM80"


def kittens_sm_define() -> str:
    """
    Return the ``KITTENS_SMxxx`` preprocessor define matching this device.

    TK exposes per-architecture branches under ``KITTENS_SM80`` (Ampere),
    ``KITTENS_SM90`` (Hopper), ``KITTENS_SM100`` / ``KITTENS_SM103``
    (datacenter Blackwell), and ``KITTENS_SM120`` (consumer Blackwell /
    GB10).
    """
    return kittens_sm_define_for_cap(get_compute_capability())


_max_shared_size_per_block : Optional[int] = None


def max_dynamic_shared_size() -> int:
    """
    Largest dynamic shared-memory size (bytes) the device will allow per
    block, queried once and cached. Used to size TK kernels' shared pool.
    """
    global _max_shared_size_per_block
    if _max_shared_size_per_block is not None:
        return _max_shared_size_per_block
    dev = get_device()
    val = _check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        dev,
    ))
    _max_shared_size_per_block = int(val) - 1024
    return _max_shared_size_per_block


def set_max_dynamic_shared_size(func) -> None:
    """
    Opt the given function into the device's full dynamic shared-memory
    budget (minus a small reserve). TK kernels routinely exceed the
    default 48 KiB cap.
    """
    size = max_dynamic_shared_size()
    _check(driver.cuFuncSetAttribute(
        func,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        size,
    ))


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


def _find_nvcc() -> Optional[str]:
    """
    Locate the ``nvcc`` binary. ``$CUDA_HOME``/``$CUDA_PATH`` win; otherwise
    fall back to ``$PATH`` and standard install locations.
    """
    for envvar in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(envvar)
        if root:
            cand = os.path.join(root, "bin", "nvcc")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    via_path = shutil.which("nvcc")
    if via_path:
        return via_path
    for cand in ("/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _run_nvcc(
    source       : str,
    output_path  : str,
    mode_args    : Sequence[str],
    *,
    source_map   : Optional[Dict[int, Tuple[str, int]]] = None,
    extra_opts   : Optional[Sequence[str]] = None,
    include_dirs : Optional[Sequence[str]] = None,
    defines      : Optional[Sequence[str]] = None,
    arch         : Optional[str] = None,
) -> None:
    """
    Invoke ``nvcc`` against ``source`` writing to ``output_path`` with the
    given ``mode_args`` (e.g. ``--cubin`` or ``-shared -Xcompiler -fPIC``).
    ``arch`` overrides the auto-detected compute capability — useful for
    cross-compiling against an arch the host device doesn't expose.
    """
    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found; set $CUDA_HOME or install the CUDA toolkit so "
            "thork can compile ThunderKittens kernels."
        )

    cap = arch if arch is not None else get_compute_capability()
    if cap.endswith("a"):
        arch_flags = [
            f"-gencode=arch=compute_{cap},code=sm_{cap}",
        ]
    else:
        arch_flags = [f"--gpu-architecture=sm_{cap}"]
    args : List[str] = [
        nvcc,
        *arch_flags,
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-Xcompiler", "-fno-strict-aliasing",
    ]
    args += list(mode_args)
    for d in _default_include_dirs():
        args += ["-I", d]
    if include_dirs:
        for d in include_dirs:
            args += ["-I", d]
    if defines:
        for d in defines:
            args.append(f"-D{d}")
    if extra_opts:
        args += list(extra_opts)

    src_dir = os.path.dirname(output_path)
    src_path = os.path.join(src_dir, "thork_kernel.cu")
    with open(src_path, "w") as f:
        f.write(source)
    args += [src_path, "-o", output_path]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        rewritten = _rewrite_compile_error(
            (proc.stderr or "") + (proc.stdout or ""), source_map,
        )
        raise RuntimeError(
            f"nvcc compile failed:\n{rewritten}\n\nSource:\n{source}"
        )


def compile_source_with_nvcc(
    source       : str,
    *,
    source_map   : Optional[Dict[int, Tuple[str, int]]] = None,
    extra_opts   : Optional[Sequence[str]] = None,
    include_dirs : Optional[Sequence[str]] = None,
    defines      : Optional[Sequence[str]] = None,
):
    """
    Compile ``source`` to a .cubin via nvcc and load it via the driver
    API. Used for plain (non-TK) CUDA kernels that still need libstdc++
    headers nvrtc can't reach.
    """
    get_device()
    with tempfile.TemporaryDirectory(prefix="thork_") as tmp:
        cubin_path = os.path.join(tmp, "thork_kernel.cubin")
        _run_nvcc(
            source, cubin_path, ["--cubin"],
            source_map=source_map, extra_opts=extra_opts,
            include_dirs=include_dirs, defines=defines,
        )
        with open(cubin_path, "rb") as f:
            cubin = f.read()
    return _check(driver.cuModuleLoadData(cubin))


_so_cache_dir : Optional[str] = None


def compile_source_to_shared_lib(
    source       : str,
    *,
    source_map   : Optional[Dict[int, Tuple[str, int]]] = None,
    extra_opts   : Optional[Sequence[str]] = None,
    include_dirs : Optional[Sequence[str]] = None,
    defines      : Optional[Sequence[str]] = None,
    arch         : Optional[str] = None,
    load         : bool = True,
):
    """
    Compile ``source`` to a host shared library via nvcc. Loads it via
    ctypes by default; pass ``load=False`` to verify it builds without
    dlopen'ing (useful for cross-compiling against arches the host can't
    run, e.g. SM_100 tcgen05 kernels on an SM_120 dev box).
    """
    get_device()
    global _so_cache_dir
    if _so_cache_dir is None:
        _so_cache_dir = tempfile.mkdtemp(prefix="thork_so_")
    so_path = tempfile.mktemp(prefix="thork_kernel_", suffix=".so", dir=_so_cache_dir)
    _run_nvcc(
        source, so_path,
        ["-shared", "-Xcompiler", "-fPIC", "-lcuda", "-lcudart"],
        source_map=source_map, extra_opts=extra_opts,
        include_dirs=include_dirs, defines=defines, arch=arch,
    )
    if not load:
        return None
    return ctypes.CDLL(so_path)


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


def _expect_kittens_arg(arr, param : ir.Param) -> np.ndarray:
    """
    Validate an argument bound to a ``kittens.Global[T]`` parameter.

    Requires a 2-D C-contiguous numpy array whose element size matches
    the parameter dtype. The numpy dtype itself is allowed to differ
    (e.g. uint16 storage for bf16 buffers when ml_dtypes is unavailable).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Argument for kittens.Global parameter '{param.name}' must be a "
            f"numpy array, got {type(arr).__name__}"
        )
    if arr.ndim != 2:
        raise TypeError(
            f"kittens.Global parameter '{param.name}' requires a 2-D numpy "
            f"array, got shape {arr.shape}"
        )
    if arr.dtype.itemsize != param.dtype.nbytes:
        raise TypeError(
            f"Argument for kittens.Global parameter '{param.name}' has "
            f"{arr.dtype.itemsize}-byte elements; expected {param.dtype.nbytes} "
            f"to match {param.dtype.name}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"Argument for kittens.Global parameter '{param.name}' must be "
            "C-contiguous"
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


def _pack_kittens_gl(d_ptr : int, rows : int, cols : int) -> np.ndarray:
    """
    Pack a ``kittens::gl<T, 1, 1, -1, -1>`` struct (40 bytes, align 8) for
    pass-by-value through ``cuLaunchKernel``.

    Layout (verified on aarch64 + CUDA 13):
      - bytes 0-7  : T* raw_ptr
      - bytes 8-9  : compiled_dim<1> batch_internal / depth_internal (1 byte each)
      - bytes 10-15: padding to 8-byte alignment
      - bytes 16-23: runtime_dim rows_internal (size_t)
      - bytes 24-31: runtime_dim cols_internal (size_t)
      - bytes 32-39: descriptor_dict<> tma_descs (empty, trailing pad)
    """
    buf = struct.pack(
        "<Q1B1B6xQQ8x",
        int(d_ptr),
        1, 1,
        int(rows), int(cols),
    )
    assert len(buf) == 40
    return np.frombuffer(buf, dtype=np.uint8).copy()


_CTYPES_SCALAR = {
    dt.float32 : ctypes.c_float,
    dt.float16 : ctypes.c_uint16,
    dt.int32   : ctypes.c_int32,
    dt.uint32  : ctypes.c_uint32,
    dt.int64   : ctypes.c_int64,
    dt.uint64  : ctypes.c_uint64,
    dt.bool_   : ctypes.c_int32,
}


def dispatch_tk(
    lib              : ctypes.CDLL,
    kernel_name      : str,
    params           : Sequence[ir.Param],
    user_args        : Sequence,
    grid_size        : tuple,
    block_size       : tuple,
    shared_mem_bytes : int,
    stream           : int = 0,
) -> None:
    """
    Launch a ThunderKittens kernel via its generated host launcher.

    The launcher takes raw device pointers (+ rows/cols for each
    ``kittens_global``) and handles gl construction + ``<<<>>>`` itself.
    Python's role is purely to manage device memory and unpack args
    through ctypes.
    """
    get_device()

    bindable = [p for p in params if p.kind in ("pointer", "constant", "kittens_global")]
    if len(user_args) != len(bindable):
        raise TypeError(
            f"TK kernel '{kernel_name}' expected {len(bindable)} arguments, "
            f"got {len(user_args)}"
        )

    bindings : List[tuple] = []
    arg_iter = iter(user_args)
    c_args : List = []
    argtypes : List = [
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
    ]
    c_args += [
        ctypes.c_uint(int(grid_size[0])), ctypes.c_uint(int(grid_size[1])), ctypes.c_uint(int(grid_size[2])),
        ctypes.c_uint(int(block_size[0])), ctypes.c_uint(int(block_size[1])), ctypes.c_uint(int(block_size[2])),
        ctypes.c_uint(int(shared_mem_bytes)),
        ctypes.c_void_p(int(stream)),
    ]

    for p in params:
        if p.kind == "pointer":
            arr = _expect_pointer_arg(next(arg_iter), p)
            d_ptr = _check(driver.cuMemAlloc(arr.nbytes))
            _check(driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes))
            argtypes.append(ctypes.c_void_p)
            c_args.append(ctypes.c_void_p(int(d_ptr)))
            bindings.append((d_ptr, arr, p))
        elif p.kind == "kittens_global":
            arr = _expect_kittens_arg(next(arg_iter), p)
            d_ptr = _check(driver.cuMemAlloc(arr.nbytes))
            _check(driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes))
            rows, cols = int(arr.shape[0]), int(arr.shape[1])
            argtypes += [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            c_args += [
                ctypes.c_void_p(int(d_ptr)),
                ctypes.c_size_t(rows),
                ctypes.c_size_t(cols),
            ]
            bindings.append((d_ptr, arr, p))
        elif p.kind == "constant":
            value = next(arg_iter)
            c_type = _CTYPES_SCALAR.get(p.dtype)
            if c_type is None:
                raise TypeError(
                    f"No ctypes mapping for constant scalar dtype {p.dtype.name}"
                )
            argtypes.append(c_type)
            c_args.append(c_type(value))
            bindings.append((None, None, p))

    launcher = getattr(lib, f"thork_launch_{kernel_name}")
    launcher.argtypes = argtypes
    launcher.restype = None

    try:
        launcher(*c_args)
        _check(driver.cuCtxSynchronize())

        for d_ptr, arr, p in bindings:
            if arr is None or not p.written:
                continue
            _check(driver.cuMemcpyDtoH(arr.ctypes.data, d_ptr, arr.nbytes))
    finally:
        for d_ptr, _arr, _p in bindings:
            if d_ptr is not None:
                driver.cuMemFree(d_ptr)


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
    Allocate device buffers for pointer / kittens-global arguments, copy
    inputs, launch the kernel, synchronize, and copy back any pointer args
    that the traced kernel wrote.
    """
    get_device()

    bindable = [p for p in params if p.kind in ("pointer", "constant", "kittens_global")]
    if len(user_args) != len(bindable):
        raise TypeError(
            f"Kernel expected {len(bindable)} arguments "
            f"(pointer + constant + kittens_global params), got {len(user_args)}"
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
        elif p.kind == "kittens_global":
            arr = _expect_kittens_arg(next(arg_iter), p)
            d_ptr = _check(driver.cuMemAlloc(arr.nbytes))
            _check(driver.cuMemcpyHtoD(d_ptr, arr.ctypes.data, arr.nbytes))
            rows, cols = int(arr.shape[0]), int(arr.shape[1])
            param_storage.append(_pack_kittens_gl(int(d_ptr), rows, cols))
            bindings.append((d_ptr, arr, p))

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

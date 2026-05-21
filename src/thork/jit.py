import inspect
from typing import Callable, Optional

from . import ir
from . import runtime
from . import tracer as _tracer
from .codegen import emit_kernel
from .tracer import KernelBuilder, PointerTracer, Tracer, VectorTracer
from .types import DevicePointerSpec, ScalarParamSpec, _ScalarTypeBase


def _annotation_to_spec(ann, kernel_name : str, param_name : str):
    """
    Normalize a parameter annotation into a DevicePointerSpec or ScalarParamSpec.

    A bare scalar class (e.g. ``tk.Uint``) becomes a by-value constant
    ScalarParamSpec. A subscripted form (e.g. ``tk.Uint3[BlockIdx]``) is
    already a ScalarParamSpec.
    """
    if isinstance(ann, DevicePointerSpec):
        return ann
    if isinstance(ann, ScalarParamSpec):
        return ann
    if inspect.isclass(ann) and issubclass(ann, _ScalarTypeBase):
        return ScalarParamSpec(
            dtype=ann._dtype,
            cuda_name=ann._cuda_name,
            vec_size=ann._vec_size,
            attribute=None,
        )
    raise TypeError(
        f"Kernel '{kernel_name}': parameter '{param_name}' has unsupported "
        f"annotation {ann!r}. Expected tk.DevicePointer[...] or tk.Uint[...] etc."
    )


class JittedKernel:
    """
    A function decorated with ``@tk.jit``.

    Calling ``kernel[grid, block](*args)`` traces (on first use), compiles
    via nvrtc, and dispatches the kernel.
    """

    def __init__(self, fn : Callable, name : Optional[str] = None):
        self._fn = fn
        self.name : str = name or fn.__name__
        self._sig = inspect.signature(fn)
        self._builder : Optional[KernelBuilder] = None
        self._source : Optional[str] = None
        self._source_map : Optional[dict] = None
        self._module = None
        self._func   = None

    def _trace(self) -> KernelBuilder:
        builder = KernelBuilder(self.name)
        tracer_args = []
        for param_name, param in self._sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"Kernel '{self.name}': *args/**kwargs not supported"
                )
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"Kernel '{self.name}': parameter '{param_name}' is missing a type annotation"
                )
            spec = _annotation_to_spec(ann, self.name, param_name)
            if isinstance(spec, DevicePointerSpec):
                builder.params.append(ir.Param(
                    name=param_name,
                    kind="pointer",
                    dtype=spec.dtype,
                ))
                tracer_args.append(PointerTracer(ir.Var(param_name), spec.dtype, builder))
            else:
                kind = "attribute" if spec.attribute is not None else "constant"
                builder.params.append(ir.Param(
                    name=param_name,
                    kind=kind,
                    dtype=spec.dtype,
                    cuda_name=spec.cuda_name,
                    vec_size=spec.vec_size,
                    attribute=spec.attribute,
                ))
                if spec.vec_size > 1:
                    tracer_args.append(VectorTracer(
                        ir.Var(param_name), spec.dtype, spec.vec_size,
                    ))
                else:
                    tracer_args.append(Tracer(ir.Var(param_name), spec.dtype))

        token = _tracer._builder.set(builder)
        try:
            self._fn(*tracer_args)
        finally:
            _tracer._builder.reset(token)

        return builder

    def _ensure_compiled(self):
        if self._func is not None:
            return
        self._builder = self._trace()
        self._source, self._source_map = emit_kernel(self._builder)
        self._module = runtime.compile_source(self._source, source_map=self._source_map)
        self._func = runtime.get_function(self._module, self.name)

    def __getitem__(self, dispatch_spec) -> "_Launcher":
        if not (isinstance(dispatch_spec, tuple) and len(dispatch_spec) == 2):
            raise TypeError(
                "Kernel must be subscripted with (grid_size, block_size), "
                f"got {dispatch_spec!r}"
            )
        grid_size, block_size = dispatch_spec
        return _Launcher(self, grid_size, block_size)

    def bind(self, grid, block) -> "BoundKernel":
        """
        Return a BoundKernel that calls this kernel with ``grid`` and
        ``block`` already baked in. Eliminates the
        ``kernel[grid, block](*args)`` boilerplate at each call site.
        """
        return BoundKernel(self, grid, block)

    @property
    def cuda_source(self) -> str:
        if self._source is None:
            self._builder = self._trace()
            self._source, self._source_map = emit_kernel(self._builder)
        return self._source

    @property
    def source_map(self) -> Optional[dict]:
        """
        Mapping of generated-CUDA line number (1-indexed) to a
        ``(python_filename, python_lineno)`` tuple.
        """
        if self._source is None:
            _ = self.cuda_source
        return self._source_map


class _Launcher:
    __slots__ = ("_kernel", "_grid_size", "_block_size")

    def __init__(self, kernel : JittedKernel, grid_size, block_size):
        self._kernel = kernel
        self._grid_size  = tuple(grid_size)
        self._block_size = tuple(block_size)

    def __call__(self, *args):
        self._kernel._ensure_compiled()
        runtime.dispatch(
            self._kernel._func,
            self._kernel._builder.params,
            args,
            self._grid_size,
            self._block_size,
        )


def jit(fn : Callable) -> JittedKernel:
    """
    Decorator that turns a Python function into a CUDA kernel.

    The function body is traced once on first launch using the type
    annotations on its parameters; the resulting CUDA source is compiled
    with nvrtc and dispatched via the driver API.
    """
    return JittedKernel(fn)


class BoundKernel:
    """
    A JittedKernel with its dispatch ``grid`` and ``block`` already
    bound. Call it directly with the kernel's pointer + constant arguments.
    """

    __slots__ = ("_kernel", "_grid", "_block")

    def __init__(self, kernel : JittedKernel, grid, block):
        self._kernel = kernel
        self._grid  = tuple(int(x) for x in grid)
        self._block = tuple(int(x) for x in block)

    def __call__(self, *args):
        self._kernel[self._grid, self._block](*args)

    @property
    def cuda_source(self) -> str:
        return self._kernel.cuda_source

    @property
    def source_map(self):
        return self._kernel.source_map

    @property
    def grid(self) -> tuple:
        return self._grid

    @property
    def block(self) -> tuple:
        return self._block

    @property
    def name(self) -> str:
        return self._kernel.name

import contextvars
import inspect
import os
import sys
from typing import Callable, Dict, List, Optional

from . import dtypes as dt
from . import ir
from .types import DevicePointerSpec


# Preserve a reference to the builtin ``range`` since this module also
# defines ``range`` (the thork kernel-loop primitive). Any Python-level for
# loops inside this module must use ``_py_range``.
_py_range = range


_THORK_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _caller_loc() -> Optional[tuple]:
    """
    Walk up the call stack and return the (filename, lineno) of the first
    frame that lives outside the thork package directory.

    Returns None if the entire stack is internal.
    """
    frame = sys._getframe(1)
    while frame is not None:
        fname = frame.f_code.co_filename
        if not fname.startswith(_THORK_PACKAGE_DIR):
            return (fname, frame.f_lineno)
        frame = frame.f_back
    return None


_builder : contextvars.ContextVar[Optional["KernelBuilder"]] = contextvars.ContextVar(
    "_thork_builder", default=None
)


def current_builder() -> "KernelBuilder":
    b = _builder.get()
    if b is None:
        raise RuntimeError(
            "thork operations may only be used inside a @tk.jit-decorated kernel"
        )
    return b


class KernelBuilder:
    def __init__(self, name : str):
        self.name : str = name
        self.params : List[ir.Param] = []
        self.stmts : List[ir.Stmt] = []
        # nvrtc recognizes __global__/threadIdx/etc. intrinsically — no
        # cuda_runtime.h needed. cuda_fp16.h / cuda_bf16.h are pulled in
        # unconditionally so half / bfloat16 types Just Work; nvrtc gets
        # an explicit ``-I<cuda>/include`` flag from the runtime.
        self.includes : List[tuple] = [
            ("cuda_fp16.h", True),
            ("cuda_bf16.h", True),
        ]
        self.usings : List[str] = []
        self.device_functions : List["DeviceFn"] = []
        self._dfn_names : set = set()
        self._name_counters : Dict[str, int] = {}

    def add_stmt(self, stmt : ir.Stmt) -> None:
        if getattr(stmt, "loc", None) is None:
            stmt.loc = _caller_loc()
        self.stmts.append(stmt)

    def fresh_name(self, prefix : str = "v") -> str:
        n = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = n + 1
        return f"{prefix}{n}"

    def add_include(self, path : str, system : bool = True) -> None:
        entry = (path, system)
        if entry not in self.includes:
            self.includes.append(entry)

    def add_using(self, namespace : str) -> None:
        if namespace not in self.usings:
            self.usings.append(namespace)

    def add_device_fn(self, df : "DeviceFn") -> None:
        """
        Register a DeviceFn as a dependency of this kernel. Traces it
        lazily if needed and recursively pulls in transitive dependencies.
        Also merges the device fn's includes/usings into this builder.
        """
        df._ensure_traced()
        for dep in df._deps:
            self.add_device_fn(dep)
        if df._name in self._dfn_names:
            return
        self._dfn_names.add(df._name)
        self.device_functions.append(df)
        for inc in df._includes:
            if inc not in self.includes:
                self.includes.append(inc)
        for ns in df._usings:
            if ns not in self.usings:
                self.usings.append(ns)


def _to_expr(value) -> ir.Expr:
    if isinstance(value, Tracer):
        return value._expr
    if isinstance(value, VectorTracer):
        raise TypeError(
            "Cannot use a vector value as a scalar — access a field like .x first"
        )
    if isinstance(value, PointerTracer):
        raise TypeError(
            "Cannot use a pointer as a scalar value — index it with ptr[i]"
        )
    if isinstance(value, bool):
        return ir.Const(value)
    if isinstance(value, (int, float)):
        return ir.Const(value)
    raise TypeError(
        f"Cannot convert {value!r} (type {type(value).__name__}) to a thork expression"
    )


def _result_dtype(a : "Tracer", other) -> Optional[dt.Dtype]:
    if isinstance(other, Tracer):
        if a._dtype is not None and other._dtype is not None:
            if a._dtype.is_float or other._dtype.is_float:
                return a._dtype if a._dtype.is_float else other._dtype
            return a._dtype
        return a._dtype or other._dtype
    return a._dtype


class Tracer:
    """
    Represents a scalar value in a traced kernel.
    """

    __slots__ = ("_expr", "_dtype")

    def __init__(self, expr : ir.Expr, dtype : Optional[dt.Dtype]):
        self._expr = expr
        self._dtype = dtype

    def _binop(self, op : str, other, *, swap : bool = False) -> "Tracer":
        rhs = _to_expr(other)
        if swap:
            expr = ir.BinOp(op, rhs, self._expr)
        else:
            expr = ir.BinOp(op, self._expr, rhs)
        return Tracer(expr, _result_dtype(self, other))

    def _cmp(self, op : str, other) -> "Tracer":
        rhs = _to_expr(other)
        return Tracer(ir.BinOp(op, self._expr, rhs), dt.bool_)

    def __add__(self, other):      return self._binop("+", other)
    def __radd__(self, other):     return self._binop("+", other, swap=True)
    def __sub__(self, other):      return self._binop("-", other)
    def __rsub__(self, other):     return self._binop("-", other, swap=True)
    def __mul__(self, other):      return self._binop("*", other)
    def __rmul__(self, other):     return self._binop("*", other, swap=True)
    def __truediv__(self, other):  return self._binop("/", other)
    def __rtruediv__(self, other): return self._binop("/", other, swap=True)
    def __floordiv__(self, other): return self._binop("/", other)
    def __rfloordiv__(self, other):return self._binop("/", other, swap=True)
    def __mod__(self, other):      return self._binop("%", other)
    def __rmod__(self, other):     return self._binop("%", other, swap=True)
    def __and__(self, other):      return self._binop("&", other)
    def __rand__(self, other):     return self._binop("&", other, swap=True)
    def __or__(self, other):       return self._binop("|", other)
    def __ror__(self, other):      return self._binop("|", other, swap=True)
    def __xor__(self, other):      return self._binop("^", other)
    def __rxor__(self, other):     return self._binop("^", other, swap=True)
    def __lshift__(self, other):   return self._binop("<<", other)
    def __rshift__(self, other):   return self._binop(">>", other)

    def __lt__(self, other): return self._cmp("<", other)
    def __le__(self, other): return self._cmp("<=", other)
    def __gt__(self, other): return self._cmp(">", other)
    def __ge__(self, other): return self._cmp(">=", other)
    def __eq__(self, other): return self._cmp("==", other)
    def __ne__(self, other): return self._cmp("!=", other)

    def __neg__(self):    return Tracer(ir.UnaryOp("-", self._expr), self._dtype)
    def __pos__(self):    return self
    def __invert__(self): return Tracer(ir.UnaryOp("~", self._expr), self._dtype)

    def __bool__(self):
        raise TypeError(
            "Cannot use a traced thork value in a Python boolean context "
            "(e.g. `if`, `and`, `or`). Use kernel-level control flow instead."
        )

    def __hash__(self):
        return id(self)


class PointerTracer:
    """
    Represents a device pointer in a traced kernel.
    """

    __slots__ = ("_expr", "_dtype", "_builder")

    def __init__(self, expr : ir.Expr, dtype : dt.Dtype, builder : KernelBuilder):
        self._expr = expr
        self._dtype = dtype
        self._builder = builder

    def __getitem__(self, index) -> Tracer:
        idx = _to_expr(index)
        return Tracer(ir.Load(self._expr, idx), self._dtype)

    def __setitem__(self, index, value) -> None:
        idx = _to_expr(index)
        val = _to_expr(value)
        self._builder.add_stmt(ir.Store(self._expr, idx, val))
        if isinstance(self._expr, ir.Var):
            for p in self._builder.params:
                if p.kind == "pointer" and p.name == self._expr.name:
                    p.written = True
                    break


class VectorTracer:
    """
    Represents a vector-typed value (uint2/uint3/int2/...) in a traced kernel.

    Component access (``v.x``, ``v.y``, ``v.z``, ``v.w``) yields a scalar Tracer.
    """

    __slots__ = ("_expr", "_dtype", "_vec_size")

    _FIELDS = ("x", "y", "z", "w")

    def __init__(self, expr : ir.Expr, elem_dtype : dt.Dtype, vec_size : int):
        self._expr = expr
        self._dtype = elem_dtype
        self._vec_size = vec_size

    def __getattr__(self, name : str) -> Tracer:
        if name in self._FIELDS:
            idx = self._FIELDS.index(name)
            if idx >= self._vec_size:
                raise AttributeError(
                    f"Vector of size {self._vec_size} has no field .{name}"
                )
            return Tracer(ir.Member(self._expr, name), self._dtype)
        raise AttributeError(name)


class Local(Tracer):
    """
    A mutable local variable declared with ``tk.local``.

    Reads behave like a normal scalar Tracer. Compound updates
    (``local += x``, ``local -= x``, ...) emit an Update statement.
    """

    __slots__ = ("_builder", "_name")

    def __init__(self, name : str, dtype : dt.Dtype, builder : KernelBuilder):
        super().__init__(ir.Var(name), dtype)
        self._builder = builder
        self._name = name

    def _update(self, op : str, value) -> "Local":
        self._builder.add_stmt(ir.Update(self._name, op, _to_expr(value)))
        return self

    def __iadd__(self, other):      return self._update("+=", other)
    def __isub__(self, other):      return self._update("-=", other)
    def __imul__(self, other):      return self._update("*=", other)
    def __itruediv__(self, other):  return self._update("/=", other)
    def __ifloordiv__(self, other): return self._update("/=", other)
    def __imod__(self, other):      return self._update("%=", other)
    def __iand__(self, other):      return self._update("&=", other)
    def __ior__(self, other):       return self._update("|=", other)
    def __ixor__(self, other):      return self._update("^=", other)
    def __ilshift__(self, other):   return self._update("<<=", other)
    def __irshift__(self, other):   return self._update(">>=", other)

    def assign(self, value) -> None:
        """
        Assign a new value to this local: ``self = value;``.
        """
        self._builder.add_stmt(ir.Update(self._name, "=", _to_expr(value)))


def local(dtype : dt.Dtype, init) -> Local:
    """
    Declare a mutable local variable in the current kernel.

    Emits ``<dtype> name = <init>;`` and returns a Local handle that supports
    compound assignment (``+=``, ``-=``, ...) and ``.assign(value)``.
    """
    builder = current_builder()
    name = builder.fresh_name("v")
    init_expr = _to_expr(init)
    builder.add_stmt(ir.Assign(name, dtype.cuda, init_expr))
    return Local(name, dtype, builder)


def range(*args):
    """
    Emit a ``for`` loop. Usage:

        for i in tk.range(end): ...
        for i in tk.range(start, end): ...
        for i in tk.range(start, end, step): ...

    Bounds and step may be Python ints or thork Tracers.
    """
    if len(args) == 1:
        start_expr = ir.Const(0)
        end_expr = _to_expr(args[0])
        step_expr = ir.Const(1)
    elif len(args) == 2:
        start_expr = _to_expr(args[0])
        end_expr = _to_expr(args[1])
        step_expr = ir.Const(1)
    elif len(args) == 3:
        start_expr = _to_expr(args[0])
        end_expr = _to_expr(args[1])
        step_expr = _to_expr(args[2])
    else:
        raise TypeError(f"tk.range expects 1-3 arguments, got {len(args)}")
    return _RangeLoop(start_expr, end_expr, step_expr)


class _RangeLoop:
    __slots__ = ("_start", "_end", "_step")

    def __init__(self, start : ir.Expr, end : ir.Expr, step : ir.Expr):
        self._start = start
        self._end = end
        self._step = step

    def __iter__(self):
        builder = current_builder()
        loop_var = builder.fresh_name("i")
        saved_stmts = builder.stmts
        builder.stmts = []
        try:
            yield Tracer(ir.Var(loop_var), dt.uint32)
        finally:
            body = builder.stmts
            builder.stmts = saved_stmts
            builder.add_stmt(ir.ForLoop(
                var_name=loop_var,
                start=self._start,
                end=self._end,
                step=self._step,
                body=body,
            ))


class SharedArray:
    """
    A shared-memory array. Supports both single-axis and tuple subscripts:

        a[i]        # 1-D
        a[i, j]     # 2-D, sugar for a[i][j]
        a[i, j, k]  # 3-D, ...
    """

    __slots__ = ("_name", "_dtype", "_shape", "_builder", "_expr")

    def __init__(
        self,
        name    : str,
        dtype   : dt.Dtype,
        shape   : tuple,
        builder : KernelBuilder,
    ):
        self._name = name
        self._dtype = dtype
        self._shape = shape
        self._builder = builder
        self._expr = ir.Var(name)

    def _indices(self, index):
        if isinstance(index, tuple):
            return tuple(_to_expr(i) for i in index)
        return (_to_expr(index),)

    def __getitem__(self, index) -> Tracer:
        indices = self._indices(index)
        expr : ir.Expr = self._expr
        for idx in indices:
            expr = ir.Load(expr, idx)
        return Tracer(expr, self._dtype)

    def __setitem__(self, index, value) -> None:
        indices = self._indices(index)
        *outer, last = indices
        ptr_expr : ir.Expr = self._expr
        for idx in outer:
            ptr_expr = ir.Load(ptr_expr, idx)
        self._builder.add_stmt(ir.Store(ptr_expr, last, _to_expr(value)))


def shared(dtype : dt.Dtype, shape) -> SharedArray:
    """
    Declare a shared-memory array of the given fixed shape.

    Shape elements must be Python ints (CUDA requires constexpr sizes for
    statically allocated shared memory).
    """
    builder = current_builder()
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            raise TypeError(
                f"shared array shape must be positive Python ints, got {shape!r}"
            )
    name = builder.fresh_name("s")
    builder.add_stmt(ir.SharedDecl(
        name=name,
        cuda_type=dtype.cuda,
        shape=list(shape),
    ))
    return SharedArray(name, dtype, shape, builder)


def syncthreads() -> None:
    """
    Emit ``__syncthreads();`` (block-wide barrier).
    """
    current_builder().add_stmt(ir.ExprStmt(ir.Call("__syncthreads", [])))


def syncwarp(mask : int = 0xFFFFFFFF) -> None:
    """
    Emit ``__syncwarp(mask);`` (warp-level barrier).
    """
    current_builder().add_stmt(ir.ExprStmt(ir.Call("__syncwarp", [ir.Const(int(mask))])))


def _warp_op(func : str, *args, result_dtype : Optional[dt.Dtype] = None) -> Tracer:
    """
    Emit a warp-level intrinsic as ``<T> warpN = func(args...);`` at the
    current statement position.

    Materializing the result into a local prevents subsequent uses (e.g.
    inside an ``if`` block) from re-inlining the call and putting the
    collective into divergent control flow.
    """
    builder = current_builder()
    arg_exprs = [_to_expr(a) for a in args]
    if result_dtype is not None:
        dtype = result_dtype
    else:
        first = args[0]
        dtype = first._dtype if isinstance(first, Tracer) else None
        if dtype is None:
            dtype = dt.float32
    name = builder.fresh_name("warp")
    builder.add_stmt(ir.Assign(name, dtype.cuda, ir.Call(func, arg_exprs)))
    return Tracer(ir.Var(name), dtype)


_FULL_MASK = ir.Const(0xFFFFFFFF)


def shfl_sync(x, src_lane, width : int = 32, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__shfl_sync(mask, x, src_lane, width)``.
    """
    return _warp_op(
        "__shfl_sync",
        ir.Const(int(mask)), x, src_lane, ir.Const(int(width)),
    )


def shfl_up_sync(x, delta, width : int = 32, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__shfl_up_sync(mask, x, delta, width)``.
    """
    return _warp_op(
        "__shfl_up_sync",
        ir.Const(int(mask)), x, delta, ir.Const(int(width)),
    )


def shfl_down_sync(x, delta, width : int = 32, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__shfl_down_sync(mask, x, delta, width)``.
    """
    return _warp_op(
        "__shfl_down_sync",
        ir.Const(int(mask)), x, delta, ir.Const(int(width)),
    )


def shfl_xor_sync(x, lane_mask, width : int = 32, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__shfl_xor_sync(mask, x, lane_mask, width)``.
    """
    return _warp_op(
        "__shfl_xor_sync",
        ir.Const(int(mask)), x, lane_mask, ir.Const(int(width)),
    )


def ballot_sync(predicate, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__ballot_sync(mask, predicate)`` → 32-bit mask of lanes for which
    ``predicate`` is true.
    """
    return _warp_op(
        "__ballot_sync",
        ir.Const(int(mask)), predicate,
        result_dtype=dt.uint32,
    )


def all_sync(predicate, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__all_sync(mask, predicate)``.
    """
    return _warp_op(
        "__all_sync",
        ir.Const(int(mask)), predicate,
        result_dtype=dt.int32,
    )


def any_sync(predicate, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    ``__any_sync(mask, predicate)``.
    """
    return _warp_op(
        "__any_sync",
        ir.Const(int(mask)), predicate,
        result_dtype=dt.int32,
    )


def warp_sum(x, mask : int = 0xFFFFFFFF) -> Tracer:
    """
    Reduce ``x`` across the warp by summing via butterfly ``__shfl_xor_sync``.

    Equivalent to ``__reduce_add_sync`` on Ampere+, but the butterfly form
    works on all CUDA-capable GPUs and on any element type with ``+=``.
    """
    builder = current_builder()
    acc_name = builder.fresh_name("warp")
    dtype = x._dtype if isinstance(x, Tracer) and x._dtype is not None else dt.float32
    builder.add_stmt(ir.Assign(acc_name, dtype.cuda, _to_expr(x)))
    offset = 16
    while offset > 0:
        builder.add_stmt(ir.Update(
            acc_name, "+=",
            ir.Call("__shfl_xor_sync", [
                ir.Const(int(mask)),
                ir.Var(acc_name),
                ir.Const(offset),
                ir.Const(32),
            ]),
        ))
        offset //= 2
    return Tracer(ir.Var(acc_name), dtype)


class _IfBlock:
    """
    Context manager produced by ``tk.if_(cond)``.
    """

    __slots__ = ("_cond", "_builder", "_saved", "_if_stmt")

    def __init__(self, cond : ir.Expr):
        self._cond = cond
        self._builder : Optional[KernelBuilder] = None
        self._saved = None
        self._if_stmt : Optional[ir.IfStmt] = None

    def __enter__(self):
        builder = current_builder()
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._if_stmt = ir.IfStmt(cond=self._cond, then_body=body, else_body=None)
        self._builder.add_stmt(self._if_stmt)
        return False

    def else_(self) -> "_ElseBlock":
        """
        Open the matching ``else { ... }`` block. Must immediately follow
        the ``with tk.if_(...)`` block.
        """
        if self._if_stmt is None:
            raise RuntimeError(
                "tk.if_(...).else_() can only be used after the if-block "
                "has been exited."
            )
        return _ElseBlock(self)


class _ElseBlock:
    __slots__ = ("_if_block", "_builder", "_saved")

    def __init__(self, if_block : _IfBlock):
        self._if_block = if_block
        self._builder : Optional[KernelBuilder] = None
        self._saved = None

    def __enter__(self):
        builder = current_builder()
        if not builder.stmts or builder.stmts[-1] is not self._if_block._if_stmt:
            raise RuntimeError(
                "tk.else_ block must immediately follow its matching tk.if_ "
                "block (no statements in between)."
            )
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._if_block._if_stmt.else_body = body
        return False


def if_(cond) -> _IfBlock:
    """
    Emit an ``if (cond) { ... }`` block.

    Usage::

        with tk.if_(thread_idx == 0):
            out[i] = total

    Pair with ``.else_()`` for an else branch::

        with tk.if_(x > 0) as branch:
            out[i] = x
        with branch.else_():
            out[i] = -x
    """
    return _IfBlock(_to_expr(cond))


class _WhileBlock:
    __slots__ = ("_cond", "_builder", "_saved")

    def __init__(self, cond : ir.Expr):
        self._cond = cond
        self._builder : Optional[KernelBuilder] = None
        self._saved = None

    def __enter__(self):
        builder = current_builder()
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._builder.add_stmt(ir.WhileLoop(cond=self._cond, body=body))
        return False


def while_(cond) -> _WhileBlock:
    """
    Emit a ``while (cond) { ... }`` loop.

    The condition is evaluated every iteration, so updates to locals
    referenced in ``cond`` take effect each iteration.
    """
    return _WhileBlock(_to_expr(cond))


def break_() -> None:
    """
    Emit a ``break;`` statement.
    """
    current_builder().add_stmt(ir.Break())


def continue_() -> None:
    """
    Emit a ``continue;`` statement.
    """
    current_builder().add_stmt(ir.Continue())


# ---------------------------------------------------------------------------
# Math intrinsics + cast
# ---------------------------------------------------------------------------


def _math_call(func : str, *args, result_dtype : Optional[dt.Dtype] = None) -> Tracer:
    """
    Pure-expression intrinsic — emits ``func(args...)`` without materializing.
    Result dtype defaults to the first arg's dtype.
    """
    arg_exprs = [_to_expr(a) for a in args]
    if result_dtype is None:
        first = args[0]
        result_dtype = first._dtype if isinstance(first, Tracer) else None
    return Tracer(ir.Call(func, arg_exprs), result_dtype)


def exp(x)   : return _math_call("expf", x)
def exp2(x)  : return _math_call("exp2f", x)
def log(x)   : return _math_call("logf", x)
def log2(x)  : return _math_call("log2f", x)
def log10(x) : return _math_call("log10f", x)
def sqrt(x)  : return _math_call("sqrtf", x)
def rsqrt(x) : return _math_call("rsqrtf", x)
def sin(x)   : return _math_call("sinf", x)
def cos(x)   : return _math_call("cosf", x)
def tan(x)   : return _math_call("tanf", x)
def asin(x)  : return _math_call("asinf", x)
def acos(x)  : return _math_call("acosf", x)
def atan(x)  : return _math_call("atanf", x)
def sinh(x)  : return _math_call("sinhf", x)
def cosh(x)  : return _math_call("coshf", x)
def tanh(x)  : return _math_call("tanhf", x)
def floor(x) : return _math_call("floorf", x)
def ceil(x)  : return _math_call("ceilf", x)
def round(x) : return _math_call("roundf", x)
def trunc(x) : return _math_call("truncf", x)
def fabs(x)  : return _math_call("fabsf", x)
def abs(x)   : return _math_call("abs", x)

def pow(x, y)   : return _math_call("powf", x, y)
def fmod(x, y)  : return _math_call("fmodf", x, y)
def atan2(y, x) : return _math_call("atan2f", y, x)
def fmin(x, y)  : return _math_call("fminf", x, y)
def fmax(x, y)  : return _math_call("fmaxf", x, y)
def min(x, y)   : return _math_call("min", x, y)
def max(x, y)   : return _math_call("max", x, y)

def fma(a, b, c) : return _math_call("fmaf", a, b, c)


def fast_exp(x)   : return _math_call("__expf", x)
def fast_log(x)   : return _math_call("__logf", x)
def fast_log2(x)  : return _math_call("__log2f", x)
def fast_log10(x) : return _math_call("__log10f", x)
def fast_sqrt(x)  : return _math_call("__fsqrt_rn", x)
def fast_rsqrt(x) : return _math_call("__frsqrt_rn", x)
def fast_sin(x)   : return _math_call("__sinf", x)
def fast_cos(x)   : return _math_call("__cosf", x)
def fast_tan(x)   : return _math_call("__tanf", x)
def fast_pow(x, y): return _math_call("__powf", x, y)


def cast(value, dtype : dt.Dtype) -> Tracer:
    """
    Emit ``static_cast<dtype>(value)``.
    """
    return Tracer(ir.Cast(dtype, _to_expr(value)), dtype)


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------


def _atomic_addr(ptr : PointerTracer, idx) -> ir.AddrOf:
    if not isinstance(ptr, PointerTracer):
        raise TypeError(
            "atomic operations expect a tk.DevicePointer, got "
            f"{type(ptr).__name__}"
        )
    return ir.AddrOf(ir.Load(ptr._expr, _to_expr(idx)))


def _mark_pointer_written(ptr : PointerTracer) -> None:
    if isinstance(ptr._expr, ir.Var):
        builder = current_builder()
        for p in builder.params:
            if p.kind == "pointer" and p.name == ptr._expr.name:
                p.written = True
                break


def _atomic_rmw(func : str, ptr : PointerTracer, idx, value) -> Tracer:
    """
    Atomic read-modify-write. Always emits a local that captures the prior
    value, so the side effect happens even if the caller ignores the result.
    """
    builder = current_builder()
    addr = _atomic_addr(ptr, idx)
    val_expr = _to_expr(value)
    call = ir.Call(func, [addr, val_expr])
    cuda_type = ptr._dtype.cuda if ptr._dtype is not None else "auto"
    name = builder.fresh_name("atom")
    builder.add_stmt(ir.Assign(name, cuda_type, call))
    _mark_pointer_written(ptr)
    return Tracer(ir.Var(name), ptr._dtype)


def atomic_add(ptr, idx, value)      : return _atomic_rmw("atomicAdd", ptr, idx, value)
def atomic_sub(ptr, idx, value)      : return _atomic_rmw("atomicSub", ptr, idx, value)
def atomic_and(ptr, idx, value)      : return _atomic_rmw("atomicAnd", ptr, idx, value)
def atomic_or(ptr, idx, value)       : return _atomic_rmw("atomicOr", ptr, idx, value)
def atomic_xor(ptr, idx, value)      : return _atomic_rmw("atomicXor", ptr, idx, value)
def atomic_min(ptr, idx, value)      : return _atomic_rmw("atomicMin", ptr, idx, value)
def atomic_max(ptr, idx, value)      : return _atomic_rmw("atomicMax", ptr, idx, value)
def atomic_exch(ptr, idx, value)     : return _atomic_rmw("atomicExch", ptr, idx, value)


def atomic_cas(ptr, idx, compare, value) -> Tracer:
    """
    ``atomicCAS(&ptr[idx], compare, value)`` → returns the old value at
    ``ptr[idx]``.
    """
    builder = current_builder()
    addr = _atomic_addr(ptr, idx)
    call = ir.Call("atomicCAS", [addr, _to_expr(compare), _to_expr(value)])
    cuda_type = ptr._dtype.cuda if ptr._dtype is not None else "auto"
    name = builder.fresh_name("atom")
    builder.add_stmt(ir.Assign(name, cuda_type, call))
    _mark_pointer_written(ptr)
    return Tracer(ir.Var(name), ptr._dtype)


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


class DeviceFn:
    """
    A reusable device-side function, traced once on first call.

    Parameter annotations must each be either ``tk.DevicePointer[dtype]`` or
    a ``dt.Dtype`` (for scalar-by-value parameters). The return annotation
    determines the function's return type: omitted (or ``None``) means
    ``void``; a ``dt.Dtype`` means the body must return a value of that type.
    """

    __slots__ = (
        "_fn", "_name", "_sig", "_traced", "_tracing",
        "_param_strs", "_return_type", "_return_dtype",
        "_stmts", "_includes", "_usings", "_deps",
    )

    def __init__(self, fn : Callable):
        self._fn = fn
        self._name : str = fn.__name__
        self._sig = inspect.signature(fn)
        self._traced : bool = False
        self._tracing : bool = False
        self._param_strs : List[str] = []
        self._return_type : str = "void"
        self._return_dtype : Optional[dt.Dtype] = None
        self._stmts : List[ir.Stmt] = []
        self._includes : List[tuple] = []
        self._usings : List[str] = []
        self._deps : List["DeviceFn"] = []

    def _ensure_traced(self) -> None:
        if self._traced:
            return
        if self._tracing:
            raise RuntimeError(
                f"Device fn '{self._name}' is recursive; CUDA does not support "
                "recursion in __device__ functions"
            )
        self._tracing = True
        try:
            self._do_trace()
            self._traced = True
        finally:
            self._tracing = False

    def _do_trace(self) -> None:
        builder = KernelBuilder(self._name)
        tracer_args = []
        for param_name, param in self._sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"Device fn '{self._name}': *args/**kwargs not supported"
                )
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"Device fn '{self._name}': parameter '{param_name}' is missing "
                    "a type annotation"
                )
            if isinstance(ann, DevicePointerSpec):
                self._param_strs.append(f"{ann.dtype.cuda} *{param_name}")
                tracer_args.append(PointerTracer(ir.Var(param_name), ann.dtype, builder))
            elif isinstance(ann, dt.Dtype):
                self._param_strs.append(f"{ann.cuda} {param_name}")
                tracer_args.append(Tracer(ir.Var(param_name), ann))
            else:
                raise TypeError(
                    f"Device fn '{self._name}': parameter '{param_name}' has "
                    f"unsupported annotation {ann!r}. Expected tk.DevicePointer[...] "
                    "or a thork dtype."
                )

        ret_ann = self._sig.return_annotation
        if ret_ann is inspect.Parameter.empty or ret_ann is None or ret_ann is type(None):
            self._return_type = "void"
            self._return_dtype = None
        elif isinstance(ret_ann, dt.Dtype):
            self._return_type = ret_ann.cuda
            self._return_dtype = ret_ann
        else:
            raise TypeError(
                f"Device fn '{self._name}': return annotation must be a thork dtype "
                f"or omitted, got {ret_ann!r}"
            )

        token = _builder.set(builder)
        try:
            result = self._fn(*tracer_args)
        finally:
            _builder.reset(token)

        if self._return_dtype is not None:
            if result is None:
                raise TypeError(
                    f"Device fn '{self._name}' declared return type "
                    f"{self._return_dtype.name} but returned None"
                )
            builder.add_stmt(ir.Return(_to_expr(result)))

        self._stmts = builder.stmts
        self._includes = list(builder.includes)
        self._usings = list(builder.usings)
        self._deps = list(builder.device_functions)

    def __call__(self, *args):
        builder = current_builder()
        builder.add_device_fn(self)
        if len(args) != len(self._param_strs):
            raise TypeError(
                f"Device fn '{self._name}' expected {len(self._param_strs)} "
                f"argument(s), got {len(args)}"
            )
        arg_exprs = [_to_expr(a) for a in args]
        call = ir.Call(self._name, arg_exprs)
        if self._return_dtype is None:
            builder.add_stmt(ir.ExprStmt(call))
            return None
        return Tracer(call, self._return_dtype)


def device_fn(fn : Callable) -> DeviceFn:
    """
    Decorator marking a function as a reusable __device__ function.

    The function is traced once on its first call from inside a kernel and
    emitted as a standalone CUDA function above the calling kernel.
    """
    return DeviceFn(fn)

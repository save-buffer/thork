"""
Trace-based stile verifier for thork kernels.

Thork is a tracing DSL — when you call a @tk.jit-decorated function,
each kernel parameter becomes a Tracer that builds IR through its
dunders. This module rides on the same trace: at module-import time we
register stile-type-combination hooks on ``thork.tracer`` (so binops,
math intrinsics, and Local updates propagate stile ``_stype`` info
alongside their IR), and @tvk.jit just hands @tk.jit a function whose
parameter Tracers carry stile types from the very first instruction.

That makes the verifier a straight-line piece of code: trace the
function, the kernel's stored value's ``_stype`` rolls up through the
same trace that produces IR, and we compare it against the spec at
``tvk.store(...)`` time. No source parsing, no second pass.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import stile.type as st
from stile.type import (
    BinaryOp,
    Constant,
    FullDim,
    Reduce,
    Sliced,
    Tensor,
    Type,
    override_dims_in_type,
    type_from_binary_op,
)
from stile.indexing import SymbolicInt, to_affine
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent

from thork import ir
from thork import dtypes as _thork_dt
from thork.types import DevicePointerSpec
from thork.tracer import (
    PointerTracer,
    Tracer,
    current_builder,
    register_stype_hooks,
)
from thork.tracer import range as _tk_range
from thork.jit import JittedKernel, jit as _thork_jit_fn


_py_range = range


# ---------------------------------------------------------------------------
# Annotation marker (unchanged from before)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedPointerSpec(DevicePointerSpec):
    """
    A thork ``DevicePointerSpec`` annotated with a stile spec string.
    Thork's @tk.jit treats this as a plain DevicePointerSpec via
    inheritance; @tvk.jit reads the extra ``stile_spec`` to build the
    matching stile ``Type`` for the input pointer.
    """

    stile_spec : str


class _Tensor:
    """
    Annotation marker: ``tvk.Tensor[dtype, "Name:DIM ..."]``.

    The first ``Tensor``-annotated kernel parameter is the output; the
    rest are inputs. The output's spec only fixes its shape — the
    computation comes from ``spec=`` on the decorator.
    """

    def __class_getitem__(cls, args) -> TypedPointerSpec:
        if not (isinstance(args, tuple) and len(args) == 2):
            raise TypeError(
                "tvk.Tensor expects two subscript arguments: "
                "[dtype, 'Name:DIM ...']"
            )
        dtype, spec = args
        if not isinstance(dtype, _thork_dt.Dtype):
            raise TypeError(
                f"tvk.Tensor's first subscript must be a thork dtype, "
                f"got {dtype!r}"
            )
        if not isinstance(spec, str):
            raise TypeError(
                f"tvk.Tensor's second subscript must be a stile spec string, "
                f"got {spec!r}"
            )
        return TypedPointerSpec(dtype=dtype, stile_spec=spec)


Tensor = _Tensor


# ---------------------------------------------------------------------------
# Slice sugar
# ---------------------------------------------------------------------------


def at(dim_atom : FullDim, index) -> Sliced:
    """
    Sugar for a 1-element slice: ``tvk.at(M, m)`` is ``Sliced(M, m, m+1)``.

    Equivalent to writing ``M[m:m + 1]`` inline.
    """
    if not isinstance(dim_atom, FullDim):
        raise TypeError(
            f"tvk.at: first arg must be a stile FullDim, got "
            f"{type(dim_atom).__name__}"
        )
    return Sliced(dim_atom, index, _add_one(index))


def _add_one(index):
    """
    Compute ``index + 1`` for a slice's upper bound. Works for plain
    ints, thork ``Tracer``s (via ``__add__``), or stile symbolic
    indices.
    """
    if isinstance(index, int):
        return index + 1
    if isinstance(index, Tracer):
        return index + 1
    return to_affine(index) + 1


# ---------------------------------------------------------------------------
# tvk.range — typed loop with accumulator reduction
# ---------------------------------------------------------------------------


# Stack of active loop scopes. Each scope tracks ``acc_name → list of
# body-RHS stypes`` so we can wrap each accumulator's _stype with a
# ``Reduce(sum, dim, body)`` at loop exit.
_loop_scope_stack : list = []


def _current_loop_scope():
    return _loop_scope_stack[-1] if _loop_scope_stack else None


def range(dim_atom : FullDim):
    """
    Typed loop: ``for k in tvk.range(K):``.

    At trace time, delegates to ``tk.range(K.size)`` and (concurrently)
    pushes a loop scope that captures each ``acc += body`` RHS stype.
    At loop exit, the accumulator's ``_stype`` is wrapped with
    ``Reduce(sum, K, body_et)`` so the verifier sees a real reduction.
    """
    if not isinstance(dim_atom, FullDim):
        raise TypeError(
            f"tvk.range expects a stile FullDim, got {type(dim_atom).__name__}"
        )
    return _TypedRangeLoop(dim_atom)


class _TypedRangeLoop:
    """
    Wraps ``tk.range`` to attach accumulator reduction semantics. Uses
    ``tk.range``'s own generator for IR emission; layers loop-scope
    bookkeeping on top so accumulators in the body roll up to a
    ``Reduce``.
    """

    __slots__ = ("_dim",)

    def __init__(self, dim_atom : FullDim):
        self._dim = dim_atom

    def __iter__(self):
        scope = {"dim": self._dim, "augassigns": {}}
        _loop_scope_stack.append(scope)
        try:
            inner = iter(_tk_range(self._dim.size))
            loop_var_tracer = next(inner)
            yield loop_var_tracer
            try:
                next(inner)
            except StopIteration:
                pass
        finally:
            popped = _loop_scope_stack.pop()
            _wrap_accumulators_after_loop(popped)


def _wrap_accumulators_after_loop(scope : dict) -> None:
    """
    For every Local whose ``+=`` RHS we captured during the loop, wrap
    its ``_stype.et`` in ``Reduce(sum, dim, body_et)``.
    """
    dim_atom = scope["dim"]
    for local_obj, body_stypes in scope["augassigns"].items():
        body_et = body_stypes[0].et
        for s in body_stypes[1:]:
            body_et = BinaryOp(op="+", lhs=body_et, rhs=s.et)
        old = local_obj._stype
        st_shape = old.st if isinstance(old, Type) else ()
        st_dt = old.dt if isinstance(old, Type) else None
        local_obj._stype = Type(
            st=st_shape,
            et=Reduce(op="sum", dim=dim_atom, child=body_et),
            dt=st_dt,
        )


# ---------------------------------------------------------------------------
# tvk.load / tvk.store
# ---------------------------------------------------------------------------


def _linear_index(axes) -> ir.Expr:
    """
    Row-major linearization of per-axis ``Sliced(DIM, lo, hi)`` starts,
    used as the per-thread element index.
    """
    if not axes:
        raise ValueError("tvk.load/store needs at least one axis")
    sizes = [_stile_dim_size(a.dim) for a in axes]
    expr : Optional[ir.Expr] = None
    for i, ax in enumerate(axes):
        stride = 1
        for j in _py_range(i + 1, len(axes)):
            stride *= sizes[j]
        start_expr = _bound_to_ir(ax.start)
        term = start_expr if stride == 1 else ir.BinOp(
            "*", start_expr, ir.Const(stride),
        )
        expr = term if expr is None else ir.BinOp("+", expr, term)
    return expr


def _stile_dim_size(d) -> int:
    if isinstance(d, FullDim):
        return d.size
    if isinstance(d, Sliced):
        return _stile_dim_size(d.dim)
    raise TypeError(f"can't extract size from stile dim {d!r}")


def _bound_to_ir(value) -> ir.Expr:
    if isinstance(value, bool):
        return ir.Const(int(value))
    if isinstance(value, (int, float)):
        return ir.Const(value)
    if isinstance(value, Tracer):
        return value._expr
    if hasattr(value, "_expr"):
        return value._expr
    raise TypeError(
        f"@tvk.jit: slice bound {value!r} (type {type(value).__name__}) "
        "isn't a thork value. Pass a thork tracer or an integer."
    )


def load(ptr : PointerTracer, *axes) -> Tracer:
    """
    Typed load: ``tvk.load(X, DIM_0[lo:hi], DIM_1[lo:hi], ...)`` or
    ``tvk.load(X, tvk.at(DIM_0, idx), ...)``.

    Returns a ``Tracer`` whose ``_stype`` is the input pointer's stile
    type with each named dim restricted by the supplied slice. The
    ShapeType is collapsed to ``()`` — per-thread loads are scalar at
    the program level, with the slice info living in the ExprType's
    ``Tensor`` leaf dims.
    """
    if not isinstance(ptr, PointerTracer):
        raise TypeError("tvk.load: first arg must be a thork DevicePointer parameter")
    if not axes:
        raise TypeError(
            "tvk.load: pass at least one DIM[lo:hi] axis so the load's slice "
            "is recorded for verification"
        )
    for a in axes:
        if not isinstance(a, Sliced):
            raise TypeError(
                f"tvk.load: axis {a!r} isn't a stile slice. Pass it as "
                "`DIM[lo:hi]` or `tvk.at(DIM, idx)`."
            )
    idx = _linear_index(axes)
    stype : Optional[Type] = None
    base = _ptr_input_type(ptr)
    if base is not None:
        restricted = override_dims_in_type(base, *axes)
        stype = Type(st=(), et=restricted.et, dt=restricted.dt)
    return Tracer(ir.Load(ptr._expr, idx), ptr._dtype, stype=stype)


def store(ptr : PointerTracer, value, *axes) -> None:
    """
    Typed store: ``tvk.store(out, value, DIM[lo:hi], ...)``.

    Emits an ``ir.Store`` to the underlying pointer, then — if the
    pointer is the kernel's declared output and ``value`` has a stile
    type — verifies the value's ExprType matches the spec restricted
    to the same tile slices.
    """
    if not isinstance(ptr, PointerTracer):
        raise TypeError("tvk.store: first arg must be a thork DevicePointer parameter")
    if not axes:
        raise TypeError(
            "tvk.store: pass at least one DIM[lo:hi] axis so the store's "
            "slice is recorded for verification"
        )
    for a in axes:
        if not isinstance(a, Sliced):
            raise TypeError(
                f"tvk.store: axis {a!r} isn't a stile slice. Pass it as "
                "`DIM[lo:hi]` or `tvk.at(DIM, idx)`."
            )
    builder = current_builder()
    idx = _linear_index(axes)
    if isinstance(value, Tracer):
        val_expr = value._expr
    elif isinstance(value, (int, float, bool)):
        val_expr = ir.Const(value)
    else:
        raise TypeError(
            f"tvk.store: value {value!r} (type {type(value).__name__}) isn't "
            "a thork tracer or scalar"
        )
    builder.add_stmt(ir.Store(ptr._expr, idx, val_expr))
    if isinstance(ptr._expr, ir.Var):
        for p in builder.params:
            if p.kind == "pointer" and p.name == ptr._expr.name:
                p.written = True
                break

    state = _current_jit_state()
    if state is None:
        return
    if not (isinstance(ptr._expr, ir.Var) and ptr._expr.name == state["out_ptr"]):
        return
    # Mark the attempt up front so the wrapper's "never writes" guard
    # doesn't mask a verification error that's about to be raised.
    state["stored"] = True
    value_stype = value._stype if isinstance(value, Tracer) else _scalar_stype_value(value)
    if value_stype is None:
        raise ValueError(
            "@tvk.jit: value stored to the output pointer doesn't have a "
            "recoverable stile type. Build it from `tvk.load(...)`s and "
            "arithmetic over them."
        )
    spec_tile = override_dims_in_type(state["spec_type"], *axes)
    if not verify_exprs_equivalent(spec_tile.et, value_stype.et):
        raise ValueError(
            f"@tvk.jit verification failed: tile stored by `tvk.store({ptr._expr.name}, ...)` "
            f"does not match spec `{state['spec']!r}`.\n"
            f"  spec ExprType (tile):   {spec_tile.et}\n"
            f"  stored ExprType (tile): {value_stype.et}"
        )


def _scalar_stype_value(value):
    if isinstance(value, bool):
        return Type(st=(), et=Constant(value=float(int(value))), dt=None)
    if isinstance(value, (int, float)):
        return Type(st=(), et=Constant(value=float(value)), dt=None)
    return None


# ---------------------------------------------------------------------------
# Per-trace state — set on @tvk.jit entry, read by tvk.store + the load
# helper to know what types each PointerTracer was originally bound to.
# ---------------------------------------------------------------------------


_jit_state_stack : list = []


def _current_jit_state():
    return _jit_state_stack[-1] if _jit_state_stack else None


def _ptr_input_type(ptr : PointerTracer) -> Optional[Type]:
    state = _current_jit_state()
    if state is None:
        return None
    if isinstance(ptr._expr, ir.Var):
        return state["input_types"].get(ptr._expr.name)
    return None


# ---------------------------------------------------------------------------
# Hooks registered with thork.tracer — supply stile-side combination
# logic without thork.tracer importing stile.
# ---------------------------------------------------------------------------


_OP_TO_STILE = {
    "+" : "+",
    "-" : "-",
    "*" : "*",
    "/" : "/",
}


def _hook_binop(lhs, rhs, op : str, swap : bool):
    if op not in _OP_TO_STILE:
        return None
    lhs_st = _stype_of(lhs)
    rhs_st = _stype_of(rhs)
    if lhs_st is None and rhs_st is None:
        return None
    if lhs_st is None:
        lhs_st = _scalar_stype_value(_unwrap_scalar(lhs))
    if rhs_st is None:
        rhs_st = _scalar_stype_value(_unwrap_scalar(rhs))
    if lhs_st is None or rhs_st is None:
        return None
    a, b = (rhs_st, lhs_st) if swap else (lhs_st, rhs_st)
    try:
        return type_from_binary_op(a, b, _OP_TO_STILE[op])
    except ValueError:
        return None


def _hook_unary(operand, op : str):
    operand_st = _stype_of(operand)
    if operand_st is None:
        return None
    if op == "-":
        zero = Type(st=operand_st.st, et=Constant(value=0.0), dt=operand_st.dt)
        try:
            return type_from_binary_op(zero, operand_st, "-")
        except ValueError:
            return None
    return None


_MATH_FUNCS = {
    "expf"   : st.exp,
    "sinf"   : st.sin,
    "cosf"   : st.cos,
    "sqrtf"  : st.sqrt,
    "__expf" : st.exp,
    "__sinf" : st.sin,
    "__cosf" : st.cos,
}


def _hook_math(func : str, args):
    if func not in _MATH_FUNCS:
        return None
    if len(args) != 1:
        return None
    operand_st = _stype_of(args[0])
    if operand_st is None:
        return None
    return _MATH_FUNCS[func](operand_st)


def _hook_scalar(value):
    return _scalar_stype_value(value)


def _hook_local_init(init):
    if isinstance(init, Tracer):
        return init._stype
    return _scalar_stype_value(init)


def _stype_of(x):
    if isinstance(x, Tracer):
        return x._stype
    return None


def _unwrap_scalar(x):
    if isinstance(x, (int, float, bool)):
        return x
    return None


register_stype_hooks(
    binop      = _hook_binop,
    unary      = _hook_unary,
    math       = _hook_math,
    scalar     = _hook_scalar,
    local_init = _hook_local_init,
)


# Also wire AugAssign on Local to the loop-scope tracker so a typed
# range can wrap accumulators with Reduce at loop exit.
from thork import tracer as _thork_tracer
_orig_local_update = _thork_tracer.Local._update


def _local_update_with_scope_tracking(self, op : str, value):
    scope = _current_loop_scope()
    if scope is not None and op == "+=":
        value_st = value._stype if isinstance(value, Tracer) else _scalar_stype_value(value)
        if value_st is not None:
            scope["augassigns"].setdefault(self, []).append(value_st)
    return _orig_local_update(self, op, value)


_thork_tracer.Local._update = _local_update_with_scope_tracking


# ---------------------------------------------------------------------------
# Top-level @tvk.jit decorator + TypedThorkKernel wrapper
# ---------------------------------------------------------------------------


def jit(
    *,
    spec   : str,
    consts : Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator producing a stile-verified thork kernel.

    Pointer parameters declare their stile shape inline via the
    ``tvk.Tensor[dtype, "Name:DIM ..."]`` annotation. The first
    ``Tensor``-annotated parameter is the output; the rest are inputs.

    Verification happens during the same trace that produces IR:
    parameter tracers carry stile types from the start, ops propagate
    them through registered hooks, and ``tvk.store`` checks the
    stored tracer's ``_stype`` against the spec restricted to its
    slice. No source parsing.
    """
    def decorate(fn : Callable) -> "TypedThorkKernel":
        sig = inspect.signature(fn)
        out_ptr_name : Optional[str] = None
        input_specs : Dict[str, str] = {}
        for name, param in sig.parameters.items():
            ann = param.annotation
            if isinstance(ann, TypedPointerSpec):
                if out_ptr_name is None:
                    out_ptr_name = name
                else:
                    input_specs[name] = ann.stile_spec
            elif isinstance(ann, DevicePointerSpec):
                raise TypeError(
                    f"@tvk.jit: pointer parameter '{name}' uses plain "
                    f"`tk.DevicePointer[...]`. Use "
                    f"`tvk.Tensor[<dtype>, 'Name:DIM ...']` so the verifier "
                    f"knows its stile shape."
                )
        if out_ptr_name is None:
            raise TypeError(
                "@tvk.jit kernel needs at least one tvk.Tensor-annotated "
                "parameter (the output)"
            )

        input_types : Dict[str, Type] = {
            n : parse_spec_into_type(s) for n, s in input_specs.items()
        }
        spec_type = parse_spec_into_type(spec)

        compiled = _make_typed_jit(
            fn, out_ptr_name, input_types, spec_type, spec,
        )
        return TypedThorkKernel(compiled, spec=spec, out_ptr_name=out_ptr_name)
    return decorate


def _make_typed_jit(
    fn         : Callable,
    out_ptr    : str,
    input_types: Dict[str, Type],
    spec_type  : Type,
    spec       : str,
) -> JittedKernel:
    """
    Wrap ``fn`` so that during @tk.jit's trace we push a jit-state
    frame (so ``tvk.load`` knows each pointer's stile type and
    ``tvk.store`` knows the spec). Trace-time only; the dispatched
    kernel is just @tk.jit's normal compiled kernel.
    """
    state = {
        "out_ptr"     : out_ptr,
        "input_types" : input_types,
        "spec_type"   : spec_type,
        "spec"        : spec,
        "stored"      : False,
    }

    def wrapper(*args, **kwargs):
        _jit_state_stack.append(state)
        try:
            return fn(*args, **kwargs)
        finally:
            _jit_state_stack.pop()
            if not state["stored"]:
                raise ValueError(
                    f"@tvk.jit kernel never writes to declared output pointer "
                    f"'{out_ptr}'. Add at least one `tvk.store({out_ptr}, "
                    f"value, DIM[lo:hi], ...)` that matches the spec "
                    f"`{spec!r}`."
                )

    wrapper.__wrapped__ = fn
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__module__ = fn.__module__
    try:
        wrapper.__signature__ = inspect.signature(fn)
    except (ValueError, TypeError):
        pass
    return _thork_jit_fn(wrapper)


class TypedThorkKernel:
    """
    A thork ``JittedKernel`` that has been verified against a stile
    spec during its first launch. Forwards subscript launches and the
    source-introspection properties to the underlying ``JittedKernel``.
    """

    __slots__ = ("_kernel", "_spec", "_out_ptr_name")

    def __init__(
        self,
        kernel       : JittedKernel,
        *,
        spec         : str,
        out_ptr_name : str,
    ):
        self._kernel = kernel
        self._spec = spec
        self._out_ptr_name = out_ptr_name

    def __getitem__(self, dispatch_spec):
        return self._kernel[dispatch_spec]

    def bind(self, grid, block):
        return self._kernel.bind(grid, block)

    @property
    def name(self) -> str:
        return self._kernel.name

    @property
    def cuda_source(self) -> str:
        return self._kernel.cuda_source

    @property
    def source_map(self):
        return self._kernel.source_map

    @property
    def spec(self) -> str:
        return self._spec

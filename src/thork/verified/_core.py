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
    ParametricReduce,
    Reduce,
    Sliced,
    Tensor,
    Type,
    override_dims_in_type,
    type_from_binary_op,
)
from stile.indexing import LoopScope, LoopVariable, SymbolicInt, to_affine
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent

from thork import ir
from thork import dtypes as _thork_dt
from thork.types import (
    BlockDim,
    BlockIdx,
    DevicePointerSpec,
    GridDim,
    KittensGlobalSpec,
    ScalarParamSpec,
    ThreadIdx,
    ThreadAttribute,
)
from thork.tracer import (
    PointerTracer,
    Tracer,
    current_builder,
    register_stype_hooks,
)
from thork.tracer import range as _tk_range
from thork.jit import JittedKernel, jit as _thork_jit_fn
from thork import kittens as _tk_kittens


_FIELDS_TO_AXIS = {"x" : 0, "y" : 1, "z" : 2}


def _ir_to_affine(
    expr,
    state      : Dict,
    grid_size  : Optional[Tuple] = None,
    block_size : Optional[Tuple] = None,
):
    """
    Best-effort conversion of a thork ``ir.Expr`` to a stile
    ``AffineExpr``. When ``grid_size`` / ``block_size`` are supplied
    (coverage time), references to ``BlockDim`` / ``GridDim``
    attribute params resolve to concrete ints — so products like
    ``BlockIdx * BlockDim`` become genuinely affine (and the
    ``i = bid*bdm + tid`` index in elementwise kernels resolves
    cleanly). Without launch constants, those references stay
    symbolic and any product involving them is not affine.
    """
    if isinstance(expr, ir.Const):
        if isinstance(expr.value, bool):
            return to_affine(int(expr.value))
        if isinstance(expr.value, (int, float)):
            return to_affine(int(expr.value))
        return None
    if isinstance(expr, ir.Var):
        return _attr_to_affine(expr.name, 0, state, grid_size, block_size)
    if isinstance(expr, ir.Member):
        if not isinstance(expr.operand, ir.Var):
            return None
        axis = _FIELDS_TO_AXIS.get(expr.field)
        if axis is None:
            return None
        return _attr_to_affine(expr.operand.name, axis, state, grid_size, block_size)
    if isinstance(expr, ir.BinOp):
        lhs = _ir_to_affine(expr.lhs, state, grid_size, block_size)
        rhs = _ir_to_affine(expr.rhs, state, grid_size, block_size)
        if lhs is None or rhs is None:
            return None
        if expr.op == "+":
            return lhs + rhs
        if expr.op == "-":
            return lhs - rhs
        if expr.op == "*":
            l_const = lhs.const if not lhs.terms else None
            r_const = rhs.const if not rhs.terms else None
            if l_const is not None:
                return rhs * l_const
            if r_const is not None:
                return lhs * r_const
            return None
        return None
    return None


def _attr_to_affine(
    name       : str,
    axis       : int,
    state      : Dict,
    grid_size  : Optional[Tuple],
    block_size : Optional[Tuple],
):
    """
    Resolve a reference to ``<name>.<axis>`` when ``<name>`` is a
    kernel attribute-param. ``BlockDim`` / ``GridDim`` references
    resolve to concrete ints from the launch (when supplied); other
    kinds become tagged ``SymbolicInt`` atoms the caller will
    enumerate.
    """
    attr_params = state.get("attr_params", {})
    if name not in attr_params:
        return None
    attr_kind, vec_size = attr_params[name]
    if axis is None or axis >= max(vec_size, 1):
        return None
    if attr_kind is BlockDim:
        if block_size is None or axis >= len(block_size):
            key = (attr_kind, axis)
            sym = state["_attr_symints_by_key"].get(key)
            if sym is None:
                sym = SymbolicInt(name=f"_BlockDim_{axis}")
                state["_attr_symints_by_key"][key] = sym
                state["attr_axis_symints"][sym] = (attr_kind, axis)
            return to_affine(sym)
        return to_affine(int(block_size[axis]))
    if attr_kind is GridDim:
        if grid_size is None or axis >= len(grid_size):
            key = (attr_kind, axis)
            sym = state["_attr_symints_by_key"].get(key)
            if sym is None:
                sym = SymbolicInt(name=f"_GridDim_{axis}")
                state["_attr_symints_by_key"][key] = sym
                state["attr_axis_symints"][sym] = (attr_kind, axis)
            return to_affine(sym)
        return to_affine(int(grid_size[axis]))
    if attr_kind in (BlockIdx, ThreadIdx):
        key = (attr_kind, axis)
        sym = state["_attr_symints_by_key"].get(key)
        if sym is None:
            sym = SymbolicInt(name=f"_{attr_kind.__name__}_{axis}")
            state["_attr_symints_by_key"][key] = sym
            state["attr_axis_symints"][sym] = (attr_kind, axis)
            if attr_kind is BlockIdx:
                state["grid_axis_symints"][sym] = axis
        return to_affine(sym)
    return None


def _coerce_bound(
    value,
    state      : Dict,
    grid_size  : Optional[Tuple] = None,
    block_size : Optional[Tuple] = None,
):
    """
    Convert a slice bound to a stile-compatible form. Plain ints and
    stile symbolic indices pass through; thork ``Tracer``s are
    unfolded via ``_ir_to_affine``. Returns ``None`` when the value
    isn't representable as an ``AffineExpr``.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, Tracer):
        return _ir_to_affine(value._expr, state, grid_size, block_size)
    return value


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
# Typed kittens.Global annotation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedKittensGlobalSpec(KittensGlobalSpec):
    """
    A thork ``KittensGlobalSpec`` that also carries a stile spec string.
    @tk.jit treats this as a plain KittensGlobalSpec via inheritance —
    only @tvk.jit reads the extra ``stile_spec``.
    """

    stile_spec : str = ""


class _TypedKittensGlobal:
    """
    Annotation marker: ``tvk.kittens.Global[dtype, "Name:DIM ..."]`` or
    ``tvk.kittens.Global[dtype, (TILE_R, TILE_C), "Name:DIM ..."]`` to
    include a TMA tile shape. Lowers to a ``kittens::gl<...>`` kernel
    parameter (via the existing thork.kittens machinery) and carries
    the stile spec for verification.
    """

    def __class_getitem__(cls, args) -> TypedKittensGlobalSpec:
        if not isinstance(args, tuple) or len(args) < 2:
            raise TypeError(
                "tvk.kittens.Global expects [dtype, 'Name:DIM ...'] or "
                "[dtype, (TILE_R, TILE_C), 'Name:DIM ...']"
            )
        dtype = args[0]
        if not isinstance(dtype, _thork_dt.Dtype):
            raise TypeError(
                f"tvk.kittens.Global's first subscript must be a thork dtype, "
                f"got {dtype!r}"
            )
        spec = args[-1]
        if not isinstance(spec, str):
            raise TypeError(
                f"tvk.kittens.Global's last subscript must be a stile spec "
                f"string, got {spec!r}"
            )
        tile_shape : Optional[tuple] = None
        middle = args[1:-1]
        if middle:
            if len(middle) != 1:
                raise TypeError(
                    "tvk.kittens.Global accepts at most one tile-shape "
                    "between the dtype and the spec"
                )
            shape = middle[0]
            if not (isinstance(shape, tuple) and len(shape) == 2
                    and all(isinstance(x, int) and x > 0 for x in shape)):
                raise TypeError(
                    "tvk.kittens.Global tile shape must be a (rows, cols) "
                    f"tuple of positive ints, got {shape!r}"
                )
            tile_shape = shape
        return TypedKittensGlobalSpec(
            dtype=dtype, tile_shape=tile_shape, stile_spec=spec,
        )


class _TvkKittensNS:
    """
    Stub namespace exposing the ``tvk.kittens.Global`` typed annotation.
    Mirrors the layout of ``tk.kittens.Global`` so users can swap the
    annotation in place when adding verification.
    """

    Global = _TypedKittensGlobal


kittens = _TvkKittensNS()


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
    ``Reduce`` (for ``Local`` accumulators) or a
    ``ParametricReduce`` over the loop var (for TK tile accumulators
    whose body slices ride on the loop variable).
    """

    __slots__ = ("_dim", "_block")

    def __init__(self, dim_atom : FullDim, block : Optional[int] = None):
        self._dim = dim_atom
        self._block = block

    def __iter__(self):
        block = self._block if self._block is not None else 1
        num_iters = self._dim.size if self._block is None else self._dim.size // block
        loop_var = LoopVariable(name=f"_tvk_t_{id(self)}")
        loop_scope_obj = LoopScope(
            name=loop_var.name, start=0, end=num_iters,
        )
        loop_scope_obj.var = loop_var  # use our pre-built LoopVariable
        scope = {
            "dim"          : self._dim,
            "block"        : block,
            "num_iters"    : num_iters,
            "loop_var"     : loop_var,
            "augassigns"   : {},
            "tile_updates" : {},
        }
        _loop_scope_stack.append(scope)
        loop_scope_obj.__enter__()
        try:
            inner = iter(_tk_range(num_iters))
            loop_var_tracer = next(inner)
            _tracer_to_symint[id(loop_var_tracer)] = loop_var
            yield loop_var_tracer
            try:
                next(inner)
            except StopIteration:
                pass
        finally:
            popped = _loop_scope_stack.pop()
            _wrap_accumulators_after_loop(popped)
            _wrap_tile_accumulators_after_loop(popped)
            loop_scope_obj.__exit__(None, None, None)


def tile_range(dim_atom : FullDim, block : int):
    """
    Typed tile loop: ``for t in tvk.tile_range(K, BLOCK):``. Iterates
    ``t`` from 0 to ``K.size // BLOCK``; the loop variable is bound to
    a stile ``LoopVariable`` so per-iteration slices ``K[t*BLOCK :
    (t+1)*BLOCK]`` carry that binding into ``Sliced`` start/end. At
    loop exit, any TK tile accumulator updated inside the body is
    wrapped in ``ParametricReduce(t, 0, K.size/BLOCK, "sum", body)`` —
    stile's normalizer collapses adjacent-tile reductions of that
    shape into a single ``Reduce(K, ...)`` over the full dim.
    """
    if not isinstance(dim_atom, FullDim):
        raise TypeError(
            f"tvk.tile_range expects a stile FullDim, got {type(dim_atom).__name__}"
        )
    if not isinstance(block, int) or block <= 0:
        raise TypeError(f"tvk.tile_range block must be a positive int, got {block!r}")
    if dim_atom.size % block != 0:
        raise ValueError(
            f"tvk.tile_range: dim {dim_atom.name} size {dim_atom.size} isn't "
            f"a multiple of block {block}"
        )
    return _TypedRangeLoop(dim_atom, block=block)


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


def _wrap_tile_accumulators_after_loop(scope : dict) -> None:
    """
    For each TK tile updated inside the loop, wrap its final ``_stype``
    with ``ParametricReduce(loop_var, 0, num_iters, "sum", body_et)``.
    Stile's normalizer collapses this to ``Reduce(K, full_domain, ...)``
    when the body is a single ``Reduce`` whose interval is affine in
    ``loop_var`` with adjacent-tile stride — i.e. the canonical tile-
    walked-matmul pattern.
    """
    loop_var = scope["loop_var"]
    lo = 0
    hi = scope["num_iters"]
    for tile, _ in scope["tile_updates"].items():
        if tile._stype is None:
            continue
        body_et = tile._stype.et
        tile._stype = Type(
            st=tile._stype.st,
            et=ParametricReduce(
                loop_var=loop_var, lo=lo, hi=hi, op="sum", body=body_et,
            ),
            dt=tile._stype.dt,
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
    state["stores"].append(tuple(axes))
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


_UNARY_MATH = {
    "expf"        : st.exp,
    "__expf"      : st.exp,
    "sinf"        : st.sin,
    "__sinf"      : st.sin,
    "cosf"        : st.cos,
    "__cosf"      : st.cos,
    "sqrtf"       : st.sqrt,
    "__fsqrt_rn"  : st.sqrt,
    "fabsf"       : st.abs,
    "abs"         : st.abs,
}


_BINARY_MATH = {
    "fminf" : st.minimum,
    "fmaxf" : st.maximum,
    "min"   : st.minimum,
    "max"   : st.maximum,
}


def _hook_math(func : str, args):
    if func in _UNARY_MATH:
        if len(args) != 1:
            return None
        operand_st = _stype_of(args[0])
        if operand_st is None:
            return None
        return _UNARY_MATH[func](operand_st)
    if func in _BINARY_MATH:
        if len(args) != 2:
            return None
        a_st = _stype_of(args[0])
        b_st = _stype_of(args[1])
        if a_st is None and b_st is None:
            return None
        if a_st is None:
            a_st = _scalar_stype_value(_unwrap_scalar(args[0]))
        if b_st is None:
            b_st = _scalar_stype_value(_unwrap_scalar(args[1]))
        if a_st is None or b_st is None:
            return None
        try:
            return _BINARY_MATH[func](a_st, b_st)
        except ValueError:
            return None
    return None


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
# TK op hooks — propagate _stype through kittens.zero/load/store/swap_layout/mma_AB
# ---------------------------------------------------------------------------


_tracer_to_symint : Dict[int, SymbolicInt] = {}


def _tracer_as_symint(value) -> Any:
    """
    Coerce a slice-bound value to a stile symbolic index. Plain ints
    pass through; thork ``Tracer``s get a ``SymbolicInt`` cached by
    ``id(tracer)`` so re-using the same tracer yields the same symbol.

    When the tracer is a member of a ``BlockIdx`` param (e.g. ``bid.y``),
    the resulting symint is also recorded in the active jit state's
    ``grid_axis_symints`` so the launch-time coverage tracker knows
    its range.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, Tracer):
        key = id(value)
        cached = _tracer_to_symint.get(key)
        if cached is not None:
            return cached
        state = _current_jit_state()
        grid_params = state.get("grid_params", set()) if state is not None else set()
        if isinstance(value._expr, ir.Member):
            operand = value._expr.operand
            field = value._expr.field
            if isinstance(operand, ir.Var) and operand.name in grid_params:
                axis = _FIELDS_TO_AXIS.get(field)
                if axis is not None:
                    sym = SymbolicInt(name=f"_grid_{operand.name}_{field}")
                    _tracer_to_symint[key] = sym
                    state["grid_axis_symints"][sym] = axis
                    return sym
        if isinstance(value._expr, ir.Var):
            base = value._expr.name
        elif isinstance(value._expr, ir.Member):
            operand = value._expr.operand
            if isinstance(operand, ir.Var):
                base = f"{operand.name}_{value._expr.field}"
            else:
                base = "v"
        else:
            base = "v"
        cached = SymbolicInt(name=f"_tvk_{base}_{key}")
        _tracer_to_symint[key] = cached
        return cached
    return value


def _full_dim_of(d) -> FullDim:
    if isinstance(d, FullDim):
        return d
    if isinstance(d, Sliced):
        return _full_dim_of(d.dim)
    raise TypeError(f"can't unwrap FullDim from {d!r}")


def _coord_tile_slices(
    parent_dims : Tuple,
    coord       : Tuple,
    tile_size   : Tuple[int, int],
) -> Tuple[Sliced, ...]:
    """
    Given a 2-D global's dim signature ``(M, K)`` (or already-sliced
    variants), a tile coord ``(row_idx, col_idx)``, and a fixed tile
    shape ``(R, C)``, build per-axis ``Sliced`` overrides ``M[row*R :
    (row+1)*R]`` and ``K[col*C : (col+1)*C]``.
    """
    if len(parent_dims) != 2 or len(coord) != 2:
        return ()
    r_size, c_size = tile_size
    row_sym = _tracer_as_symint(coord[0])
    col_sym = _tracer_as_symint(coord[1])
    return (
        Sliced(
            _full_dim_of(parent_dims[0]),
            to_affine(row_sym) * r_size,
            to_affine(row_sym) * r_size + r_size,
        ),
        Sliced(
            _full_dim_of(parent_dims[1]),
            to_affine(col_sym) * c_size,
            to_affine(col_sym) * c_size + c_size,
        ),
    )


def _tk_zero_hook(rt) -> None:
    """
    ``kittens.zero(rt)`` sets the tile's stype to a scalar 0. Shape is
    unknown until the first load / mma fills it in; we keep the rt's
    current ShapeType (``()`` initially) and only set the ExprType.
    """
    rt._stype = Type(st=(), et=Constant(value=0.0), dt=None)


def _tk_load_hook(dst, src, coord) -> None:
    """
    ``kittens.load(dst, src, [coord])`` propagates stype from src to dst.

    - global → tile (shared or register, with coord): restrict src's
      dims to the tile slice computed from ``coord`` + ``dst`` size.
    - shared → register (no coord): pass through.
    """
    if isinstance(src, _tk_kittens.KittensGlobalTracer):
        src_st = src._stype
        if src_st is None or coord is None:
            return
        if not (isinstance(dst, (_tk_kittens.SharedTile, _tk_kittens.RegisterTile))):
            return
        slices = _coord_tile_slices(src_st.st, coord, (dst._rows, dst._cols))
        dst._stype = override_dims_in_type(src_st, *slices)
        return
    if (
        isinstance(src, _tk_kittens.SharedTile)
        and isinstance(dst, _tk_kittens.RegisterTile)
    ):
        dst._stype = src._stype
        return


def _tk_store_hook(dst, src, coord) -> None:
    """
    ``kittens.store(dst, src, [coord])``. If ``dst`` is the output
    global, verifies ``src._stype`` against the spec restricted to the
    coord's tile.
    """
    if isinstance(dst, _tk_kittens.SharedTile):
        # shared store from a register tile: pass stype through so the
        # shared tile carries the same expression for any subsequent
        # global store.
        dst._stype = src._stype
        return
    if not isinstance(dst, _tk_kittens.KittensGlobalTracer):
        return
    state = _current_jit_state()
    if state is None:
        return
    if dst._param_name != state["out_ptr"]:
        return
    state["stored"] = True
    spec_type = state["spec_type"]
    slices = _coord_tile_slices(spec_type.st, coord, (src._rows, src._cols))
    state["stores"].append(slices)
    src_st = src._stype
    if src_st is None:
        raise ValueError(
            "@tvk.jit: tile stored via kittens.store to the output pointer "
            "doesn't have a recoverable stile type."
        )
    spec_tile = override_dims_in_type(spec_type, *slices)
    if not verify_exprs_equivalent(spec_tile.et, src_st.et):
        raise ValueError(
            f"@tvk.jit verification failed: tile stored by "
            f"`kittens.store({dst._param_name}, ...)` does not match spec "
            f"`{state['spec']!r}`.\n"
            f"  spec ExprType (tile):   {spec_tile.et}\n"
            f"  stored ExprType (tile): {src_st.et}"
        )


def _tk_swap_layout_hook(dst, src) -> None:
    """
    Layout swap reinterprets the same data; stype passes through
    unchanged.
    """
    dst._stype = src._stype


def _tk_mma_AB_hook(c, a, b, c_in) -> None:
    """
    ``kittens.mma_AB(c, a, b, c_in)`` computes ``c = a @ b + c_in``.

    Stile-side: detect the shared dim and emit an einsum. If ``c_in``
    is the additive identity (the scalar ``Constant(0)`` set by
    ``kittens.zero``), use the einsum result directly — the shape
    promotion is automatic. Otherwise the shapes must match and we
    add normally.
    """
    a_st, b_st, c_in_st = a._stype, b._stype, c_in._stype
    if a_st is None or b_st is None or c_in_st is None:
        return
    if not (len(a_st.st) == 2 and len(b_st.st) == 2):
        return
    a_m, a_k = a_st.st
    b_k, b_n = b_st.st
    if _full_dim_of(a_k) is not _full_dim_of(b_k):
        return
    einstr = (
        f"{_full_dim_of(a_m).name} {_full_dim_of(a_k).name}, "
        f"{_full_dim_of(b_k).name} {_full_dim_of(b_n).name} -> "
        f"{_full_dim_of(a_m).name} {_full_dim_of(b_n).name}"
    )
    contracted = st.einsum(a_st, b_st, einstr)
    if (
        isinstance(c_in_st.et, Constant)
        and float(c_in_st.et.value) == 0.0
        and c_in_st.st == ()
    ):
        c._stype = contracted
    else:
        try:
            new_type = type_from_binary_op(c_in_st, contracted, "+")
        except ValueError:
            return
        c._stype = new_type
    scope = _current_loop_scope()
    if scope is not None and "tile_updates" in scope:
        scope["tile_updates"][c] = True


_tk_kittens.register_tk_stype_hooks(
    zero        = _tk_zero_hook,
    load        = _tk_load_hook,
    store       = _tk_store_hook,
    swap_layout = _tk_swap_layout_hook,
    mma_AB      = _tk_mma_AB_hook,
    mma_AB_t    = _tk_mma_AB_hook,
)


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
        grid_params : set = set()
        attr_params : Dict[str, tuple] = {}
        for name, param in sig.parameters.items():
            ann = param.annotation
            if isinstance(ann, (TypedPointerSpec, TypedKittensGlobalSpec)):
                if out_ptr_name is None:
                    out_ptr_name = name
                else:
                    input_specs[name] = ann.stile_spec
            elif isinstance(ann, (DevicePointerSpec, KittensGlobalSpec)):
                kind = (
                    "tvk.kittens.Global"
                    if isinstance(ann, KittensGlobalSpec)
                    else "tvk.Tensor"
                )
                raise TypeError(
                    f"@tvk.jit: pointer parameter '{name}' uses an un-typed "
                    f"annotation. Use `{kind}[<dtype>, 'Name:DIM ...']` so "
                    f"the verifier knows its stile shape."
                )
            elif isinstance(ann, ScalarParamSpec) and ann.attribute is not None:
                attr_params[name] = (ann.attribute, ann.vec_size)
                if ann.attribute is BlockIdx:
                    grid_params.add(name)
        if out_ptr_name is None:
            raise TypeError(
                "@tvk.jit kernel needs at least one tvk.Tensor or "
                "tvk.kittens.Global-annotated parameter (the output)"
            )

        input_types : Dict[str, Type] = {
            n : parse_spec_into_type(s) for n, s in input_specs.items()
        }
        spec_type = parse_spec_into_type(spec)

        compiled, state = _make_typed_jit(
            fn, out_ptr_name, input_types, spec_type, spec, grid_params, attr_params,
        )
        return TypedThorkKernel(
            compiled, state=state, spec=spec, out_ptr_name=out_ptr_name,
        )
    return decorate


def _make_typed_jit(
    fn          : Callable,
    out_ptr     : str,
    input_types : Dict[str, Type],
    spec_type   : Type,
    spec        : str,
    grid_params : set,
    attr_params : Dict[str, tuple],
) -> Tuple[JittedKernel, Dict[str, Any]]:
    """
    Wrap ``fn`` so that during @tk.jit's trace we push a jit-state
    frame (so ``tvk.load`` knows each pointer's stile type and
    ``tvk.store`` knows the spec). Trace-time only; the dispatched
    kernel is just @tk.jit's normal compiled kernel. Returns both
    the JittedKernel and the state dict so the wrapper kernel can
    run coverage checks after trace.
    """
    state = {
        "out_ptr"            : out_ptr,
        "input_types"        : input_types,
        "spec_type"          : spec_type,
        "spec"               : spec,
        "stored"             : False,
        "grid_params"        : grid_params,
        "attr_params"        : attr_params,
        "attr_axis_symints"  : {},
        "_attr_symints_by_key" : {},
        "grid_axis_symints"  : {},
        "stores"             : [],
        "out_dims"           : spec_type.st,
    }

    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    def wrapper(*args, **kwargs):
        _tracer_to_symint.clear()
        _jit_state_stack.append(state)
        # Hand each KittensGlobalTracer its input stile type so subsequent
        # kittens.load / store hooks have something to restrict.
        for i, arg in enumerate(args):
            if i >= len(param_names):
                break
            name = param_names[i]
            if isinstance(arg, _tk_kittens.KittensGlobalTracer):
                t = input_types.get(name)
                if t is not None:
                    arg._stype = t
        try:
            return fn(*args, **kwargs)
        finally:
            _jit_state_stack.pop()
            if not state["stored"]:
                raise ValueError(
                    f"@tvk.jit kernel never writes to declared output pointer "
                    f"'{out_ptr}'. Add at least one `tvk.store({out_ptr}, "
                    f"value, DIM[lo:hi], ...)` (or `kittens.store(...)` for "
                    f"TK kernels) that matches the spec `{spec!r}`."
                )

    wrapper.__wrapped__ = fn
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__module__ = fn.__module__
    try:
        wrapper.__signature__ = inspect.signature(fn)
    except (ValueError, TypeError):
        pass
    return _thork_jit_fn(wrapper), state


# ---------------------------------------------------------------------------
# Coverage check — verify the union of every store's slice covers the
# declared output shape under the dispatch's grid range.
# ---------------------------------------------------------------------------


def _resolve_bound(expr, substitutions : Dict) -> Optional[int]:
    """
    Resolve an ``AffineExpr`` / ``SymbolicInt`` / int to a concrete int
    under ``substitutions: SymbolicInt → int``. Returns ``None`` if the
    expression isn't a stile symbolic index (e.g. a raw thork tracer in
    a ``DIM[i:i+1]`` slice) or any referenced symbol is missing from
    the substitution map — in either case the caller treats the slice
    as opaque and skips coverage for that interval.
    """
    if isinstance(expr, int):
        return expr
    try:
        affine = to_affine(expr)
    except TypeError:
        return None
    total = affine.const
    for atom, coeff in affine.terms:
        if atom in substitutions:
            total += coeff * substitutions[atom]
        else:
            return None
    return int(total)


def _enumerate_attr_substitutions(
    attr_axis_symints : Dict, grid_size : Tuple, block_size : Tuple,
) -> list:
    """
    Build all ``{SymbolicInt: int}`` substitutions for every symbol
    derived from a kernel ``ThreadAttribute`` param:

    - ``BlockDim`` / ``GridDim`` get substituted with their fixed
      constants from the launch (single value each).
    - ``BlockIdx`` enumerates over ``[0, grid[axis])``.
    - ``ThreadIdx`` enumerates over ``[0, block[axis])``.

    Constants are applied first to keep the cartesian product small,
    then BlockIdx and ThreadIdx are enumerated.
    """
    base : Dict = {}
    enumerated : list = []
    for sym, info in attr_axis_symints.items():
        attr_kind, axis = info
        if attr_kind is BlockDim:
            if axis < len(block_size):
                base[sym] = int(block_size[axis])
            continue
        if attr_kind is GridDim:
            if axis < len(grid_size):
                base[sym] = int(grid_size[axis])
            continue
        if attr_kind is BlockIdx:
            if axis < len(grid_size):
                enumerated.append((sym, list(_py_range(int(grid_size[axis])))))
            continue
        if attr_kind is ThreadIdx:
            if axis < len(block_size):
                enumerated.append((sym, list(_py_range(int(block_size[axis])))))
            continue
    results = [base]
    for sym, values in enumerated:
        results = [{**s, sym : v} for s in results for v in values]
    return results


def _union_intervals(intervals : list) -> list:
    """
    Merge a list of ``[lo, hi)`` intervals into the disjoint union,
    sorted. Empty list → ``[]``; touching intervals merge.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _check_coverage(
    state      : Dict[str, Any],
    grid_size  : Tuple,
    block_size : Tuple,
) -> None:
    """
    For each declared output dim, verify the union of per-store slices
    (substituted over the grid and block ranges, with BlockDim/GridDim
    fixed to their launch constants) covers ``[0, dim.size)``.
    """
    out_dims = state.get("out_dims", ())
    stores = state.get("stores", [])
    if not stores:
        return

    # Coerce every store's bounds upfront — _ir_to_affine populates
    # ``attr_axis_symints`` as it walks references to Block/Thread/etc.
    # attribute params, so the enumeration step below sees the full set
    # of symbols regardless of whether they were ever materialized at
    # trace time.
    coerced : list = []
    for axes in stores:
        coerced_axes : list = []
        for sliced in axes:
            d_name = sliced.dim.name if hasattr(sliced.dim, "name") else _full_dim_of(sliced.dim).name
            lo_aff = _coerce_bound(sliced.start, state, grid_size, block_size)
            hi_aff = _coerce_bound(sliced.end, state, grid_size, block_size)
            coerced_axes.append((d_name, lo_aff, hi_aff))
        coerced.append(coerced_axes)

    per_dim_intervals : Dict[str, list] = {d.name : [] for d in out_dims}
    substitutions_list = _enumerate_attr_substitutions(
        state.get("attr_axis_symints", {}), grid_size, block_size,
    )
    if not substitutions_list:
        substitutions_list = [{}]

    for axes in coerced:
        for d_name, lo_aff, hi_aff in axes:
            if d_name not in per_dim_intervals:
                continue
            if lo_aff is None or hi_aff is None:
                continue
            for subs in substitutions_list:
                lo = _resolve_bound(lo_aff, subs)
                hi = _resolve_bound(hi_aff, subs)
                if lo is None or hi is None:
                    continue
                per_dim_intervals[d_name].append((lo, hi))

    for d in out_dims:
        intervals = per_dim_intervals.get(d.name, [])
        if not intervals:
            continue
        merged = _union_intervals(intervals)
        expected = [(0, d.size)]
        if merged != expected:
            raise ValueError(
                f"@tvk.jit coverage check failed for output '{state['out_ptr']}': "
                f"stores cover {merged!r} along dim `{d.name}` (size {d.size}), "
                f"expected {expected!r}. Some output elements are never written."
            )


class TypedThorkKernel:
    """
    A thork ``JittedKernel`` that has been verified against a stile
    spec during its first launch. Forwards subscript launches and the
    source-introspection properties to the underlying ``JittedKernel``,
    interposing a launch-time coverage check that verifies the union
    of per-store tile slices covers the declared output shape.
    """

    __slots__ = ("_kernel", "_state", "_spec", "_out_ptr_name")

    def __init__(
        self,
        kernel       : JittedKernel,
        *,
        state        : Dict[str, Any],
        spec         : str,
        out_ptr_name : str,
    ):
        self._kernel = kernel
        self._state = state
        self._spec = spec
        self._out_ptr_name = out_ptr_name

    def __getitem__(self, dispatch_spec):
        base = self._kernel[dispatch_spec]
        return _TypedLauncher(base, self._state)

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


class _TypedLauncher:
    """
    Wraps a thork ``_Launcher`` to interpose a coverage check before
    dispatch. The base launcher already calls ``_ensure_compiled``,
    which triggers the trace that populates ``state['stores']`` and
    ``state['grid_axis_symints']``; we then enumerate the grid range
    and verify the per-axis interval union of all stores covers the
    declared output shape.
    """

    __slots__ = ("_base", "_state")

    def __init__(self, base, state : Dict[str, Any]):
        self._base = base
        self._state = state

    def __call__(self, *args):
        self._base._kernel._ensure_compiled()
        _check_coverage(self._state, self._base._grid_size, self._base._block_size)
        return self._base(*args)

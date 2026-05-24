"""
thork.kittens — DSL wrappers for the ThunderKittens primitive set.

Exposes register tiles, shared tiles, global descriptors, tile loads and
stores, layout swaps, and warp-level mma so a thork kernel can express
the level-04-style tile-based matmul shown in the TK educational gemm
examples.

This module is a thin trace-time front end over TK's C++ template API;
no Python-side computation happens, only IR emission. Kernels that
touch any wrapper in this module set ``builder.uses_kittens`` so the
runtime routes them through ``nvcc`` (TK pulls in libstdc++ headers
that nvrtc can't satisfy on its own).
"""

from typing import Optional

from . import dtypes as dt
from . import ir
from .tracer import (
    KernelBuilder,
    Tracer,
    _to_expr,
    current_builder,
)
from .types import KittensGlobal as Global  # re-export


__all__ = [
    "Global",
    "KittensGlobalTracer",
    "RegisterTile",
    "SharedTile",
    "rt_bf",
    "rt_hf",
    "rt_fl",
    "rt_bf_col",
    "rt_hf_col",
    "rt_fl_col",
    "st_bf",
    "st_hf",
    "st_fl",
    "zero",
    "load",
    "store",
    "swap_layout",
    "mma_AB",
    "mma_AB_t",
    "warpgroup",
    "Semaphore",
    "semaphore",
    "init_semaphore",
    "wait",
    "tma",
    "TensorTile",
    "TensorAllocator",
    "tt_bf",
    "tt_hf",
    "tt_fl",
    "tensor_allocator",
    "tensor_alloc",
    "tcgen05_commit",
    "tensor_load_wait",
]


_TK_ELEM = {
    "float32"  : "float",
    "float16"  : "kittens::half",
    "bfloat16" : "kittens::bf16",
}


def _tk_elem(d : dt.Dtype) -> str:
    if d.name not in _TK_ELEM:
        raise TypeError(f"thork dtype {d.name} has no ThunderKittens equivalent")
    return _TK_ELEM[d.name]


class _TKHandle:
    """
    Base class for handles to TK objects (register tiles, shared tiles,
    global descriptors). Subclasses set ``_expr`` to the IR ``Var`` that
    references the object by its emitted C++ name.
    """

    __slots__ = ("_expr", "_dtype", "_builder")

    def __init__(self, expr : ir.Expr, dtype : dt.Dtype, builder : KernelBuilder):
        self._expr = expr
        self._dtype = dtype
        self._builder = builder


class KittensGlobalTracer(_TKHandle):
    """
    Handle to a ``kittens::gl<T, 1, 1, -1, -1>`` kernel parameter.

    Pass it directly to ``kittens.load`` / ``kittens.store`` along with a
    ``(row, col)`` tile coordinate.
    """

    __slots__ = ("_param_name",)

    def __init__(self, expr : ir.Var, dtype : dt.Dtype, builder : KernelBuilder):
        super().__init__(expr, dtype, builder)
        self._param_name = expr.name


class RegisterTile(_TKHandle):
    """
    Handle to a register tile declared with ``rt_bf`` / ``rt_hf`` / ``rt_fl``.
    """

    __slots__ = ("_rows", "_cols", "_layout")

    def __init__(
        self,
        expr    : ir.Expr,
        dtype   : dt.Dtype,
        builder : KernelBuilder,
        rows    : int,
        cols    : int,
        layout  : str,
    ):
        super().__init__(expr, dtype, builder)
        self._rows = rows
        self._cols = cols
        self._layout = layout


class TensorTile(_TKHandle):
    """
    Handle to a Blackwell tensor-memory tile (``kittens::tt<T, R, C>``).
    The declaration is uninitialized; allocate storage with
    ``kittens.tensor_alloc(tt, allocator, col_offset)`` (typically gated
    on the warpgroup's first lane).
    """

    __slots__ = ("_rows", "_cols")

    def __init__(
        self,
        expr    : ir.Expr,
        dtype   : dt.Dtype,
        builder : KernelBuilder,
        rows    : int,
        cols    : int,
    ):
        super().__init__(expr, dtype, builder)
        self._rows = rows
        self._cols = cols


class TensorAllocator(_TKHandle):
    """
    Handle to a ``kittens::tensor_allocator<nblocks_per_sm, ncta>``.
    """

    __slots__ = ()

    def __init__(self, name : str, builder : KernelBuilder):
        super().__init__(ir.Var(name), dt.uint32, builder)


class Semaphore(_TKHandle):
    """
    Handle to a ``__shared__ kittens::semaphore``. Use ``init_semaphore``
    and ``wait`` (or ``tma::expect_bytes`` / ``tma::load_async``) to drive
    it.
    """

    __slots__ = ()

    def __init__(self, name : str, builder : KernelBuilder):
        super().__init__(ir.Var(name), dt.uint32, builder)


class SharedTile(_TKHandle):
    """
    Handle to a shared tile (``__shared__`` allocation through TK's
    ``shared_allocator``).
    """

    __slots__ = ("_rows", "_cols")

    def __init__(
        self,
        expr    : ir.Expr,
        dtype   : dt.Dtype,
        builder : KernelBuilder,
        rows    : int,
        cols    : int,
    ):
        super().__init__(expr, dtype, builder)
        self._rows = rows
        self._cols = cols


def _mark_kittens(builder : KernelBuilder) -> None:
    """
    Flip the ``uses_kittens`` flag so the kernel is routed through nvcc
    and the codegen emits the shared-allocator preamble.
    """
    builder.uses_kittens = True
    builder.add_include("kittens.cuh", system=False)


def _rt_factory(layout : str, type_alias : str, elem_dtype : dt.Dtype):
    """
    Build the per-(dtype, layout) register-tile constructor:
    ``rt_bf(R, C)`` / ``rt_bf_col(R, C)`` / ``rt_fl(R, C)`` / ...
    """
    def make(rows : int, cols : int) -> RegisterTile:
        for v, label in ((rows, "rows"), (cols, "cols")):
            if not isinstance(v, int) or v <= 0:
                raise TypeError(f"{type_alias} {label} must be a positive int, got {v!r}")
        builder = current_builder()
        _mark_kittens(builder)
        name = builder.fresh_name("rt")
        if layout == "row":
            cuda_type = f"kittens::{type_alias}<{rows}, {cols}>"
        else:
            cuda_type = (
                f"kittens::{type_alias}<{rows}, {cols}, "
                "kittens::ducks::rt_layout::col>"
            )
        builder.add_stmt(ir.DefaultDecl(name=name, cuda_type=cuda_type))
        return RegisterTile(
            ir.Var(name), elem_dtype, builder, rows, cols, layout,
        )
    return make


rt_bf     = _rt_factory("row", "rt_bf", dt.bfloat16)
rt_hf     = _rt_factory("row", "rt_hf", dt.float16)
rt_fl     = _rt_factory("row", "rt_fl", dt.float32)
rt_bf_col = _rt_factory("col", "rt_bf", dt.bfloat16)
rt_hf_col = _rt_factory("col", "rt_hf", dt.float16)
rt_fl_col = _rt_factory("col", "rt_fl", dt.float32)


def _st_factory(type_alias : str, elem_dtype : dt.Dtype):
    """
    Build the per-dtype shared-tile constructor.

    Emits ``auto &name = __thork_smem_al.allocate<kittens::st_bf<R, C>>();``
    so each tile lives in the dynamic shared-memory pool the codegen sets
    up at the top of every TK kernel.
    """
    def make(rows : int, cols : int) -> SharedTile:
        for v, label in ((rows, "rows"), (cols, "cols")):
            if not isinstance(v, int) or v <= 0:
                raise TypeError(f"{type_alias} {label} must be a positive int, got {v!r}")
        builder = current_builder()
        _mark_kittens(builder)
        name = builder.fresh_name("st")
        cuda_type = f"kittens::{type_alias}<{rows}, {cols}>"
        ref_type = f"{cuda_type} &"
        call = ir.Call(
            f"__thork_smem_al.allocate<{cuda_type}>", [],
        )
        builder.add_stmt(ir.Assign(name=name, cuda_type=ref_type, value=call))
        return SharedTile(ir.Var(name), elem_dtype, builder, rows, cols)
    return make


st_bf = _st_factory("st_bf", dt.bfloat16)
st_hf = _st_factory("st_hf", dt.float16)
st_fl = _st_factory("st_fl", dt.float32)


def _tt_factory(type_alias : str, elem_dtype : dt.Dtype):
    """
    Build per-dtype constructors for ``kittens::tt<...>`` tensor-memory
    tiles. Like ``rt_*`` factories, this emits just the declaration; the
    storage is bound via ``tensor_alloc``.
    """
    def make(rows : int, cols : int) -> TensorTile:
        for v, label in ((rows, "rows"), (cols, "cols")):
            if not isinstance(v, int) or v <= 0:
                raise TypeError(f"{type_alias} {label} must be a positive int, got {v!r}")
        builder = current_builder()
        _mark_kittens(builder)
        name = builder.fresh_name("tt")
        cuda_type = f"kittens::{type_alias}<{rows}, {cols}>"
        builder.add_stmt(ir.DefaultDecl(name=name, cuda_type=cuda_type))
        return TensorTile(ir.Var(name), elem_dtype, builder, rows, cols)
    return make


tt_bf = _tt_factory("tt_bf", dt.bfloat16)
tt_hf = _tt_factory("tt_hf", dt.float16)
tt_fl = _tt_factory("tt_fl", dt.float32)


def tensor_allocator(nblocks_per_sm : int = 1, ncta : int = 1) -> TensorAllocator:
    """
    Declare ``kittens::tensor_allocator<nblocks_per_sm, ncta> name{};`` and
    return a handle. Used to back ``tt`` tiles with Blackwell tensor memory.
    """
    if nblocks_per_sm not in (1, 2) or ncta not in (1, 2):
        raise ValueError("tensor_allocator: nblocks_per_sm and ncta must each be 1 or 2")
    builder = current_builder()
    _mark_kittens(builder)
    name = builder.fresh_name("tm")
    builder.add_stmt(ir.ConstructorDecl(
        name=name,
        cuda_type=f"kittens::tensor_allocator<{nblocks_per_sm}, {ncta}>",
        args=[],
    ))
    return TensorAllocator(name, builder)


def tensor_alloc(tt : TensorTile, alloc : TensorAllocator, col_offset = 0) -> None:
    """
    Emit ``tt = alloc.allocate<decltype(tt)>(col_offset);`` — bind a
    declared ``tt`` to a slice of the tensor allocator's pool. Must be
    called by a single lane of the warpgroup (typically lane 0), with a
    ``warpgroup::sync`` afterwards so the rest of the lanes see the
    populated handle.
    """
    if not isinstance(tt, TensorTile):
        raise TypeError("tensor_alloc tt must be a TensorTile")
    if not isinstance(alloc, TensorAllocator):
        raise TypeError("tensor_alloc alloc must be a TensorAllocator")
    builder = current_builder()
    _mark_kittens(builder)
    call = ir.MethodCall(
        obj=alloc._expr,
        method="allocate",
        template_args=[ir.Raw(f"decltype({tt._expr.name})")],
        args=[_to_expr(col_offset)],
    )
    builder.add_stmt(ir.Update(name=tt._expr.name, op="=", value=call))


def tcgen05_commit(sem : Semaphore, ncta : int = 1) -> None:
    """
    Emit ``kittens::detail::tcgen05::commit<ncta>(sem);`` — issued by a
    single lane after the final ``mma_AB`` to flip ``sem`` once all
    outstanding tcgen05 MMAs have written their accumulators.
    """
    if not isinstance(sem, Semaphore):
        raise TypeError("tcgen05_commit sem must be a Semaphore")
    if ncta not in (1, 2):
        raise ValueError("tcgen05_commit ncta must be 1 or 2")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(
        f"kittens::detail::tcgen05::commit<{int(ncta)}>", [sem._expr],
    )))


def tensor_load_wait() -> None:
    """
    Emit ``kittens::tensor_load_wait();`` — wait for outstanding
    ``warpgroup::load_async`` reads from tensor memory to complete.
    """
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call("kittens::tensor_load_wait", [])))


def semaphore() -> Semaphore:
    """
    Declare a ``__shared__ kittens::semaphore`` in the current kernel and
    return a handle. Initialize with ``tk.kittens.init_semaphore`` (call
    from a single thread) followed by a ``tk.syncthreads()``.
    """
    builder = current_builder()
    _mark_kittens(builder)
    name = builder.fresh_name("sem")
    builder.add_stmt(ir.DefaultDecl(
        name=name, cuda_type="__shared__ kittens::semaphore",
    ))
    return Semaphore(name, builder)


def init_semaphore(sem : Semaphore, n_arrived : int, n_threads : int) -> None:
    """
    Emit ``kittens::init_semaphore(sem, n_arrived, n_threads);``. Must be
    called by a single thread of the block before any thread waits on
    the semaphore.
    """
    if not isinstance(sem, Semaphore):
        raise TypeError("init_semaphore expects a kittens.semaphore handle")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(
        "kittens::init_semaphore",
        [sem._expr, ir.Const(int(n_arrived)), ir.Const(int(n_threads))],
    )))


def wait(sem : Semaphore, phase) -> None:
    """
    Emit ``kittens::wait(sem, phase);`` — block this thread until the
    semaphore's phase bit flips.
    """
    if not isinstance(sem, Semaphore):
        raise TypeError("wait expects a kittens.semaphore handle")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(
        "kittens::wait", [sem._expr, _to_expr(phase)],
    )))


def _coord_expr(coord) -> ir.Raw:
    """
    Convert a Python ``(row, col)`` tuple of Tracers/ints into the TK
    coord literal ``{0, 0, row, col}``.
    """
    if not (isinstance(coord, tuple) and len(coord) == 2):
        raise TypeError(
            f"kittens load/store coord must be a (row, col) tuple, got {coord!r}"
        )
    row, col = coord
    from .codegen import format_expr as _fmt
    row_s = _fmt(_to_expr(row))
    col_s = _fmt(_to_expr(col))
    return ir.Raw(f"{{0, 0, {row_s}, {col_s}}}")


def _zero(scope : str, rt : RegisterTile) -> None:
    if not isinstance(rt, RegisterTile):
        raise TypeError(f"kittens.zero expects a register tile, got {type(rt).__name__}")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::zero", [rt._expr])))


def _load(scope : str, dst, src, coord) -> None:
    builder = current_builder()
    _mark_kittens(builder)
    if isinstance(dst, SharedTile) and isinstance(src, KittensGlobalTracer):
        if coord is None:
            raise TypeError("load(shared, global, coord) requires a coord tuple")
        args = [dst._expr, src._expr, _coord_expr(coord)]
        builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::load", args)))
        return
    if isinstance(dst, RegisterTile) and isinstance(src, SharedTile):
        if coord is not None:
            raise TypeError("load(register, shared) does not take a coord")
        args = [dst._expr, src._expr]
        builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::load", args)))
        return
    if isinstance(dst, RegisterTile) and isinstance(src, KittensGlobalTracer):
        if coord is None:
            raise TypeError("load(register, global, coord) requires a coord tuple")
        args = [dst._expr, src._expr, _coord_expr(coord)]
        builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::load", args)))
        return
    raise TypeError(
        f"kittens.load: unsupported argument combination "
        f"({type(dst).__name__}, {type(src).__name__})"
    )


def _store(scope : str, dst, src, coord) -> None:
    builder = current_builder()
    _mark_kittens(builder)
    if isinstance(dst, KittensGlobalTracer):
        if coord is None:
            raise TypeError("store(global, tile, coord) requires a coord tuple")
        for p in builder.params:
            if p.kind == "kittens_global" and p.name == dst._param_name:
                p.written = True
                break
        args = [dst._expr, src._expr, _coord_expr(coord)]
        builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::store", args)))
        return
    if isinstance(dst, SharedTile) and isinstance(src, RegisterTile):
        if coord is not None:
            raise TypeError("store(shared, register) does not take a coord")
        args = [dst._expr, src._expr]
        builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::store", args)))
        return
    raise TypeError(
        f"kittens.store: unsupported argument combination "
        f"({type(dst).__name__}, {type(src).__name__})"
    )


def _swap_layout(scope : str, dst : RegisterTile, src : RegisterTile) -> None:
    if not (isinstance(dst, RegisterTile) and isinstance(src, RegisterTile)):
        raise TypeError("kittens.swap_layout expects two register tiles")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(
        f"kittens::{scope}::swap_layout", [dst._expr, src._expr],
    )))


def _mma(
    scope : str,
    func  : str,
    *args,
) -> None:
    if not args or not (3 <= len(args) <= 4):
        raise TypeError(f"kittens.{func} expects 3 or 4 register-tile args")
    c, a, b = args[:3]
    c_in = args[3] if len(args) == 4 else None
    if not all(isinstance(x, RegisterTile) for x in (c, a, b)):
        raise TypeError(f"kittens.{func} expects register tiles")
    if c_in is None:
        c_in = c
    elif not isinstance(c_in, RegisterTile):
        raise TypeError(f"kittens.{func} c_in must be a RegisterTile")
    builder = current_builder()
    _mark_kittens(builder)
    ir_args = [c._expr, a._expr, b._expr, c_in._expr]
    builder.add_stmt(ir.ExprStmt(ir.Call(f"kittens::{scope}::{func}", ir_args)))


def _wg_tcgen05_mma(func : str, d, a, b, sem) -> None:
    if not isinstance(d, TensorTile):
        raise TypeError(f"warpgroup.{func} (tcgen05) d must be a TensorTile")
    if not isinstance(a, SharedTile):
        raise TypeError(f"warpgroup.{func} (tcgen05) a must be a SharedTile")
    if not isinstance(b, SharedTile):
        raise TypeError(f"warpgroup.{func} (tcgen05) b must be a SharedTile")
    if not isinstance(sem, Semaphore):
        raise TypeError(f"warpgroup.{func} (tcgen05) sem must be a Semaphore")
    builder = current_builder()
    _mark_kittens(builder)
    builder.add_stmt(ir.ExprStmt(ir.Call(
        f"kittens::warpgroup::{func}",
        [d._expr, a._expr, b._expr, sem._expr],
    )))


def zero(rt : RegisterTile) -> None:
    """
    Emit ``kittens::warp::zero(rt);``.
    """
    _zero("warp", rt)


def load(dst, src, coord = None) -> None:
    """
    Emit a warp-scoped ThunderKittens load.

    - ``load(shared_tile, global, (row, col))`` — global → shared tile at
      the given tile coordinate.
    - ``load(reg_tile, shared_tile)`` — shared → register, no coord.
    - ``load(reg_tile, global, (row, col))`` — global → register direct.
    """
    _load("warp", dst, src, coord)


def store(dst, src, coord = None) -> None:
    """
    Emit a warp-scoped ThunderKittens store.

    - ``store(global, reg_or_shared, (row, col))`` — write a tile out to
      global memory at the given tile coordinate.
    - ``store(shared, reg_tile)`` — write a register tile to a shared tile.
    """
    _store("warp", dst, src, coord)


def swap_layout(dst : RegisterTile, src : RegisterTile) -> None:
    """
    Emit ``kittens::warp::swap_layout(dst, src);`` — reinterpret ``src``
    in the opposite (row ↔ col) layout into ``dst``.
    """
    _swap_layout("warp", dst, src)


def mma_AB(
    c    : RegisterTile,
    a    : RegisterTile,
    b    : RegisterTile,
    c_in : Optional[RegisterTile] = None,
) -> None:
    """
    Emit ``kittens::warp::mma_AB(c, a, b, c_in);``. Performs
    ``c = a @ b + c_in``. ``b`` must be in column layout; ``c`` and
    ``c_in`` must be accumulator (float) tiles.
    """
    _mma("warp", "mma_AB", c, a, b, c_in)


def mma_AB_t(
    c    : RegisterTile,
    a    : RegisterTile,
    b    : RegisterTile,
    c_in : Optional[RegisterTile] = None,
) -> None:
    """
    Emit ``kittens::warp::mma_AB_t(c, a, b, c_in);``. Performs
    ``c = a @ b^T + c_in``.
    """
    _mma("warp", "mma_AB_t", c, a, b, c_in)


class _WarpgroupNS:
    """
    Warpgroup-scoped (group<4>) versions of the warp ops. Use when the
    kernel is launched with at least 4 warps per block; TK's warpgroup
    operations assume that 4-warp shape and target Hopper/Blackwell
    instructions where applicable.
    """

    @staticmethod
    def zero(rt : RegisterTile) -> None:
        """
        Emit ``kittens::warpgroup::zero(rt);``.
        """
        _zero("warpgroup", rt)

    @staticmethod
    def load(dst, src, coord = None) -> None:
        """
        Warpgroup-scoped load. Same shapes as the module-level ``load``.
        """
        _load("warpgroup", dst, src, coord)

    @staticmethod
    def store(dst, src, coord = None) -> None:
        """
        Warpgroup-scoped store. Same shapes as the module-level ``store``.
        """
        _store("warpgroup", dst, src, coord)

    @staticmethod
    def swap_layout(dst : RegisterTile, src : RegisterTile) -> None:
        """
        Emit ``kittens::warpgroup::swap_layout(dst, src);``.
        """
        _swap_layout("warpgroup", dst, src)

    @staticmethod
    def mma_AB(*args) -> None:
        """
        Two overloads, dispatched on the destination type:

        - ``mma_AB(reg_c, reg_a, reg_b[, reg_c_in])`` — emits
          ``kittens::warpgroup::mma_AB(c, a, b, c_in)``; the usual
          register-tile accumulation.
        - ``mma_AB(tt_d, st_a, st_b, sem)`` — emits the tcgen05 MMA that
          accumulates into a Blackwell tensor-memory tile and signals
          ``sem`` when the operands have been read.
        """
        if args and isinstance(args[0], TensorTile):
            _wg_tcgen05_mma("mma_AB", *args)
            return
        _mma("warpgroup", "mma_AB", *args)

    @staticmethod
    def mma_AB_t(*args) -> None:
        """
        Register- and tcgen05-overloads of ``mma_AB_t``. Same dispatch
        rule as ``mma_AB``.
        """
        if args and isinstance(args[0], TensorTile):
            _wg_tcgen05_mma("mma_AB_t", *args)
            return
        _mma("warpgroup", "mma_AB_t", *args)

    @staticmethod
    def mm_AB(d : TensorTile, a : SharedTile, b : SharedTile, sem : Semaphore) -> None:
        """
        Emit ``kittens::warpgroup::mm_AB(d, a, b, sem);`` — tcgen05 MMA
        without a prior accumulator (used on the first K iteration; later
        iterations use ``mma_AB`` to accumulate).
        """
        _wg_tcgen05_mma("mm_AB", d, a, b, sem)

    @staticmethod
    def load_async(dst : RegisterTile, src : TensorTile) -> None:
        """
        Emit ``kittens::warpgroup::load_async(dst, src);`` — start an
        async load from a tensor-memory tile into a register tile. Pair
        with ``tensor_load_wait`` before reading ``dst``.
        """
        if not isinstance(dst, RegisterTile):
            raise TypeError("warpgroup.load_async dst must be a RegisterTile")
        if not isinstance(src, TensorTile):
            raise TypeError("warpgroup.load_async src must be a TensorTile")
        builder = current_builder()
        _mark_kittens(builder)
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::warpgroup::load_async", [dst._expr, src._expr],
        )))

    @staticmethod
    def sync(barrier_id : int = 1) -> None:
        """
        Emit ``kittens::warpgroup::sync(<id>);`` — block-level
        barrier shared by exactly the 4 warps of this warpgroup.
        """
        builder = current_builder()
        _mark_kittens(builder)
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::warpgroup::sync", [ir.Const(int(barrier_id))],
        )))

    @staticmethod
    def laneid() -> Tracer:
        """
        ``kittens::warpgroup::laneid()`` — this thread's index within the
        warpgroup (0..127). Returned as a Tracer so it can be used in
        arithmetic / comparisons.
        """
        builder = current_builder()
        _mark_kittens(builder)
        name = builder.fresh_name("wglane")
        builder.add_stmt(ir.Assign(
            name=name, cuda_type="int",
            value=ir.Call("kittens::warpgroup::laneid", []),
        ))
        return Tracer(ir.Var(name), dt.int32)


warpgroup = _WarpgroupNS()


def _mark_tma(builder : KernelBuilder) -> None:
    """
    Flag the kernel as a TMA user. Switches the shared-memory allocator
    from ``shared_allocator`` to ``tma_swizzle_allocator``, which lays
    tiles out with the alignment TMA requires.
    """
    _mark_kittens(builder)
    builder.uses_tma = True


class _TmaNS:
    """
    ``kittens::tma::*`` wrappers — async global ↔ shared copies that use
    Hopper/Blackwell's Tensor Memory Accelerator. Requires a
    ``kittens.Global[dtype, (TILE_R, TILE_C)]`` parameter so a TMA
    descriptor can be encoded host-side.
    """

    @staticmethod
    def expect_bytes(sem : Semaphore, n_bytes) -> None:
        """
        Emit ``kittens::tma::expect_bytes(sem, n_bytes);``. Tells the
        semaphore how many bytes to wait for before flipping its phase.
        """
        if not isinstance(sem, Semaphore):
            raise TypeError("tma.expect_bytes expects a Semaphore")
        builder = current_builder()
        _mark_tma(builder)
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::tma::expect_bytes", [sem._expr, _to_expr(n_bytes)],
        )))

    @staticmethod
    def load_async(
        dst   : SharedTile,
        src   : KittensGlobalTracer,
        coord : tuple,
        sem   : Semaphore,
    ) -> None:
        """
        Emit ``kittens::tma::load_async(stile, gl, {0,0,row,col}, sem);``.
        """
        if not isinstance(dst, SharedTile):
            raise TypeError("tma.load_async dst must be a SharedTile")
        if not isinstance(src, KittensGlobalTracer):
            raise TypeError("tma.load_async src must be a kittens.Global parameter")
        if not isinstance(sem, Semaphore):
            raise TypeError("tma.load_async sem must be a Semaphore")
        builder = current_builder()
        _mark_tma(builder)
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::tma::load_async",
            [dst._expr, src._expr, _coord_expr(coord), sem._expr],
        )))

    @staticmethod
    def store_async(
        dst   : KittensGlobalTracer,
        src   : SharedTile,
        coord : tuple,
    ) -> None:
        """
        Emit ``kittens::tma::store_async(gl, stile, {0,0,row,col});``.
        """
        if not isinstance(dst, KittensGlobalTracer):
            raise TypeError("tma.store_async dst must be a kittens.Global parameter")
        if not isinstance(src, SharedTile):
            raise TypeError("tma.store_async src must be a SharedTile")
        builder = current_builder()
        _mark_tma(builder)
        for p in builder.params:
            if p.kind == "kittens_global" and p.name == dst._param_name:
                p.written = True
                break
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::tma::store_async",
            [dst._expr, src._expr, _coord_expr(coord)],
        )))

    @staticmethod
    def store_async_read_wait() -> None:
        """
        Emit ``kittens::tma::store_async_read_wait();`` — wait for all
        outstanding TMA stores issued by this thread to finish reading
        their source shared tiles (so the tiles may be safely reused).
        """
        builder = current_builder()
        _mark_tma(builder)
        builder.add_stmt(ir.ExprStmt(ir.Call(
            "kittens::tma::store_async_read_wait", [],
        )))


tma = _TmaNS()

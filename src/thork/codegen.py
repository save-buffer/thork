from typing import Dict, List, Optional, Tuple

from . import ir
from .tracer import KernelBuilder


# Per-line source map: maps an output line (1-indexed) to a Python source
# location (filename, lineno).
SourceMap = Dict[int, Tuple[str, int]]


_PREC = {
    "||" : 1,
    "&&" : 2,
    "|"  : 3,
    "^"  : 4,
    "&"  : 5,
    "==" : 6, "!=" : 6,
    "<"  : 7, "<=" : 7, ">" : 7, ">=" : 7,
    "<<" : 8, ">>" : 8,
    "+"  : 9, "-"  : 9,
    "*"  : 10, "/" : 10, "%" : 10,
}


def _format_const(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if value < 0 or value <= 2147483647:
            return str(value)
        if value <= 4294967295:
            return f"{value}u"
        return f"{value}ull"
    if isinstance(value, float):
        text = repr(value)
        if "." not in text and "e" not in text and "n" not in text:
            text += ".0"
        return text + "f"
    raise TypeError(f"Cannot format constant of type {type(value).__name__}: {value!r}")


def format_expr(expr : ir.Expr, parent_prec : int = 0) -> str:
    if isinstance(expr, ir.Var):
        return expr.name
    if isinstance(expr, ir.Const):
        return _format_const(expr.value)
    if isinstance(expr, ir.Load):
        return f"{format_expr(expr.ptr, 100)}[{format_expr(expr.index, 0)}]"
    if isinstance(expr, ir.Member):
        return f"{format_expr(expr.operand, 100)}.{expr.field}"
    if isinstance(expr, ir.BinOp):
        prec = _PREC.get(expr.op, 0)
        s = f"{format_expr(expr.lhs, prec)} {expr.op} {format_expr(expr.rhs, prec + 1)}"
        if prec < parent_prec:
            s = f"({s})"
        return s
    if isinstance(expr, ir.UnaryOp):
        return f"{expr.op}{format_expr(expr.operand, 11)}"
    if isinstance(expr, ir.AddrOf):
        return f"&{format_expr(expr.operand, 11)}"
    if isinstance(expr, ir.Cast):
        return f"static_cast<{expr.dtype.cuda}>({format_expr(expr.operand, 0)})"
    if isinstance(expr, ir.Call):
        args = ", ".join(format_expr(a, 0) for a in expr.args)
        return f"{expr.func}({args})"
    if isinstance(expr, ir.MethodCall):
        obj_str = format_expr(expr.obj, 100)
        if expr.template_args:
            targs = []
            for t in expr.template_args:
                if isinstance(t, ir.Expr):
                    targs.append(format_expr(t, 0))
                else:
                    targs.append(str(t))
            tmpl = f"<{', '.join(targs)}>"
        else:
            tmpl = ""
        args = ", ".join(format_expr(a, 0) for a in expr.args)
        return f"{obj_str}.{expr.method}{tmpl}({args})"
    if isinstance(expr, ir.Raw):
        return expr.text
    raise TypeError(f"Unknown expression node: {type(expr).__name__}")


def _format_for_step(var_name : str, step : ir.Expr) -> str:
    if isinstance(step, ir.Const) and step.value == 1:
        return f"{var_name}++"
    if isinstance(step, ir.Const) and step.value == -1:
        return f"{var_name}--"
    return f"{var_name} += {format_expr(step, 0)}"


def _line_count(text : str) -> int:
    """
    Number of lines in ``text`` (1 for a string with no newlines).
    """
    return text.count("\n") + 1


def format_stmt(stmt : ir.Stmt, indent : int = 4) -> Tuple[str, Dict[int, tuple]]:
    """
    Format a statement.

    Returns (text, locs) where ``locs`` maps a 0-indexed line within ``text``
    to a Python source location.
    """
    pad = " " * indent
    self_loc = getattr(stmt, "loc", None)
    locs : Dict[int, tuple] = {}
    if self_loc:
        locs[0] = self_loc

    if isinstance(stmt, ir.Store):
        text = (
            f"{pad}{format_expr(stmt.ptr, 100)}[{format_expr(stmt.index, 0)}] = "
            f"{format_expr(stmt.value, 0)};"
        )
        return text, locs
    if isinstance(stmt, ir.Assign):
        return f"{pad}{stmt.cuda_type} {stmt.name} = {format_expr(stmt.value, 0)};", locs
    if isinstance(stmt, ir.Update):
        return f"{pad}{stmt.name} {stmt.op} {format_expr(stmt.value, 0)};", locs
    if isinstance(stmt, ir.ExprStmt):
        return f"{pad}{format_expr(stmt.expr, 0)};", locs
    if isinstance(stmt, ir.SharedDecl):
        dims = "".join(f"[{d}]" for d in stmt.shape)
        return f"{pad}__shared__ {stmt.cuda_type} {stmt.name}{dims};", locs
    if isinstance(stmt, ir.DefaultDecl):
        return f"{pad}{stmt.cuda_type} {stmt.name};", locs
    if isinstance(stmt, ir.ConstructorDecl):
        args = ", ".join(format_expr(a, 0) for a in stmt.args)
        return f"{pad}{stmt.cuda_type} {stmt.name}({args});", locs
    if isinstance(stmt, ir.Break):
        return f"{pad}break;", locs
    if isinstance(stmt, ir.Continue):
        return f"{pad}continue;", locs
    if isinstance(stmt, ir.Return):
        if stmt.value is None:
            return f"{pad}return;", locs
        return f"{pad}return {format_expr(stmt.value, 0)};", locs

    if isinstance(stmt, ir.ForLoop):
        header = (
            f"{pad}for (unsigned int {stmt.var_name} = {format_expr(stmt.start, 0)}; "
            f"{stmt.var_name} < {format_expr(stmt.end, 0)}; "
            f"{_format_for_step(stmt.var_name, stmt.step)})"
        )
        body_text, body_locs = format_stmts(stmt.body, indent + 4)
        if not body_text:
            body_text = f"{pad}    // empty"
        text = f"{header}\n{pad}{{\n{body_text}\n{pad}}}"
        for rl, loc in body_locs.items():
            locs[2 + rl] = loc
        return text, locs

    if isinstance(stmt, ir.WhileLoop):
        header = f"{pad}while ({format_expr(stmt.cond, 0)})"
        body_text, body_locs = format_stmts(stmt.body, indent + 4)
        if not body_text:
            body_text = f"{pad}    // empty"
        text = f"{header}\n{pad}{{\n{body_text}\n{pad}}}"
        for rl, loc in body_locs.items():
            locs[2 + rl] = loc
        return text, locs

    if isinstance(stmt, ir.IfStmt):
        header = f"{pad}if ({format_expr(stmt.cond, 0)})"
        then_text, then_locs = format_stmts(stmt.then_body, indent + 4)
        if not then_text:
            then_text = f"{pad}    // empty"
        text = f"{header}\n{pad}{{\n{then_text}\n{pad}}}"
        for rl, loc in then_locs.items():
            locs[2 + rl] = loc
        if stmt.else_body is not None:
            then_lines = _line_count(text)
            else_text, else_locs = format_stmts(stmt.else_body, indent + 4)
            if not else_text:
                else_text = f"{pad}    // empty"
            text += f"\n{pad}else\n{pad}{{\n{else_text}\n{pad}}}"
            else_offset = then_lines + 2
            for rl, loc in else_locs.items():
                locs[else_offset + rl] = loc
        return text, locs

    raise TypeError(f"Unknown statement node: {type(stmt).__name__}")


def format_stmts(stmts : List[ir.Stmt], indent : int = 4) -> Tuple[str, Dict[int, tuple]]:
    """
    Format a list of statements.

    Returns (text, locs) where ``locs`` maps a 0-indexed line within ``text``
    to a Python source location.
    """
    if not stmts:
        return "", {}
    parts = []
    locs : Dict[int, tuple] = {}
    current_line = 0
    for stmt in stmts:
        text, sub_locs = format_stmt(stmt, indent)
        parts.append(text)
        for rl, loc in sub_locs.items():
            locs[current_line + rl] = loc
        current_line += _line_count(text)
    return "\n".join(parts), locs


def emit_device_fn(df) -> Tuple[str, Dict[int, tuple]]:
    """
    Emit a complete __device__ function definition in Allman style.

    Returns (text, locs) with locs keyed by line index within the device-fn
    text (0 = signature line).
    """
    params_block = ", ".join(df._param_strs)
    body_text, body_locs = format_stmts(df._stmts, indent=4)
    if not body_text:
        body_text = "    // empty"
    sig_line = f"__device__ {df._return_type} {df._name}({params_block})"
    text = f"{sig_line}\n{{\n{body_text}\n}}"
    locs : Dict[int, tuple] = {}
    for rl, loc in body_locs.items():
        locs[2 + rl] = loc
    return text, locs


def _format_kernel_param(p : ir.Param) -> str:
    """
    Render a single pointer/constant kernel parameter declaration.

    Attribute params are NOT rendered here — they become local declarations
    at the top of the kernel body, since CUDA doesn't pass them as args.
    """
    if p.kind == "pointer":
        return f"    {p.dtype.cuda} *{p.name}"
    if p.kind == "constant":
        return f"    {p.cuda_name or p.dtype.cuda} {p.name}"
    raise ValueError(f"Param kind {p.kind!r} is not a kernel argument")


def _attribute_init_text(p : ir.Param) -> str:
    """
    Build the initializer expression for an attribute parameter.

    For a scalar attribute like BlockIdx with vec_size==1, picks the .x
    component. For a vector attribute (e.g. Uint3[BlockIdx]) renders as
    ``uint3{blockIdx.x, blockIdx.y, blockIdx.z}`` so .x/.y/.z accesses on
    the traced value work.
    """
    assert p.attribute is not None
    base = p.attribute.cuda_expr

    # Scalars like WarpSize / LaneId / WarpId have no .x/.y/.z components.
    if base.startswith("(") or base == "warpSize":
        return base

    if p.vec_size == 1:
        return f"{base}.x"
    fields = ("x", "y", "z", "w")[: p.vec_size]
    fields_expr = ", ".join(f"{base}.{f}" for f in fields)
    return f"{p.cuda_name}{{{fields_expr}}}"


def emit_kernel(builder : KernelBuilder) -> Tuple[str, SourceMap]:
    """
    Render the full CUDA source for a kernel.

    Returns ``(source, source_map)`` where ``source_map`` maps a 1-indexed
    output line in ``source`` to a ``(python_filename, python_lineno)`` tuple,
    for any line that was emitted with a tracked source location.
    """
    lines : List[str] = []
    source_map : SourceMap = {}

    def add_line(text : str, loc : Optional[tuple] = None) -> None:
        lines.append(text)
        if loc:
            source_map[len(lines)] = loc

    def add_block(text : str, locs : Dict[int, tuple]) -> None:
        start = len(lines) + 1
        for ln in text.rstrip("\n").split("\n"):
            lines.append(ln)
        for rl, loc in locs.items():
            source_map[start + rl] = loc

    for path, system in builder.includes:
        add_line(f"#include <{path}>" if system else f'#include "{path}"')
    add_line("")

    for ns in builder.usings:
        add_line(f"using namespace {ns};")
    if builder.usings:
        add_line("")

    for df in builder.device_functions:
        df_text, df_locs = emit_device_fn(df)
        add_block(df_text, df_locs)
        add_line("")

    bindable_params = [p for p in builder.params if p.kind in ("pointer", "constant")]
    attribute_params = [p for p in builder.params if p.kind == "attribute"]

    param_lines = [_format_kernel_param(p) for p in bindable_params]

    add_line(f'extern "C" __global__ void {builder.name}(')
    if param_lines:
        for line in param_lines[:-1]:
            add_line(line + ",")
        add_line(param_lines[-1] + ")")
    else:
        add_line(")")
    add_line("{")

    for p in attribute_params:
        decl_type = p.cuda_name if p.vec_size > 1 else p.dtype.cuda
        add_line(f"    {decl_type} {p.name} = {_attribute_init_text(p)};")

    body_text, body_locs = format_stmts(builder.stmts, indent=4)
    if body_text:
        add_block(body_text, body_locs)
    elif not attribute_params:
        add_line("    // empty kernel")

    add_line("}")

    source = "\n".join(lines) + "\n"
    return source, source_map

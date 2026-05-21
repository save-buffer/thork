from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

from . import dtypes as dt
from .types import ThreadAttribute


class Expr:
    pass


@dataclass
class Var(Expr):
    name : str


@dataclass
class Const(Expr):
    value : Union[int, float, bool]


@dataclass
class Load(Expr):
    ptr   : Expr
    index : Expr


@dataclass
class Member(Expr):
    """
    Access a field on a vector or struct expression, e.g. ``tid.x``.
    """

    operand : Expr
    field   : str


@dataclass
class BinOp(Expr):
    op  : str
    lhs : Expr
    rhs : Expr


@dataclass
class UnaryOp(Expr):
    op      : str
    operand : Expr


@dataclass
class AddrOf(Expr):
    """
    Address-of expression: ``&<expr>``.
    """

    operand : Expr


@dataclass
class Cast(Expr):
    dtype   : dt.Dtype
    operand : Expr


@dataclass
class Call(Expr):
    """
    A function-call expression: ``func(arg0, arg1, ...)``.
    """

    func : str
    args : List["Expr"] = field(default_factory=list)


@dataclass
class MethodCall(Expr):
    """
    A method-call expression: ``obj.method<template_args>(args)``.

    Template args may be Expr nodes (rendered with format_expr) or plain
    Python ints/strs (rendered with str()). Empty template_args produces
    no ``<...>`` suffix.
    """

    obj           : Expr
    method        : str
    template_args : List = field(default_factory=list)
    args          : List["Expr"] = field(default_factory=list)


@dataclass
class Raw(Expr):
    """
    A verbatim chunk of CUDA source embedded as an expression.
    """

    text : str


class Stmt:
    pass


@dataclass
class Store(Stmt):
    ptr   : Expr
    index : Expr
    value : Expr


@dataclass
class Assign(Stmt):
    """
    Declare-and-assign a local: ``<cuda_type> name = <expr>;``.
    """

    name      : str
    cuda_type : str
    value     : Expr


@dataclass
class Update(Stmt):
    """
    Re-assign an existing local: ``name <op>= <expr>;`` (op may be ``=``).
    """

    name  : str
    op    : str
    value : Expr


@dataclass
class ForLoop(Stmt):
    """
    A ``for (unsigned int i = start; i < end; i += step) { ... }`` loop.
    """

    var_name : str
    start    : Expr
    end      : Expr
    step     : Expr
    body     : List[Stmt] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    """
    An ``if (cond) { ... } else { ... }`` statement; ``else_body`` may be None.
    """

    cond      : Expr
    then_body : List[Stmt] = field(default_factory=list)
    else_body : Optional[List[Stmt]] = None


@dataclass
class WhileLoop(Stmt):
    """
    A ``while (cond) { ... }`` loop.
    """

    cond : Expr
    body : List[Stmt] = field(default_factory=list)


@dataclass
class Break(Stmt):
    pass


@dataclass
class Continue(Stmt):
    pass


@dataclass
class Return(Stmt):
    """
    Function return. ``value`` is None for void functions.
    """

    value : Optional[Expr] = None


@dataclass
class ExprStmt(Stmt):
    """
    A statement whose effect is just evaluating an expression
    (typically a Call): ``<expr>;``.
    """

    expr : Expr


@dataclass
class SharedDecl(Stmt):
    """
    A shared-memory array declaration:
    ``__shared__ <cuda_type> <name>[D0][D1]...;``.

    All dimensions must be Python ints (CUDA requires constexpr sizes for
    statically allocated shared arrays).
    """

    name      : str
    cuda_type : str
    shape     : List[int]


@dataclass
class DefaultDecl(Stmt):
    """
    Default-initialized declaration: ``<cuda_type> <name>;``.
    """

    name      : str
    cuda_type : str


@dataclass
class ConstructorDecl(Stmt):
    """
    Constructor-initialized declaration: ``<cuda_type> <name>(arg0, arg1, ...);``.
    """

    name      : str
    cuda_type : str
    args      : List[Expr] = field(default_factory=list)


@dataclass
class Param:
    name       : str
    kind       : str
    dtype      : dt.Dtype
    cuda_name  : Optional[str] = None
    vec_size   : int = 1
    attribute  : Optional[Type[ThreadAttribute]] = None
    written    : bool = False

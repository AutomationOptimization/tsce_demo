"""ODE solver with sandboxed derivative expressions.

The ``prepare_inputs`` function compiles a derivative expression into a
callable. Expressions are parsed with :mod:`ast` and only a restricted
subset of Python syntax is permitted. Currently allowed are simple
arithmetic operators and calls to selected :mod:`math` functions using the
variables ``y`` and ``t``. Any other constructs will raise
:class:`ValueError`.
"""

from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Callable



_ALLOWED_FUNCS = {k: getattr(math, k) for k in (
    "sin", "cos", "tan", "exp", "log", "sqrt"
)}
_ALLOWED_NAMES = {"y", "t"} | set(_ALLOWED_FUNCS.keys()) | {"math"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate(node: ast.AST) -> None:
    """Recursively validate ``node`` against the allowed subset."""
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError("Disallowed operator")
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError("Disallowed unary operator")
        _validate(node.operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
            ):
                raise ValueError("Only math.* calls allowed")
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:  # pragma: no cover - not hit in tests
            raise ValueError("Invalid function")
        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Function {func_name!r} not allowed")
        for arg in node.args:
            _validate(arg)
    elif isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Name {node.id!r} not allowed")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants allowed")
    else:
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def prepare_inputs(code: str) -> Callable[[float, float], float]:
    """Compile ``code`` into ``f(y, t)`` after validation."""
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        raise ValueError("Code must be a single return statement")
    _validate(tree.body[0].value)

    func_def = ast.FunctionDef(
        name="_f",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="y"), ast.arg(arg="t")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=tree.body,
        decorator_list=[],
    )
    mod = ast.Module([func_def], type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace: dict[str, Callable] = {"math": math, **_ALLOWED_FUNCS}
    exec(compile(mod, "<ast>", "exec"), namespace)
    return namespace["_f"]


def run_ode(
    code: str,
    y0: float,
    t0: float,
    t1: float,
    dt: float,
    *,
    out_dir: str = "results",
) -> Path:
    """Solve ``dy/dt`` defined by ``code`` from ``t0`` to ``t1``."""
    if t1 - t0 > 10:
        raise RuntimeError("Time span too large")
    func = prepare_inputs(code)
    t_values = []
    y_values = []
    y = y0
    t = t0
    while t <= t1 + 1e-12:
        t_values.append(t)
        y_values.append(y)
        y += dt * func(y, t)
        t += dt

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    data_file = out / "ode_results.json"
    meta_file = out / "ode_results.meta.json"
    data_file.write_text(json.dumps({"t": t_values, "y": y_values}))
    meta_file.write_text(
        json.dumps({"solver": "odeint", "timestamp": time.strftime("%Y-%m-%d")})
    )
    return data_file


__all__ = ["prepare_inputs", "run_ode"]

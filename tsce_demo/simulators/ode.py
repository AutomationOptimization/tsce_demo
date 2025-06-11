"""Simple ODE integration demo using scipy.odeint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import FunctionType


def prepare_inputs(func_str: str) -> FunctionType:
    """Compile ``func_str`` into a Python function returning dy/dt."""
    if "return" not in func_str:
        raise ValueError("ODE function must contain a return statement")
    ns: dict[str, object] = {}
    code = "def f(y, t):\n    " + func_str.replace("\n", "\n    ")
    try:
        exec(code, {}, ns)
    except Exception as exc:  # pragma: no cover - compile error
        raise ValueError(f"Invalid ODE function: {exc}") from exc
    func = ns.get("f")
    if not isinstance(func, FunctionType):
        raise ValueError("Failed to compile ODE function")
    return func


def run_ode(
    func_str: str,
    y0: float,
    t_start: float,
    t_end: float,
    step: float,
    *,
    out_dir: str,
    precision: str = "float64",
    seed: int | None = None,
) -> Path:
    """Run the ODE solver and return the results path."""
    if seed is not None:
        import numpy as np

        np.random.seed(seed)
    func = prepare_inputs(func_str)
    dtype = "float32" if precision == "float32" else "float64"

    import numpy as np
    from scipy.integrate import odeint

    t = np.arange(t_start, t_end + step, step, dtype=dtype)
    if len(t) > 10000:
        raise RuntimeError("t_span too large")
    y0 = np.asarray([y0], dtype=dtype)

    start = time.time()
    y = odeint(func, y0, t)
    runtime_ms = int((time.time() - start) * 1000)

    result = Path(out_dir) / "ode_results.json"
    meta = Path(out_dir) / "ode_results.meta.json"
    with result.open("w", encoding="utf-8") as f:
        json.dump({"t": t.tolist(), "y": y[:, 0].astype(dtype).tolist()}, f)
    with meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "solver": "odeint",
                "dtype": dtype,
                "n_points": len(t),
                "runtime_ms": runtime_ms,
            },
            f,
        )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ODE simulator")
    p.add_argument("--func", required=True)
    p.add_argument("--y0", type=float, required=True)
    p.add_argument("--t-span", nargs=3, type=float, metavar=("START", "END", "STEP"), required=True)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--precision", choices=["float32", "float64"], default="float64")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_ode(
        args.func,
        args.y0,
        args.t_span[0],
        args.t_span[1],
        args.t_span[2],
        out_dir=args.out_dir,
        precision=args.precision,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations
import subprocess
from pathlib import Path
import sys
import time

from .base_agent import BaseAgent


def run_simulation(path: str) -> str:
    """Execute a Python file and store the result in ``results/``.

    Parameters
    ----------
    path: str
        Path to the Python file to execute.

    Returns
    -------
    str
        Path to the log file containing stdout and stderr.
    """
    script = Path(path)
    if not script.is_file():
        raise FileNotFoundError(f"No such file: {script}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = out_dir / f"{script.stem}_{timestamp}.log"

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
        f.write(f"\n--- return code: {proc.returncode} ---\n")

    return str(log_file)


class Simulator(BaseAgent):
    """Execute Python scripts and record the output."""

    def __init__(self, *, log_dir: str | None = None) -> None:
        super().__init__(name="Simulator", log_dir=log_dir)

    def send_message(self, message: str) -> str:  # pragma: no cover
        return self.act(message)

    # ------------------------------------------------------------------
    def act(self, path: str) -> str:
        """Run ``path`` and return the log file location."""
        return run_simulation(path)


__all__ = ["Simulator", "run_simulation"]

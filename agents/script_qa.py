from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from .base_agent import BaseAgent


def run_tests(path: str | os.PathLike) -> Tuple[bool, str]:
    """Run tests located at ``path``.

    ``path`` may point to a single file or a directory containing tests.  If
    the target looks like a test suite (a directory or a ``*_test.py`` file),
    ``pytest`` is invoked.  Otherwise the file is executed directly with the
    current Python interpreter.

    Before executing, the file is checked for a ``# GOLDEN_THREAD`` marker.  If
    missing, the tests are skipped and an error message is returned.

    Returns a tuple ``(success, details)`` where ``success`` is ``True`` when
    the command exits with a zero status code and ``details`` contains the
    combined stdout/stderr output.
    """
    target = Path(path)

    if target.is_file():
        try:
            if "# GOLDEN_THREAD" not in target.read_text():
                return False, "Missing '# GOLDEN_THREAD' marker"
        except Exception as exc:  # pragma: no cover - unlikely to fail
            return False, f"Could not read {target}: {exc}"

    if target.is_dir() or target.name.endswith("_test.py"):
        cmd = ["pytest", str(target), "-q"]
    else:
        cmd = [sys.executable, str(target)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output


class ScriptQA(BaseAgent):
    """Agent that executes a script's tests."""

    def __init__(self, *, log_dir: str | None = None) -> None:
        super().__init__(name="ScriptQA", log_dir=log_dir)

    def send_message(self, message: str) -> str:  # pragma: no cover
        success, output = self.act(message)
        return output if success else output

    # ------------------------------------------------------------------
    def act(self, path: str | os.PathLike) -> Tuple[bool, str]:
        """Run tests at ``path`` and return ``(success, details)``."""
        return run_tests(path)


__all__ = ["ScriptQA", "run_tests"]

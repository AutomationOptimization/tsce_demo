"""Run arbitrary Python scripts and capture their stdout/stderr.

Typical usage::

    output = RunScriptTool()("my_script.py")

Returns
-------
str
    Combined standard output and standard error text, or an error message.
"""

import subprocess


class RunScriptTool:
    """Execute a Python script and return its output."""

    def __call__(self, path: str) -> str:
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, text=True, check=False, timeout=30
            )
        except FileNotFoundError:
            return f"Error: {path} not found"
        except Exception as exc:  # pragma: no cover - execution issues
            return f"Error: {exc}"

        return proc.stdout + proc.stderr

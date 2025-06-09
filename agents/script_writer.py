from __future__ import annotations

import re
import textwrap
import uuid

from .base_agent import BaseAgent


class ScriptWriter(BaseAgent):
    """Return a short Python snippet for a textual request."""

    def __init__(self, *, log_dir: str | None = None) -> None:
        super().__init__(name="ScriptWriter", log_dir=log_dir)

    def send_message(self, message: str) -> tuple[str, str]:  # pragma: no cover
        return self.act(message)

    # ------------------------------------------------------------------
    def act(self, request: str) -> tuple[str, str]:
        """Return a short Python code snippet for ``request``.

        The implementation is intentionally lightweight and recognises only a
        few common patterns.  If the request cannot be handled, a comment
        describing the limitation is returned instead of code.
        """
        lower = request.lower()
        gid = uuid.uuid4().hex

        if "hello" in lower and "world" in lower:
            script = 'print("Hello, world!")'
            script = f"# GOLDEN_THREAD:{gid}\n{script}"
            return script, gid

        m = re.search(r"fibonacci(?: up to)? (\d+)", lower)
        if m:
            n = int(m.group(1))
            script = textwrap.dedent(f"""
                def fib(n):
                    a, b = 0, 1
                    result = []
                    for _ in range(n):
                        result.append(a)
                        a, b = b, a + b
                    return result

                print(fib({n}))
            """).strip()
            script = f"# GOLDEN_THREAD:{gid}\n{script}"
            return script, gid

        m = re.search(r"factorial(?: of)? (\d+)", lower)
        if m:
            n = int(m.group(1))
            script = textwrap.dedent(f"""
                def factorial(n):
                    return 1 if n <= 1 else n * factorial(n-1)

                print(factorial({n}))
            """).strip()
            script = f"# GOLDEN_THREAD:{gid}\n{script}"
            return script, gid

        script = f"# TODO: unable to generate script for: {request}"
        script = f"# GOLDEN_THREAD:{gid}\n{script}"
        return script, gid


__all__ = ["ScriptWriter"]

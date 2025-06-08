import re
import textwrap


def act(request: str) -> str:
    """Return a short Python code snippet for the given request.

    The implementation is intentionally lightweight and recognises only a few
    common patterns.  If the request cannot be handled, a comment describing
    the limitation is returned instead of code.
    """
    lower = request.lower()

    if "hello" in lower and "world" in lower:
        return 'print("Hello, world!")'

    m = re.search(r"fibonacci(?: up to)? (\d+)", lower)
    if m:
        n = int(m.group(1))
        return textwrap.dedent(f"""
            def fib(n):
                a, b = 0, 1
                result = []
                for _ in range(n):
                    result.append(a)
                    a, b = b, a + b
                return result

            print(fib({n}))
        """).strip()

    m = re.search(r"factorial(?: of)? (\d+)", lower)
    if m:
        n = int(m.group(1))
        return textwrap.dedent(f"""
            def factorial(n):
                return 1 if n <= 1 else n * factorial(n-1)

            print(factorial({n}))
        """).strip()

    return f"# TODO: unable to generate script for: {request}"

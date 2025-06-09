"""Simple file reader that returns the entire file contents.

Example::

    text = ReadFileTool()("notes.txt")

Returns
-------
str
    The text content of the file or an error description.
"""


class ReadFileTool:
    """Read a text file."""

    def __call__(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: {path} not found"
        except Exception as exc:
            return f"Error: {exc}"

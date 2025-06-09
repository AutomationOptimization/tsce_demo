"""Remove files safely.

Typical usage::

    DeleteFileTool()("temp.txt")

Returns
-------
str
    ``"Deleted <path>"`` on success or an error message if deletion fails.
"""

import os


class DeleteFileTool:
    """Delete a file from disk."""

    def __call__(self, path: str) -> str:
        try:
            os.remove(path)
            return f"Deleted {path}"
        except FileNotFoundError:
            return f"Error: {path} not found"
        except Exception as exc:
            return f"Error: {exc}"

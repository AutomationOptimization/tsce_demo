class CreateFileTool:
    """Create a new text file."""

    def __call__(self, path: str, content: str = "") -> str:
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(content)
            return f"Created {path}"
        except FileExistsError:
            return f"Error: {path} already exists"
        except Exception as exc:
            return f"Error: {exc}"

class EditFileTool:
    """Overwrite a text file with new content."""

    def __call__(self, path: str, content: str) -> str:
        try:
            with open(path, "r+", encoding="utf-8") as f:
                f.seek(0)
                f.write(content)
                f.truncate()
            return f"Updated {path}"
        except FileNotFoundError:
            return f"Error: {path} not found"
        except Exception as exc:
            return f"Error: {exc}"

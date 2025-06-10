"""Convenience package exposing all built-in tool classes.

Importing ``tools`` makes the individual helpers available directly::

    from tools import GoogleSearch, ReadFileTool

Each callable returns strings or lists describing its outcome as documented in
the respective modules.
"""

from .google_search import GoogleSearch
from .web_scrape import WebScrape
from .file_create import CreateFileTool
from .file_read import ReadFileTool
from .file_edit import EditFileTool
from .file_delete import DeleteFileTool
from .run_script import RunScriptTool

TOOL_CLASSES = {
    "google_search": GoogleSearch,
    "web_scrape": WebScrape,
    "create_file": CreateFileTool,
    "read_file": ReadFileTool,
    "edit_file": EditFileTool,
    "delete_file": DeleteFileTool,
    "run_script": RunScriptTool,
}


def use_tool(name: str, args: dict | None = None):
    """Instantiate and execute a tool by ``name`` with ``args``.

    Parameters
    ----------
    name:
        Tool identifier from ``TOOL_CLASSES``.
    args:
        Keyword arguments forwarded to the tool call.
    """
    cls = TOOL_CLASSES.get(name.lower())
    if not cls:
        raise ValueError(f"Unknown tool: {name}")
    args = args or {}
    return cls()(**args)

__all__ = [
    "GoogleSearch",
    "WebScrape",
    "CreateFileTool",
    "ReadFileTool",
    "EditFileTool",
    "DeleteFileTool",
    "RunScriptTool",
    "use_tool",
]

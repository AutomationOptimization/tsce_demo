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
from .literature_search.arxiv import ArxivSearch
from .literature_search.pubmed import PubMedSearch
from .bio import PubMedTool, ChEMBLTool
import os
from pathlib import Path
import numpy as np
import pandas as pd


def embed_text(text: str) -> list[float]:
    """Return a vector embedding for ``text`` using OpenAI or a local model."""
    try:  # pragma: no cover - requires network/API key
        import openai

        client = openai.OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding
    except Exception:  # pragma: no cover - fallback path
        from sentence_transformers import SentenceTransformer

        if not hasattr(embed_text, "_model"):
            embed_text._model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = embed_text._model.encode(text)
        return vec.tolist()

TOOL_CLASSES = {
    "google_search": GoogleSearch,
    "web_scrape": WebScrape,
    "create_file": CreateFileTool,
    "read_file": ReadFileTool,
    "edit_file": EditFileTool,
    "delete_file": DeleteFileTool,
    "run_script": RunScriptTool,
    "pubmedtool": PubMedTool,
    "chembltool": ChEMBLTool,
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


def memory_search(query: str, k: int = 5) -> list[str]:
    """Return the ``text`` fields most similar to ``query`` from ``logs/memory.parquet``."""
    mem_path = Path(os.getenv("MEMORY_PATH", "logs/memory.parquet"))
    if not mem_path.exists():
        return []
    df = pd.read_parquet(mem_path)
    if df.empty:
        return []
    q_vec = np.array(embed_text(query))
    mat = np.vstack(df["embedding"].to_list())
    denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q_vec) or 1e-12)
    scores = mat.dot(q_vec) / np.where(denom == 0, 1e-12, denom)
    idx = np.argsort(scores)[::-1][:k]
    return df.iloc[idx]["text"].tolist()

__all__ = [
    "GoogleSearch",
    "WebScrape",
    "CreateFileTool",
    "ReadFileTool",
    "EditFileTool",
    "DeleteFileTool",
    "RunScriptTool",
    "PubMedTool",
    "ChEMBLTool",
    "use_tool",
    "memory_search",
    "embed_text",
]

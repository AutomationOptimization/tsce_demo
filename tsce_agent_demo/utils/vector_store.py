"""Simple FAISS-based retrieval utilities."""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - optional dependency
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore
    FAISS = None  # type: ignore


def query(text: str, k: int = 5, index_dir: str = "vector_store") -> list[str]:
    """Return the contents of the top-k documents similar to ``text``."""
    if FAISS is None or OpenAIEmbeddings is None:
        return []
    path = Path(index_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Vector store directory '{index_dir}' not found. "
            "Run the ingest step to build it."
        )
    store = FAISS.load_local(str(path), OpenAIEmbeddings())
    docs, _ = store.similarity_search_with_score(text, k=k)
    return [d.page_content for d in docs]

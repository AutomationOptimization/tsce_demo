"""Simple FAISS-based retrieval utilities."""

from __future__ import annotations

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def query(text: str, k: int = 5, index_dir: str = "vector_store") -> list[str]:
    """Return the contents of the top-k documents similar to ``text``."""
    store = FAISS.load_local(index_dir, OpenAIEmbeddings())
    docs, _ = store.similarity_search_with_score(text, k=k)
    return [d.page_content for d in docs]

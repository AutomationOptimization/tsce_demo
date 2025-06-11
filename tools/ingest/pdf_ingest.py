"""Download referenced PDFs and store their texts as vectors."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def ingest_papers(papers: Iterable, index_dir: str = "vector_store") -> None:
    """Download and embed PDF documents referenced in *papers*.

    Parameters
    ----------
    papers:
        Iterable of objects with ``url`` and ``title`` attributes.
    index_dir:
        Directory where the FAISS index should be stored.
    """

    embeddings = OpenAIEmbeddings()
    store = FAISS(embeddings.embed_query, embeddings.embedding_size)
    Path(index_dir).mkdir(exist_ok=True)

    for paper in papers:
        url = getattr(paper, "url", "")
        title = getattr(paper, "title", "")
        if not url.endswith(".pdf"):
            continue

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            subprocess.check_call(["curl", "-L", url, "-o", pdf_path])
            txt = subprocess.check_output(["python", "-m", "pypdf", pdf_path])
            chunks = [txt[i : i + 1000] for i in range(0, len(txt), 1000)]
            store.add_texts(chunks, metadatas=[{"title": title}] * len(chunks))

    store.save_local(index_dir)

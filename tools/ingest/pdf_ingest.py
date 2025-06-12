"""Download referenced PDFs and store their texts as vectors."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import requests

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def _download_pdf(url: str, dest: Path, retries: int = 3, timeout: int = 10) -> bool:
    """Download *url* to *dest*, retrying on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code >= 400:
                logging.error("HTTP error %s while downloading %s", resp.status_code, url)
                return False
            dest.write_bytes(resp.content)
            return True
        except Exception as exc:  # pragma: no cover - network issues
            if attempt == retries - 1:
                logging.error("Download failed for %s: %s", url, exc)
            else:
                continue
    return False


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
            if not _download_pdf(url, pdf_path):
                continue
            txt = subprocess.check_output(["python", "-m", "pypdf", pdf_path])
            chunks = [txt[i : i + 1000] for i in range(0, len(txt), 1000)]
            store.add_texts(chunks, metadatas=[{"title": title}] * len(chunks))

    store.save_local(index_dir)

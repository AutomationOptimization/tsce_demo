"""Download referenced PDFs and store their texts as vectors."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import requests

try:  # pragma: no cover - optional dependency
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
except Exception:  # pragma: no cover - library not available
    OpenAIEmbeddings = None  # type: ignore
    FAISS = None  # type: ignore


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


def ingest_papers(
    papers: Iterable,
    index_dir: str = "vector_store",
    *,
    force: bool = False,
) -> None:
    """Download and embed PDF documents referenced in *papers*.

    Parameters
    ----------
    papers:
        Iterable of objects with ``url`` and ``title`` attributes.
    index_dir:
        Directory where the FAISS index should be stored.
    """

    embeddings = OpenAIEmbeddings()

    index_path = Path(index_dir)
    existing_titles: set[str] = set()
    existing_urls: set[str] = set()

    if not force and index_path.exists() and hasattr(FAISS, "load_local"):
        store = FAISS.load_local(str(index_path), embeddings)
        try:
            docs = getattr(store, "docstore")._dict.values()
            for doc in docs:
                meta = getattr(doc, "metadata", {})
                existing_titles.add(meta.get("title", ""))
                existing_urls.add(meta.get("url", ""))
        except Exception:  # pragma: no cover - best effort
            pass
    else:
        store = FAISS(embeddings.embed_query, embeddings.embedding_size)
        index_path.mkdir(exist_ok=True)

    for paper in papers:
        url = getattr(paper, "url", "")
        title = getattr(paper, "title", "")
        if not url.endswith(".pdf"):
            continue

        if url in existing_urls or title in existing_titles:
            logging.info("Skipping already indexed paper: %s", url or title)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            if not _download_pdf(url, pdf_path):
                continue
            txt = subprocess.check_output(["python", "-m", "pypdf", pdf_path])
            chunks = [txt[i : i + 1000] for i in range(0, len(txt), 1000)]
            metadata = {"title": title, "url": url}
            store.add_texts(chunks, metadatas=[metadata] * len(chunks))
            existing_titles.add(title)
            existing_urls.add(url)

    store.save_local(str(index_path))


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest paper PDFs")
    parser.add_argument("papers", help="JSON file with list of {'url':..., 'title':...}")
    parser.add_argument("--index-dir", default="vector_store", help="Index directory")
    parser.add_argument("--force", action="store_true", help="Force reindexing")
    args = parser.parse_args()

    with open(args.papers) as fh:
        paper_list = json.load(fh)

    ingest_papers(paper_list, index_dir=args.index_dir, force=args.force)

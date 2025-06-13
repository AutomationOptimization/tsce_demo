from __future__ import annotations

"""Index PubMed abstracts into OpenSearch."""

import os
from typing import Iterable, Dict, Any

from opensearchpy import OpenSearch, helpers

from tools.literature_search.pubmed import PubMedSearch
from tools import embed_text

INDEX_NAME = "pubmed_fibrosis"
DEFAULT_QUERY = "fibrosis"
DEFAULT_SIZE = 40000


def _fetch_papers(query: str, k: int) -> Iterable:
    search = PubMedSearch()
    return search.run(query, k=k)


def _iter_docs(papers: Iterable) -> Iterable[Dict[str, Any]]:
    for idx, paper in enumerate(papers):
        abstr = getattr(paper, "abstract", "")
        vec = embed_text(abstr)
        yield {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": getattr(paper, "pmid", idx),
            "title": getattr(paper, "title", ""),
            "abstract": abstr,
            "embedding": vec,
        }


def build_index(query: str = DEFAULT_QUERY, *, size: int = DEFAULT_SIZE, host: str = "localhost") -> None:
    client = OpenSearch([{"host": host, "port": 9200}])
    if not client.indices.exists(INDEX_NAME):
        client.indices.create(
            INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "embedding": {"type": "knn_vector", "dimension": len(embed_text("test"))}
                    }
                }
            },
        )
    papers = _fetch_papers(query, k=size)
    helpers.bulk(client, _iter_docs(papers))
    client.indices.refresh(INDEX_NAME)


if __name__ == "__main__":  # pragma: no cover
    q = os.getenv("QUERY", DEFAULT_QUERY)
    sz = int(os.getenv("SIZE", DEFAULT_SIZE))
    build_index(q, size=sz)

"""PubMed literature search via NCBI E-utilities.

Requests are throttled to three per second without an API key. Setting the
``NCBI_API_KEY`` environment variable increases the limit to around ten
requests per second. The key is passed using the ``api_key`` parameter on both
the ``esearch`` and ``esummary`` endpoints.
"""

import os, requests
from typing import List
from tsce_agent_demo.models.research_task import PaperMeta
from .base import LiteratureSearchTool

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

class PubMedSearch(LiteratureSearchTool):
    def run(self, query: str, k: int = 10) -> List[PaperMeta]:
        api_key = os.getenv("NCBI_API_KEY")

        params = {"db": "pubmed", "retmode": "json", "term": query, "retmax": k}
        if api_key:
            params["api_key"] = api_key
        ids = requests.get(
            f"{_EUTILS}esearch.fcgi",
            params=params,
            timeout=30,
        ).json()["esearchresult"]["idlist"]

        params = {"db": "pubmed", "retmode": "json", "id": ",".join(ids)}
        if api_key:
            params["api_key"] = api_key
        summaries = requests.get(
            f"{_EUTILS}esummary.fcgi",
            params=params,
            timeout=30,
        ).json()["result"]

        papers: List[PaperMeta] = []
        for pid in ids:
            meta = summaries[pid]
            papers.append(
                PaperMeta(
                    title=meta["title"],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                    authors=[a["name"] for a in meta["authors"]],
                    year=int(meta["pubdate"][:4]),
                    abstract=meta.get("elocationid", ""),
                )
            )
        return papers

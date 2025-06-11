import os, requests
from typing import List
from tsce_agent_demo.models.research_task import PaperMeta
from .base import LiteratureSearchTool

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

class PubMedSearch(LiteratureSearchTool):
    def run(self, query: str, k: int = 10) -> List[PaperMeta]:
        api_key = os.getenv("NCBI_API_KEY")
        ids = requests.get(
            f"{_EUTILS}esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "term": query, "retmax": k, "api_key": api_key},
            timeout=30,
        ).json()["esearchresult"]["idlist"]

        summaries = requests.get(
            f"{_EUTILS}esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(ids), "api_key": api_key},
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

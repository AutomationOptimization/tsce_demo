"""Query PubMed abstracts via Entrez."""

from __future__ import annotations

from typing import Iterable

from Bio import Entrez


class PubMedTool:
    """Return PubMed abstracts for a query string."""

    def __call__(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=top_k)
            ids = Entrez.read(search_handle)["IdList"]
            search_handle.close()
            if not ids:
                return []
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
            records = Entrez.read(fetch_handle)
            fetch_handle.close()
        except Exception as exc:  # pragma: no cover - network issues
            return [{"error": str(exc)}]

        results: list[dict] = []
        for article in records.get("PubmedArticle", []):
            citation = article["MedlineCitation"]
            art = citation["Article"]
            pmid = citation["PMID"]
            title = art.get("ArticleTitle", "")
            abstract = " ".join(art.get("Abstract", {}).get("AbstractText", []))
            results.append(
                {"pmid": str(pmid), "title": str(title), "abstract": str(abstract)}
            )
        return results[:top_k]

"""Retrieve quick search result titles using Google's public endpoint.

Example::

    GoogleSearch()("python testing", num_results=3)

Returns
-------
list[str]
    A list of result titles or a single-element list with an error message.
"""

import re
import requests


class GoogleSearch:
    """Lightweight wrapper around Google search."""

    def __call__(self, query: str, num_results: int = 5) -> list[str]:
        if query.lower().startswith("pubmed:"):
            q = query.split(":", 1)[1].strip()
            return _entrez_pubmed(q, num_results)

        url = "https://www.google.com/search"
        try:
            resp = requests.get(
                url,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network issues
            return [f"Search error: {exc}"]

        titles = re.findall(r"<h3[^>]*>(.+?)</h3>", resp.text, re.DOTALL)
        clean = [re.sub(r"<.*?>", "", t).strip() for t in titles]
        return clean[:num_results]


def _entrez_pubmed(query: str, n: int) -> list[str]:
    """Return PubMed article titles matching ``query``."""
    from Bio import Entrez
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=n)
        ids = Entrez.read(handle)["IdList"]
        handle.close()

        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as exc:  # pragma: no cover - network issues
        return [f"Search error: {exc}"]

    titles = []
    for article in records.get("PubmedArticle", []):
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        titles.append(str(title))
    return titles[:n]

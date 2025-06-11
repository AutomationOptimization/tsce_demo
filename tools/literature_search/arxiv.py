import os, requests
from typing import List
from xml.etree import ElementTree
from tsce_agent_demo.models.research_task import PaperMeta
from .base import LiteratureSearchTool

_ARXIV = "https://export.arxiv.org/api/query"

class ArxivSearch(LiteratureSearchTool):
    def run(self, query: str, k: int = 10) -> List[PaperMeta]:
        params = {"search_query": query, "start": 0, "max_results": k}
        xml = requests.get(_ARXIV, params=params, timeout=30).text
        root = ElementTree.fromstring(xml)
        papers: List[PaperMeta] = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            papers.append(
                PaperMeta(
                    title=entry.findtext("{http://www.w3.org/2005/Atom}title").strip(),
                    url=entry.findtext("{http://www.w3.org/2005/Atom}id"),
                    authors=[
                        a.findtext("{http://www.w3.org/2005/Atom}name")
                        for a in entry.findall("{http://www.w3.org/2005/Atom}author")
                    ],
                    year=int(entry.findtext(
                        "{http://www.w3.org/2005/Atom}published")[:4]),
                    abstract=entry.findtext("{http://www.w3.org/2005/Atom}summary").strip(),
                )
            )
        return papers[:k]

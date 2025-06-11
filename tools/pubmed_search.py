"""Query PubMed for articles matching a search term."""

from __future__ import annotations

from typing import List

from tsce_agent_demo.models.research_task import Paper


class PubMedSearch:
    """Offline stub for querying PubMed."""

    def run(self, query: str, k: int = 5) -> List[Paper]:
        return [Paper(title=f"PubMed result {i} - {query}", year=2024) for i in range(k)]

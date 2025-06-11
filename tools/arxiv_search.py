"""Simple wrapper for querying the arXiv API."""

from __future__ import annotations

from typing import List

from tsce_agent_demo.models.research_task import Paper


class ArxivSearch:
    """Offline stub for querying arXiv."""

    def run(self, query: str, k: int = 5) -> List[Paper]:
        return [Paper(title=f"ArXiv result {i} - {query}", year=2024) for i in range(k)]

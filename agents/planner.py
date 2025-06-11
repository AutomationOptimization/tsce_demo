"""Simple planning agent."""
from __future__ import annotations

from typing import List

from tools import ArxivSearch, PubMedSearch
from tsce_agent_demo.models.research_task import ResearchTask

from .base_agent import BaseAgent, compose_sections


class Planner(BaseAgent):
    """Generate a numbered plan from the agent's context."""

    def act(self) -> List[str]:
        """Return a step-by-step plan derived from ``self.context``.

        The method looks for ``self.context`` (a string) and splits it into
        individual statements. Each statement becomes a numbered step.
        """
        context = getattr(self, "context", "")
        if not isinstance(context, str) or not context.strip():
            return ["Step 1: No context provided."]

        # Split on newlines first; fall back to sentences.
        parts = [p.strip() for p in context.splitlines() if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in context.split(".") if p.strip()]

        return [f"Step {i + 1}: {part}" for i, part in enumerate(parts)]

    def send_message(self, message: str) -> str:  # pragma: no cover
        self.context = message
        output = "\n".join(self.act())
        return compose_sections("", "", output)


def plan(task: ResearchTask) -> ResearchTask:
    """Phase-1: build ranked bibliography."""
    searchers = [ArxivSearch(), PubMedSearch()]
    results = []
    for s in searchers:
        results.extend(s.run(task.question, k=20))
    # dedupe by title
    seen = set()
    unique = []
    for p in results:
        if p.title.lower() not in seen:
            unique.append(p)
            seen.add(p.title.lower())
    # rank: newer first, then arXiv ID or PubMed PMID
    task.literature = sorted(unique, key=lambda p: (-p.year, p.title))
    return task

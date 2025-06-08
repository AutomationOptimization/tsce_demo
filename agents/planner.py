"""Simple planning agent."""
from __future__ import annotations

from typing import List

from .base import BaseAgent  # type: ignore


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

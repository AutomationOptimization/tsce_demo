from __future__ import annotations

from typing import List
from time import sleep

from .base_agent import BaseAgent


class Judge(BaseAgent):
    """Simple agent that approves or rejects simulation results."""

    def approve(self, transcript: str) -> bool:
        """Return ``True`` if the judge approves ``transcript``."""
        prompt = (
            "You are a quality assurance judge. "
            "Respond with YES to approve the following output or NO to reject.\n\n"
            f"{transcript}"
        )
        reply = self.send_message(prompt)
        return "yes" in reply.lower()


class JudgePanel:
    """Coordinator that requires unanimous approval from multiple judges."""

    def __init__(self, count: int = 9) -> None:
        self.judges: List[Judge] = [Judge(name=f"Judge{i+1}") for i in range(count)]

    def vote(self, transcript: str) -> bool:
        """Return ``True`` when all judges approve ``transcript``."""
        results = [j.approve(transcript) for j in self.judges]
        return all(results)

    def vote_until_unanimous(self, transcript: str, *, delay: float = 0.5) -> bool:
        """Repeatedly poll judges until all approve ``transcript``.

        The optional ``delay`` parameter waits the specified number of seconds
        between successive rounds of voting to prevent a tight loop.
        """
        while True:
            if self.vote(transcript):
                return True
            sleep(delay)


__all__ = ["Judge", "JudgePanel"]

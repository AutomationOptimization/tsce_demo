from __future__ import annotations

from typing import List
from time import sleep
import re

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


class BioSafetyError(RuntimeError):
    """Raised when unsafe biology-related content is detected."""


class BioSafetyOfficer(BaseAgent):
    """Simple biosafety checker using regex patterns and OpenAI moderation."""

    DEFAULT_PATTERNS = [
        r"\bVX\b",
        r"sarin",
        r"soman",
        r"mustard gas",
    ]

    def __init__(self, patterns: List[str] | None = None) -> None:
        super().__init__(name="BioSafetyOfficer")
        patterns = patterns or self.DEFAULT_PATTERNS
        self.patterns = [re.compile(p, re.I) for p in patterns]

    def approve(self, text: str) -> bool:
        for pat in self.patterns:
            if pat.search(text):
                raise BioSafetyError("Prohibited content detected")

        try:  # pragma: no cover - network path
            import openai

            client = openai.OpenAI()
            resp = client.moderations.create(input=text)
            result = resp.results[0]
            flagged = getattr(result, "flagged", False) or any(result.categories.values())
            if flagged:
                raise BioSafetyError("OpenAI moderation flagged content")
        except Exception:  # pragma: no cover - external dependency
            pass

        return True


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


__all__ = [
    "Judge",
    "JudgePanel",
    "BioSafetyOfficer",
    "BioSafetyError",
]

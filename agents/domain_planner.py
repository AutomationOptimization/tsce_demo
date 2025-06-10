from __future__ import annotations

from types import SimpleNamespace

from .base_agent import BaseAgent, compose_sections


class DomainAwarePlanner(BaseAgent):
    """Generate a basic research plan broken into standard sections."""

    def __init__(self, *, log_dir: str | None = None) -> None:
        class EchoChat:
            def __call__(self, message: str) -> SimpleNamespace:
                return SimpleNamespace(content=message)

        super().__init__(name="DomainAwarePlanner", chat=EchoChat(), log_dir=log_dir)

    def send_message(self, message: str) -> str:  # pragma: no cover - thin wrapper
        return compose_sections("", "", self.act(message))

    # ------------------------------------------------------------------
    def act(self, question: str) -> str:
        """Return a multi-section plan for ``question``."""
        return (
            f"Literature Search:\n"
            f"- Identify prior work related to {question}.\n\n"
            f"Method Design:\n"
            f"- Outline an approach to investigate {question}.\n\n"
            f"Expected Results:\n"
            f"- Predict likely findings.\n\n"
            f"Analysis Plan:\n"
            f"- Describe how the results will be analyzed."
        )


__all__ = ["DomainAwarePlanner"]

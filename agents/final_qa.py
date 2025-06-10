from __future__ import annotations

from .base_agent import BaseAgent
from tsce_agent_demo.tsce_chat import TSCEChat


class FinalQA(BaseAgent):
    """Lightweight QA agent that validates a final summary."""

    def __init__(self, *, log_dir: str | None = None, chat: TSCEChat | None = None, model: str | None = None) -> None:
        super().__init__(name="FinalQA", chat=chat, model=model, log_dir=log_dir)

    def send_message(self, message: str) -> bool:  # pragma: no cover - thin wrapper
        return self.act(message)

    # ------------------------------------------------------------------
    def act(self, text: str) -> bool:
        """Return ``True`` if ``text`` appears to satisfy the goal."""
        prompt = (
            "You are a quality assurance assistant. "
            "Respond with YES if the following summary correctly solves the task, otherwise NO.\n\n"
            f"{text}"
        )
        reply = self.chat(prompt).content.strip().lower()
        return "yes" in reply or "true" in reply


__all__ = ["FinalQA"]

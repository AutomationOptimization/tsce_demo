from __future__ import annotations

from abc import ABC
from typing import Dict, List

from tsce_agent_demo.tsce_chat import TSCEChat


class BaseAgent(ABC):
    """Minimal interface for conversational agents using :class:`TSCEChat`."""

    def __init__(self, name: str, *, chat: TSCEChat | None = None, model: str | None = None) -> None:
        self.name = name
        self.history: List[Dict[str, str]] = []
        self.chat = chat or TSCEChat(model=model)

    # ------------------------------------------------------------------
    def send_message(self, message: str) -> str:
        """Send ``message`` to the underlying :class:`TSCEChat` instance."""
        reply = self.chat(message).content
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": self.name.lower(), "content": reply})
        return reply

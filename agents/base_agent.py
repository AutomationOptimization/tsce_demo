from __future__ import annotations

from abc import ABC
import os
from typing import Dict, List

from tsce_agent_demo.tsce_chat import TSCEChat


class BaseAgent(ABC):
    """Minimal interface for conversational agents using :class:`TSCEChat`."""

    def __init__(self, name: str, *, chat: TSCEChat | None = None, model: str | None = None, log_dir: str | None = None) -> None:
        self.name = name
        self.history: List[Dict[str, str]] = []
        self.chat = chat or TSCEChat(model=model)
        self.log_file: str | None = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"{self.name.lower()}_history.log")

    # ------------------------------------------------------------------
    def send_message(self, message: str) -> str:
        """Send ``message`` to the underlying :class:`TSCEChat` instance."""
        reply = self.chat(message).content
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": self.name.lower(), "content": reply})
        self._write_log(message, reply)
        return reply

    # ------------------------------------------------------------------
    def _write_log(self, message: str, reply: str) -> None:
        """Append ``message`` and ``reply`` to ``self.log_file`` if set."""
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"USER: {message}\n")
            f.write(f"{self.name.upper()}: {reply}\n")

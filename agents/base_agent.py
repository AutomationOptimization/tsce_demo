from __future__ import annotations

from abc import ABC
import json
import os
import re
from typing import Any, Dict, List, Tuple, Callable
import functools
import logging
try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    structlog = None

from core.logging_setup import setup_logging

from tools import use_tool

from tsce_agent_demo.tsce_chat import TSCEChat


class BaseAgent(ABC):
    """Minimal interface for conversational agents using :class:`TSCEChat`."""

    def __init__(self, name: str, *, chat: TSCEChat | None = None, model: str | None = None, log_dir: str | None = None) -> None:
        setup_logging()
        self.name = name
        self.history: List[Dict[str, str]] = []
        self.chat = chat or TSCEChat(model=model)
        self.log_file: str | None = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"{self.name.lower()}_history.log")
        self._wrap_act()

    def _wrap_act(self) -> None:
        if hasattr(self, "act"):
            method = getattr(self, "act")
            if callable(method) and not getattr(method, "_wrapped", False):
                if structlog:
                    logger = structlog.get_logger(self.name)
                else:  # fallback
                    logger = logging.getLogger(self.name)

                @functools.wraps(method)
                def wrapper(*args, **kwargs):
                    logger.info("act.start")
                    result = method(*args, **kwargs)
                    logger.info("act.end", result=str(result))
                    return result

                wrapper._wrapped = True
                setattr(self, "act", wrapper)

    # ------------------------------------------------------------------
    def send_message(self, message: str) -> str:
        """Send ``message`` to the underlying :class:`TSCEChat` instance."""
        reply = self.chat(message).content
        thoughts, critical, speak = parse_sections(reply)

        result: Any = speak
        if speak.strip().startswith("{"):
            try:
                payload = json.loads(speak)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(payload, dict) and "tool" in payload:
                    result = use_tool(payload["tool"], payload.get("args", {}))
                    if isinstance(result, list):
                        result = "\n".join(result)

        formatted = compose_sections(thoughts, critical, str(result))
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": self.name.lower(), "content": formatted})
        self._write_log(message, formatted)
        return formatted

    # ------------------------------------------------------------------
    def _write_log(self, message: str, reply: str) -> None:
        """Append ``message`` and ``reply`` to ``self.log_file`` if set."""
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"USER: {message}\n")
            f.write(f"{self.name.upper()}: {reply}\n")


# ----------------------------------------------------------------------
def compose_sections(thoughts: str, critical: str, speak: str) -> str:
    """Return a unified message string with the standard sections."""
    return (
        "Thoughts:\n" + thoughts.strip() +
        "\n\nCritical Thinking:\n" + critical.strip() +
        "\n\nSpeak:\n" + speak.strip()
    )


def parse_sections(text: str) -> Tuple[str, str, str]:
    """Return the (thoughts, critical thinking, speak) tuple from ``text``."""
    pattern = (
        r"Thoughts:\s*(.*?)\n\s*Critical Thinking:\s*(.*?)\n\s*Speak:\s*(.*)"
    )
    m = re.search(pattern, text, re.S)
    if not m:
        return "", "", text.strip()
    return tuple(part.strip() for part in m.groups())

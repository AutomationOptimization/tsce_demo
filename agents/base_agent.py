from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseAgent(ABC):
    """Minimal interface for conversational agents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def send_message(self, message: str) -> str:
        """Send a message to the agent and return the response."""
        raise NotImplementedError

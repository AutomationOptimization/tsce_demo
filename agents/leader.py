from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .base_agent import BaseAgent


@dataclass
class Leader(BaseAgent):
    """Simple leader agent that issues high level goals sequentially."""

    goals: List[str] = field(default_factory=list)
    step: int = 0

    def __init__(self, goals: List[str] | None = None) -> None:
        super().__init__(name="Leader")
        self.history: List[str] = []
        self.goals = goals or []
        self.step = 0

    def observe(self, message: str) -> None:
        self.history.append(message)

    def send_message(self, message: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def act(self) -> str:
        """Return the next goal or indicate completion."""
        if self.step < len(self.goals):
            goal = self.goals[self.step]
            self.step += 1
            self.history.append(goal)
            return goal
        return "All goals completed."

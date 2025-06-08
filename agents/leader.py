from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


class BaseAgent:
    """Minimal base agent with conversation state."""

    def __init__(self, name: str = "agent") -> None:
        self.name = name
        self.history: List[str] = []

    def observe(self, message: str) -> None:
        """Record a message from another agent/user."""
        self.history.append(message)

    def act(self) -> str:  # pragma: no cover - to be implemented by subclasses
        raise NotImplementedError


@dataclass
class Leader(BaseAgent):
    """Simple leader agent that issues high level goals sequentially."""

    goals: List[str] = field(default_factory=list)
    step: int = 0

    def act(self) -> str:
        """Return the next goal or indicate completion."""
        if self.step < len(self.goals):
            goal = self.goals[self.step]
            self.step += 1
            self.history.append(goal)
            return goal
        return "All goals completed."

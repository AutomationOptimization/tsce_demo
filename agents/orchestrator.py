from __future__ import annotations

from typing import List, Dict

from .leader import Leader
from .planner import Planner
from .scientist import Scientist
from tsce_agent_demo.tsce_chat import TSCEChat


class Orchestrator:
    """Coordinate a simple round-robin conversation between agents."""

    def __init__(self, goals: List[str], *, model: str | None = None) -> None:
        self.leader = Leader(goals=goals)
        self.planner = Planner(name="Planner")
        self.scientist = Scientist(name="Scientist")
        self.chat = TSCEChat(model=model)
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, str]]:
        """Run the group chat until a terminate token is observed."""
        while True:
            goal = self.leader.act()
            self.history.append({"role": "leader", "content": goal})
            if "terminate" in goal.lower():
                break

            plan_prompt = f"You are Planner. Devise a brief plan for: {goal}"
            plan = self.chat(plan_prompt).content
            self.history.append({"role": "planner", "content": plan})
            if "terminate" in plan.lower():
                break

            sci_prompt = (
                "You are Scientist. Based on this plan, provide your analysis:\n"
                f"{plan}"
            )
            answer = self.chat(sci_prompt).content
            self.history.append({"role": "scientist", "content": answer})
            if "terminate" in answer.lower():
                break
        return self.history


__all__ = ["Orchestrator"]


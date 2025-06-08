from __future__ import annotations

from .base_agent import BaseAgent


class Scientist(BaseAgent):
    """High-level planner that coordinates research tasks."""

    def request_information(self, researcher: BaseAgent, query: str) -> str:
        """Ask the Researcher agent to gather information about *query*."""
        instruction = f"Research the following and report back: {query}"
        return researcher.send_message(instruction)

    def direct_researcher(self, researcher: BaseAgent, instructions: str) -> str:
        """Provide step-by-step guidance to the Researcher agent."""
        return researcher.send_message(instructions)

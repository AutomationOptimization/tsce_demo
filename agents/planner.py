"""Simple planning agent."""
from __future__ import annotations

from typing import List

MAX_STEPS = 5

from tsce_agent_demo.models.research_task import ResearchTask
from tsce_agent_demo.models.research_task import MethodPlan
from tsce_agent_demo.utils.vector_store import query
from openai import OpenAI

from .base_agent import BaseAgent, compose_sections


class Planner(BaseAgent):
    """Generate a numbered plan from the agent's context."""

    def act(self) -> List[str]:
        """Return a step-by-step plan derived from ``self.context``.

        The method looks for ``self.context`` (a string) and splits it into
        individual statements. Each statement becomes a numbered step.
        """
        context = getattr(self, "context", "")
        if not isinstance(context, str) or not context.strip():
            return ["Step 1: No context provided."]

        # Split on newlines first; fall back to sentences.
        parts = [p.strip() for p in context.splitlines() if p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in context.split(".") if p.strip()]

        parts = parts[:MAX_STEPS]

        return [f"Step {i + 1}: {part}" for i, part in enumerate(parts)]

    def send_message(self, message: str) -> str:  # pragma: no cover
        self.context = message
        output = "\n".join(self.act())
        return compose_sections("", "", output)


def design_method(task: ResearchTask, model: str = "gpt-4o-mini") -> ResearchTask:
    """Generate a JSON MethodPlan using LLM + retrieved evidence."""

    evidence = "\n\n".join(query(task.question, k=8))
    prompt = f"""You are a lab PI. Draft a JSON experiment plan.

Question: {task.question}

Relevant literature snippets:
{evidence}

Return ONLY valid JSON matching this schema:
{MethodPlan.schema_json()}
"""
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    plan_json = resp.choices[0].message.content
    task.method_plan = MethodPlan.model_validate_json(plan_json)
    return task


def plan(step_text: str, retriever, *, k: int = 5) -> List[str]:
    """Generate a numbered plan for ``step_text`` using retrieval first."""
    hits = retriever.search(step_text)
    context = "\n".join(hits[:k])
    if context:
        context += "\n" + step_text
    else:
        context = step_text

    p = Planner(name="Planner")
    p.context = context
    steps = p.act()
    plan.retriever_hits = len(hits)
    return steps

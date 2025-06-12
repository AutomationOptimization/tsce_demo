from __future__ import annotations

from .base_agent import BaseAgent


# Prompt injected into every Scientist call to the language model
SYSTEM_PROMPT = (
    "When the plan involves biomedical targets, you may:\n"
    "\u2022 call google_search with a \"PUBMED:\" prefix\n"
    "\u2022 inspect MeSH terms from the returned JSON\n"
    "\u2022 rank findings by in-vitro potency (IC50) and drug-likeness"
)


class Scientist(BaseAgent):
    """High-level planner that coordinates research tasks."""

    def __init__(
        self,
        name: str = "Scientist",
        *,
        chat=None,
        model: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        super().__init__(name=name, chat=chat, model=model, log_dir=log_dir)
        # preserve the underlying chat object and expose a method instead
        self._chat = self.chat
        del self.chat
        self.system_prompt = SYSTEM_PROMPT

    def chat(self, prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)

    def request_information(self, researcher: BaseAgent, query: str) -> str:
        """Ask the Researcher agent to gather information about *query*."""
        instruction = f"Research the following and report back: {query}"
        # record the instruction via the Scientist's own channel
        self.send_message(instruction)
        return researcher.send_message(instruction)

    def direct_researcher(self, researcher: BaseAgent, instructions: str) -> str:
        """Provide step-by-step guidance to the Researcher agent."""
        self.send_message(instructions)
        return researcher.send_message(instructions)

    def execute_plan(self, plan: str, researcher: BaseAgent) -> None:
        """Execute each numbered step in ``plan`` using ``researcher``.

        Plan lines starting with a number (``1.``, ``Step 1:``, etc.) are
        interpreted as individual tasks.  For each task the method attempts to
        call a matching helper on ``researcher`` such as ``search`` or
        ``scrape``.  If no direct helper is recognised, the step is forwarded to
        the researcher via :meth:`direct_researcher`.

        All actions and their results are appended to ``self.history``.
        """

        import re

        step_re = re.compile(r"^\s*(?:Step\s*)?\d+[:.)-]?\s*(.*)", re.I)
        steps = []
        for line in plan.splitlines():
            m = step_re.match(line)
            if m:
                steps.append(m.group(1).strip())

        if not steps and plan.strip():
            steps = [plan.strip()]

        for step in steps:
            lower = step.lower()
            result: str

            if lower.startswith("search"):
                query = step.partition("search")[2].strip()
                result = researcher.search(query)
            elif lower.startswith("scrape"):
                match = re.search(r"https?://\S+", step)
                url = match.group(0) if match else step.partition("scrape")[2].strip()
                result = researcher.scrape(url)
            elif lower.startswith("read"):
                path = step.partition("read")[2].strip()
                result = researcher.read_file(path)
            elif lower.startswith("write"):
                remainder = step.partition("write")[2].strip()
                if " " in remainder:
                    path, content = remainder.split(" ", 1)
                else:
                    path, content = remainder, ""
                result = researcher.write_file(path, content)
            elif lower.startswith("create"):
                path = step.partition("create")[2].strip()
                result = researcher.create_file(path)
            elif lower.startswith("delete"):
                path = step.partition("delete")[2].strip()
                result = researcher.delete_file(path)
            elif lower.startswith("run"):
                path = step.partition("run")[2].strip()
                result = researcher.run_script(path)
            else:
                result = self.direct_researcher(researcher, step)

            self.history.append({"role": "scientist", "content": step})
            self.history.append({"role": "researcher", "content": str(result)})

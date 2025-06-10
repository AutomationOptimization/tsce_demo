from __future__ import annotations

from typing import Any, List, Dict
import os
import re
import uuid
from dataclasses import dataclass, field
from collections import deque
import warnings

from .leader import Leader
from .planner import Planner
from .scientist import Scientist
from .researcher import Researcher
from .script_writer import ScriptWriter
from .script_qa import ScriptQA
from .simulator import Simulator
from .evaluator import Evaluator
from .final_qa import FinalQA
from .hypothesis import record_agreed_hypothesis
from .judge import JudgePanel
from tsce_agent_demo.tsce_chat import TSCEChat


@dataclass
class Message:
    """Simple container for agent communication."""

    sender: str
    recipients: List[str]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Coordinate a simple round-robin conversation between agents."""

    def __init__(self, goals: List[str], *, model: str | None = None, output_dir: str = "output", log_dir: str | None = None) -> None:
        self.leader = Leader(goals=goals, log_dir=log_dir)
        self.planner = Planner(name="Planner", log_dir=log_dir)
        self.scientist = Scientist(name="Scientist", log_dir=log_dir)
        self.researcher = Researcher(log_dir=log_dir)
        self.script_writer = ScriptWriter(log_dir=log_dir)
        self.script_qa = ScriptQA(log_dir=log_dir)
        self.simulator = Simulator(log_dir=log_dir)
        # Store all results in the project’s top‑level ``results`` directory
        # so that no artifacts end up under ``tsce_agent_demo``.
        self.results_dir = "results"
        self.evaluator = Evaluator(results_dir=self.results_dir, log_dir=log_dir)
        self.final_qa = FinalQA(log_dir=log_dir)
        self.judge_panel = JudgePanel()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.hypothesis_dir = os.path.join(self.output_dir, "hypothesis")
        os.makedirs(self.hypothesis_dir, exist_ok=True)
        self.chat = TSCEChat(model=model)
        self.history: List[Dict[str, str]] = []
        self.log_dir = log_dir
        self.stages = {
            "planner": True,
            "hypothesis": True,
            # research starts disabled so the planner runs first
            "research": False,
            "script": True,
            "qa": True,
            "simulate": True,
            "evaluate": True,
            "judge": True,
        }
        warnings.warn(
            "stage flags are deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    def _is_trivial(self, text: str) -> bool:
        """Return ``True`` if ``text`` indicates a trivial task."""
        return "hello world" in text.lower()

    def needs_code(self, plan: str) -> bool:
        """Return ``True`` if ``plan`` suggests writing or running code."""
        lowered = plan.lower()
        keywords = [
            "code",
            "script",
            "python",
            "execute",
            "run",
            "compute",
            "calculate",
            "fibonacci",
            "factorial",
        ]
        return any(word in lowered for word in keywords)

    def drop_stage(self, stage: str) -> None:
        """Legacy helper to disable a processing stage.

        This method remains for backward compatibility with older
        scripts that toggled pipeline stages manually.
        """
        if stage in self.stages:
            self.stages[stage] = False

    def activate_stage(self, stage: str) -> None:
        """Legacy helper to enable a processing stage.

        Provided for scripts that still toggle stages manually.
        """
        if stage in self.stages:
            self.stages[stage] = True
            if stage == "script" and self.script_writer is None:
                self.script_writer = ScriptWriter(log_dir=self.log_dir)
            elif stage == "simulate" and self.simulator is None:
                self.simulator = Simulator(log_dir=self.log_dir)
            elif stage == "evaluate" and self.evaluator is None:
                self.evaluator = Evaluator(results_dir=self.results_dir, log_dir=self.log_dir)

    def _sanitize_script(self, script: str) -> str:
        """Return ``script`` wrapped in a docstring if it fails to compile."""
        try:
            compile(script, "<string>", "exec")
        except SyntaxError:
            escaped = script.replace('"""', '\"\"\"')
            script = f'r"""\n{escaped}\n"""'
        return script

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, str]]:
        """Run the group chat following the long-form pipeline."""
        prev_plan = ""
        queue: deque[Message] = deque()

        goal = self.leader.act()
        self.history.append({"role": "leader", "content": goal})
        queue.append(Message("leader", ["planner"], goal))

        while queue:
            msg = queue.popleft()
            sender = msg.sender.lower()
            content = msg.content

            if sender == "leader":
                if not self.stages.get("research"):
                    self.activate_stage("planner")
                if "terminate" in content.lower():
                    break
                if self.stages.get("planner") and not self.stages.get("research"):
                    plan_prompt = f"You are Planner. Devise a brief plan for: {content}"
                    plan = self.chat(plan_prompt).content
                    self.history.append({"role": "planner", "content": plan})
                    sci_msg = ""
                    exchange_counter = 0
                    while exchange_counter < 3:
                        sci_prompt = (
                            f"You are Scientist. Review the plan and provide feedback: {plan}"
                        )
                        sci_msg = self.chat(sci_prompt).content
                        self.history.append({"role": "scientist", "content": sci_msg})
                        plan_prompt = (
                            "You are Planner. Update the plan considering this feedback: "
                            f"{sci_msg}"
                        )
                        plan = self.chat(plan_prompt).content
                        self.history.append({"role": "planner", "content": plan})
                        exchange_counter += 1
                    if self._is_trivial(plan) or self._is_trivial(sci_msg):
                        for stage in ("research", "script", "simulate", "evaluate"):
                            self.drop_stage(stage)
                        final = "Hello, world!"
                        self.history.append({"role": "evaluator", "content": final})
                        verdict = self.final_qa.act(final)
                        self.history.append({"role": "final_qa", "content": str(verdict)})
                        if self.stages.get("judge"):
                            self.judge_panel.vote_until_unanimous(final)
                            self.history.append({"role": "judge_panel", "content": "approved"})
                        queue.clear()
                        break
                else:
                    plan = prev_plan

                if plan.strip() == prev_plan.strip() and self.stages.get("research"):
                    data = self.researcher.search(content)
                    interject = f"INTERJECT: {data}"
                    self.history.append({"role": "researcher", "content": interject})
                    self.leader.observe(interject)
                prev_plan = plan

                if "terminate" in plan.lower():
                    break

                queue.append(Message("planner", ["scientist"], plan))

            elif sender == "planner":
                plan = content
                if self.stages.get("hypothesis"):
                    sci_hyp = self.chat(
                        f"You are Scientist. Propose a short hypothesis for: {plan}"
                    ).content
                    self.history.append({"role": "scientist", "content": sci_hyp})
                    res_hyp = self.researcher.send_message(sci_hyp)
                    self.history.append({"role": "researcher", "content": res_hyp})

                    token = record_agreed_hypothesis(
                        sci_hyp,
                        res_hyp,
                        path=os.path.join(self.hypothesis_dir, "leading_hypothesis.txt"),
                        researcher=self.researcher,
                    )
                    if token:
                        self.history.append({"role": "hypothesis", "content": token})
                        self.drop_stage("planner")
                        self.drop_stage("hypothesis")
                        self.activate_stage("research")

                queue.append(Message("scientist", ["researcher"], plan))

            elif sender == "scientist":
                plan = content
                if self.stages.get("research"):
                    instr_prompt = (
                        f"You are Scientist. Provide instructions for the Researcher to gather data: {plan}"
                    )
                    sci_instr = self.chat(instr_prompt).content
                    self.history.append({"role": "scientist", "content": sci_instr})
                    res_msg = self.researcher.send_message(sci_instr)
                    self.history.append({"role": "researcher", "content": res_msg})

                    data_parts: List[str] = []
                    urls = re.findall(r"https?://\S+", sci_instr)
                    for url in urls:
                        scrap = self.researcher.scrape(url)
                        self.history.append({"role": "researcher", "content": scrap})
                        data_parts.append(scrap)

                    scripts = re.findall(r"run\s+(\S+)", sci_instr)
                    for script_path in scripts:
                        output = self.researcher.run_script(script_path)
                        self.history.append({"role": "researcher", "content": output})
                        data_parts.append(output)

                    if not data_parts:
                        search_results = self.researcher.search(self.leader.history[-1])
                        if isinstance(search_results, list):
                            search_results = "\n".join(search_results)
                        data_parts.append(search_results)

                    data = "\n".join(data_parts)
                    plan = f"{plan}\n{data}" if data else plan
                    research_path = os.path.join(self.output_dir, "research.txt")
                    if os.path.exists(research_path):
                        prev = self.researcher.read_file(research_path)
                        new_content = prev + ("\n" if prev else "") + data
                        self.researcher.write_file(research_path, new_content)
                    else:
                        self.researcher.create_file(research_path, data)
                    self.drop_stage("research")

                queue.append(Message("researcher", ["script_writer"], plan))

            elif sender == "researcher":
                plan = content
                need_code = self.needs_code(plan)
                if not need_code:
                    self.drop_stage("script")

                if self.stages.get("script") and need_code:
                    script, gid = self.script_writer.act(plan)
                    script = self._sanitize_script(script)
                    self.history.append({"role": "script_writer", "content": script})
                    path = os.path.join(
                        self.hypothesis_dir,
                        f"test_hypothesis_{uuid.uuid4().hex}.py",
                    )
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(script)

                    if self.stages.get("qa"):
                        success, qa_output = self.script_qa.act(path)
                        self.history.append(
                            {"role": "script_qa", "content": qa_output, "success": success}
                        )

                    if self.stages.get("simulate"):
                        log_path = self.simulator.act(path)
                        self.history.append({"role": "simulator", "content": log_path})
                        sim_result = self.evaluator.parse_simulator_log(
                            log_path, dest_dir=self.results_dir
                        )
                        self.history.append({"role": "evaluator", "content": sim_result["summary_file"]})

                queue.append(Message("script_writer", ["evaluator"], ""))

            elif sender == "script_writer":
                if self.stages.get("evaluate"):
                    result = self.evaluator.act()
                    self.history.append({"role": "evaluator", "content": result["summary"]})
                    verdict = self.final_qa.act(result["summary"])
                    self.history.append({"role": "final_qa", "content": str(verdict)})
                    if self.stages.get("judge"):
                        self.judge_panel.vote_until_unanimous(result["summary"])
                        self.history.append({"role": "judge_panel", "content": "approved"})
                    if result.get("success"):
                        queue.clear()
                        break
                    self.leader.step -= 1
                    prev_plan = ""
                    next_goal = self.leader.act()
                    self.history.append({"role": "leader", "content": next_goal})
                    queue.append(Message("leader", ["planner"], next_goal))

            if not queue and self.leader.step < len(self.leader.goals):
                next_goal = self.leader.act()
                self.history.append({"role": "leader", "content": next_goal})
                queue.append(Message("leader", ["planner"], next_goal))

        return self.history

    # ------------------------------------------------------------------
    def run_legacy(self) -> List[Dict[str, str]]:
        """Execute the original round-robin flow using the message queue."""
        warnings.warn(
            "run_legacy() is deprecated; use run() for the full pipeline",
            DeprecationWarning,
            stacklevel=2,
        )

        queue: deque[Message] = deque()
        goal = self.leader.act()
        self.history.append({"role": "leader", "content": goal})
        queue.append(Message("leader", ["planner"], goal))

        while queue:
            msg = queue.popleft()
            sender = msg.sender.lower()
            content = msg.content

            if sender == "leader":
                if "terminate" in content.lower():
                    break
                plan_prompt = f"You are Planner. Devise a brief plan for: {content}"
                plan = self.chat(plan_prompt).content
                self.history.append({"role": "planner", "content": plan})
                if "terminate" in plan.lower():
                    break
                queue.append(Message("planner", ["scientist"], plan))

            elif sender == "planner":
                sci_prompt = (
                    "You are Scientist. Based on this plan, provide your analysis:\n"
                    f"{content}"
                )
                answer = self.chat(sci_prompt).content
                self.history.append({"role": "scientist", "content": answer})
                if "terminate" in answer.lower():
                    break

                if self.leader.step < len(self.leader.goals):
                    next_goal = self.leader.act()
                    self.history.append({"role": "leader", "content": next_goal})
                    queue.append(Message("leader", ["planner"], next_goal))

        return self.history


__all__ = ["Orchestrator", "Message"]


from __future__ import annotations

from typing import List, Dict
import os

from .leader import Leader
from .planner import Planner
from .scientist import Scientist
from .researcher import Researcher
from .script_writer import ScriptWriter
from .script_qa import ScriptQA
from .simulator import Simulator
from .evaluator import Evaluator
from .hypothesis import record_agreed_hypothesis
from .judge import JudgePanel
from tsce_agent_demo.tsce_chat import TSCEChat


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
        self.evaluator = Evaluator(results_dir="tsce_agent_demo/results", log_dir=log_dir)
        self.judge_panel = JudgePanel()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.chat = TSCEChat(model=model)
        self.history: List[Dict[str, str]] = []
        self.stages = {
            "hypothesis": True,
            "research": True,
            "script": True,
            "qa": True,
            "simulate": True,
            "evaluate": True,
            "judge": True,
        }

    def drop_stage(self, stage: str) -> None:
        """Disable a processing stage."""
        if stage in self.stages:
            self.stages[stage] = False

    def activate_stage(self, stage: str) -> None:
        """Enable a processing stage."""
        if stage in self.stages:
            self.stages[stage] = True

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, str]]:
        """Run the group chat following the long-form pipeline."""
        prev_plan = ""

        while True:
            goal = self.leader.act()
            self.history.append({"role": "leader", "content": goal})
            if "terminate" in goal.lower():
                break

            plan_prompt = f"You are Planner. Devise a brief plan for: {goal}"
            plan = self.chat(plan_prompt).content
            self.history.append({"role": "planner", "content": plan})

            if plan.strip() == prev_plan.strip():
                data = self.researcher.search(goal)
                interject = f"INTERJECT: {data}"
                self.history.append({"role": "researcher", "content": interject})
                self.leader.observe(interject)
            prev_plan = plan

            if "terminate" in plan.lower():
                break

            # --- Hypothesis -------------------------------------------------
            if self.stages.get("hypothesis"):
                sci_hyp = self.chat(
                    f"You are Scientist. Propose a short hypothesis for: {plan}"
                ).content
                self.history.append({"role": "scientist", "content": sci_hyp})
                res_hyp = self.researcher.send_message(sci_hyp)
                self.history.append({"role": "researcher", "content": res_hyp})

            # --- Research aggregation --------------------------------------
            if self.stages.get("research"):
                data = self.researcher.search(goal)
                self.history.append({"role": "researcher", "content": data})
                plan = f"{plan}\n{data}"
                self.researcher.create_file(
                    os.path.join(self.output_dir, "research.txt"),
                    data,
                )
                token = record_agreed_hypothesis(
                    sci_hyp,
                    res_hyp,
                    path=os.path.join(self.output_dir, "leading_hypothesis.txt"),
                    researcher=self.researcher,
                )
                if token:
                    self.history.append({"role": "hypothesis", "content": token})
                    break

            # --- Script writing -------------------------------------------
            if self.stages.get("script"):
                script = self.script_writer.act(plan)
                self.history.append({"role": "script_writer", "content": script})
                path = "generated_script.py"
                self.researcher.create_file(path, script)

                if self.stages.get("qa"):
                    success, qa_output = self.script_qa.act(path)
                    self.history.append(
                        {"role": "script_qa", "content": qa_output}
                    )

                if self.stages.get("simulate"):
                    log_path = self.simulator.act(path)
                    self.history.append({"role": "simulator", "content": log_path})
                    sim_result = self.evaluator.parse_simulator_log(log_path)
                    self.history.append({"role": "evaluator", "content": sim_result["summary"]})

            # --- Evaluation -------------------------------------------------
            if self.stages.get("evaluate"):
                result = self.evaluator.act()
                self.history.append({"role": "evaluator", "content": result["summary"]})
                approved = True
                if self.stages.get("judge"):
                    approved = self.judge_panel.vote(result["summary"])
                    status = "approved" if approved else "rejected"
                    self.history.append({"role": "judge_panel", "content": status})
                if result.get("success") and approved:
                    break

        return self.history


__all__ = ["Orchestrator"]


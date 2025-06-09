from __future__ import annotations

from typing import List, Dict
import os
import re
import uuid

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
        self.results_dir = "tsce_agent_demo/results"
        self.evaluator = Evaluator(results_dir=self.results_dir, log_dir=log_dir)
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
            if stage == "research":
                # Remove heavy agents until later phases
                self.script_writer = None
                self.simulator = None
                self.evaluator = None
                self.stages["script"] = False
                self.stages["simulate"] = False
                self.stages["evaluate"] = False

    def activate_stage(self, stage: str) -> None:
        """Enable a processing stage."""
        if stage in self.stages:
            self.stages[stage] = True
            if stage == "script" and self.script_writer is None:
                self.script_writer = ScriptWriter(log_dir=self.log_dir)
            elif stage == "simulate" and self.simulator is None:
                self.simulator = Simulator(log_dir=self.log_dir)
            elif stage == "evaluate" and self.evaluator is None:
                self.evaluator = Evaluator(results_dir=self.results_dir, log_dir=self.log_dir)

    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, str]]:
        """Run the group chat following the long-form pipeline."""
        prev_plan = ""

        while True:
            goal = self.leader.act()
            # Reactivate planner for a new goal only when research is inactive
            if not self.stages.get("research"):
                self.activate_stage("planner")
            self.history.append({"role": "leader", "content": goal})
            if "terminate" in goal.lower():
                break

            if self.stages.get("planner") and not self.stages.get("research"):
                plan_prompt = f"You are Planner. Devise a brief plan for: {goal}"
                plan = self.chat(plan_prompt).content
                self.history.append({"role": "planner", "content": plan})

                # Allow Planner and Scientist to refine the plan through
                # a short back-and-forth before any research begins.
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
            else:
                plan = prev_plan

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
                    data_parts.append(self.researcher.search(goal))

                data = "\n".join(data_parts)
                plan = f"{plan}\n{data}" if data else plan
                research_path = os.path.join(self.output_dir, "research.txt")
                if os.path.exists(research_path):
                    prev = self.researcher.read_file(research_path)
                    new_content = prev + ("\n" if prev else "") + data
                    self.researcher.write_file(research_path, new_content)
                else:
                    self.researcher.create_file(research_path, data)
                token = record_agreed_hypothesis(
                    sci_hyp,
                    res_hyp,
                    path=os.path.join(self.hypothesis_dir, "leading_hypothesis.txt"),
                    researcher=self.researcher,
                )
                if token:
                    self.history.append({"role": "hypothesis", "content": token})
                    # Planner is no longer needed once research begins
                    self.drop_stage("planner")
                    # Mark hypothesis stage complete and proceed
                    self.drop_stage("hypothesis")

            # --- Script writing -------------------------------------------
            if self.stages.get("script"):
                script, gid = self.script_writer.act(plan)
                self.history.append({"role": "script_writer", "content": script})
                path = os.path.join(
                    self.hypothesis_dir,
                    f"test_hypothesis_{uuid.uuid4().hex}.py",
                )
                self.researcher.create_file(path, script)

                if self.stages.get("qa"):
                    success, qa_output = self.script_qa.act(path)
                    self.history.append(
                        {
                            "role": "script_qa",
                            "content": qa_output,
                            "success": success,
                        }
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
                # retry same goal on failure or rejection
                self.leader.step -= 1
                prev_plan = ""
                continue

        return self.history


__all__ = ["Orchestrator"]


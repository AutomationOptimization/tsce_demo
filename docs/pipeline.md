# Multi-Stage Agent Pipeline

This document explains the long form workflow implemented by the `Orchestrator`.
Each stage is a distinct agent that feeds its output to the next.

1. **Leader** – pulls the next goal from the list.
2. **Planner** – drafts a step-by-step plan that the **Leader** assigns to the **Scientist**.
3. **Scientist** – executes the assigned plan, critiquing and iteratively refining it with the Planner.
4. **Researcher** – performs web or file searches to gather data for the plan.
5. **Hypothesis** – Scientist and Researcher must agree on a written hypothesis.
   When they do, a `TERMINATE` token is logged, the planner stage is disabled and
   the research stage is activated.
6. **ScriptWriter** – generates an executable Python script. Files are stored
   under `output/run_<id>/hypothesis/` where `<id>` is a unique identifier for
   each run.
7. **ScriptQA** – optional lint / unit test pass over the generated script.
8. **Simulator** – runs the script and saves a log file. The **Evaluator** parses
   this log and creates a summary report.
9. **JudgePanel** – nine independent Judge agents review the evaluator’s
   summary. The orchestrator calls `vote_until_unanimous()` so execution waits
   until every judge approves. Only a failed evaluation triggers a retry.

## Message Routing

The orchestrator maintains a queue of ``Message`` objects. Each message records
the sender, intended recipients and text. When ``run()`` starts the Leader's goal
is queued. Agents process messages addressed to them and may append new messages
for subsequent stages. This queue-driven approach ensures that every agent
receives the latest context in order.

If the hypothesis stage emits ``TERMINATE`` or the orchestrator detects a trivial
task such as printing ``hello world``, planning is skipped and later stages run
immediately. This short‑circuit behaviour lets simple prompts finish without any
manual stage management.

## Demo Simulators

| ID  | Parameters | Defaults | Output files |
|-----|------------|----------|--------------|
| ode | func, y0, t-span, precision, seed | precision=float64, seed=None | `ode_results.json`, `ode_results.meta.json` |
| chem | reactants, seed | seed=None | `reaction_product.json`, `reaction_product.meta.json` |

SciPy's ``odeint`` is used for its stability on stiff systems while keeping the
implementation short. RDKit's ``CombineMols`` merely concatenates molecules as a
stand-in for real reaction prediction.

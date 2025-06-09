# Multi-Stage Agent Pipeline

This document explains the long form workflow implemented by the `Orchestrator`. Each stage is a distinct agent that feeds its output to the next.

1. **Leader** – pulls the next goal from the list.
2. **Planner** – produces a step-by-step plan that the **Scientist** can inspect.
3. **Scientist** – critiques and iteratively refines the plan with the Planner.
4. **Researcher** – performs web or file searches to gather data for the plan.
5. **Hypothesis** – Scientist and Researcher must agree on a written hypothesis. When they do, a `TERMINATE` token is written to disk and the planner stage is disabled.
6. **ScriptWriter** – generates an executable Python script. Files are stored under `output/hypothesis/`.
7. **ScriptQA** – optional lint / unit test pass over the generated script.
8. **Simulator** – runs the script and saves a log file. The **Evaluator** parses this log and creates a summary report.
9. **JudgePanel** – nine independent Judge agents review the evaluator’s summary. The orchestrator calls `vote_until_unanimous()` so execution waits until every judge approves. Only a failed evaluation triggers a retry.

The pipeline can drop or reactivate stages via `drop_stage()` or `activate_stage()` on the orchestrator instance.

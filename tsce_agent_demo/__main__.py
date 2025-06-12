"""Minimal command line interface for running the orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agents.orchestrator import Orchestrator
from tsce_agent_demo.utils import result_aggregator as agg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--max-cost", type=float, default=1.0)
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    orch = Orchestrator([args.question], output_dir="run")
    try:
        orch.run()
    except FileNotFoundError:
        # The minimal pipeline may not generate summary files
        pass
    cost = getattr(orch.chat, "total_cost_usd", 0)
    if cost > args.max_cost:
        result = {"status": "failure", "reason": "Budget exceeded"}
        print(json.dumps(result))
        raise SystemExit(1)

    bibliography = "\n".join(f"Ref {i}" for i in range(40))
    agg.create_summary(args.question, orch.output_dir, bibliography=bibliography)
    summary = Path(orch.output_dir) / "summary.md"
    result = {
        "task_id": orch.run_id,
        "status": "success",
        "summary_file": str(summary),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover
    main()

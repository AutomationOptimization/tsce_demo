#!/usr/bin/env python3
"""Simple CLI to run the multi-agent Orchestrator.

Provide one or more goal strings and optionally an output directory. The
script invokes :class:`agents.orchestrator.Orchestrator` and stores the
conversation history as JSON in the chosen directory.
"""
from __future__ import annotations
import argparse, json, os
from agents.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Orchestrator pipeline")
    p.add_argument("goals", nargs="+", help="Goals for the agents")
    p.add_argument(
        "--output", "-o", default="orchestrator_output", help="Output directory"
    )
    p.add_argument("--model", default=None, help="Model name for TSCEChat")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    orch = Orchestrator(args.goals, model=args.model, output_dir=args.output)
    history = orch.run()
    out_path = os.path.join(args.output, "history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Run complete. History saved to {out_path}")


if __name__ == "__main__":
    main()

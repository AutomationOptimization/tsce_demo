#!/usr/bin/env python3
"""Simple CLI to run the multi-agent Orchestrator.

Provide one or more goal strings and optionally an output directory. The
script invokes :class:`agents.orchestrator.Orchestrator` and stores the
conversation history as JSON in the chosen directory.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from agents.orchestrator import Orchestrator
from tsce_agent_demo.tsce_chat import TSCEReply
import pandas as pd
from tools import embed_text


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

    def _serialise(item):
        if isinstance(item, TSCEReply):
            return item.__dict__
        if isinstance(item, list):
            return [_serialise(x) for x in item]
        if isinstance(item, dict):
            return {k: _serialise(v) for k, v in item.items()}
        return item

    out_path = os.path.join(orch.output_dir, "history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serialise(history), f, indent=2)

    # ---- Update memory store -------------------------------------------
    mem_path = Path("logs/memory.parquet")
    mem_path.parent.mkdir(exist_ok=True)
    rows = []
    for msg in history:
        text = msg.get("content", "")
        if not text:
            continue
        rows.append({"embedding": embed_text(text), "text": text})

    if rows:
        df_new = pd.DataFrame(rows)
        if mem_path.exists():
            df = pd.read_parquet(mem_path)
            df_new = pd.concat([df, df_new], ignore_index=True)
        df_new.to_parquet(mem_path, index=False)

    print(f"Run complete. History saved to {out_path}")


if __name__ == "__main__":
    main()

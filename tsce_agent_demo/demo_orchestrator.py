#!/usr/bin/env python3
"""Minimal demo for the multi-agent Orchestrator.

This script runs the full pipeline on a trivial goal list and prints the
conversation history.  It serves as a quick sanity check that the
Orchestrator and agents are wired correctly.
"""
from __future__ import annotations
import json

from agents import Orchestrator


def main() -> None:
    # Sample goals for the agents. The TERMINATE token ends the run.
    goals = ["Print hello world", "TERMINATE"]

    # Create the orchestrator using default model settings.
    orch = Orchestrator(goals)
    history = orch.run()

    # Pretty-print the conversation history
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

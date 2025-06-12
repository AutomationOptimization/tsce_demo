"""Launch a reproducible TSCE scientific-discovery run.

Usage::
    python main.py "<question>"

See the README section on "CLI Usage" for a quick start and
`docs/user_guide.md` for a detailed explanation of every flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from rich.progress import Progress
from rich.logging import RichHandler
from pydantic import ValidationError
from pkg_resources import get_distribution

from tsce_agent_demo.models import ResearchTask
from agents import planner, script_writer, aggregator
from tools.ingest import pdf_ingest
import tsce_agent_demo.sandbox_run as sandbox_run
from tsce_agent_demo.utils import cost_tracker


__version__ = get_distribution("tsce-agent-demo").version

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BUDGET_USD = 1.0
SANDBOX_IMAGE = "ghcr.io/tsce/tsce_sandbox:latest"
VECTOR_STORE_DIR = "vector_store"

PHASES = ("plan1", "ingest", "plan2", "write_code", "sandbox", "aggregate")


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    default_output = Path("orchestrator_output") / ts / f"run_{uuid.uuid4().hex}"
    p = argparse.ArgumentParser(description="Run the TSCE demo pipeline")
    p.add_argument("question", help="Research question for the agents")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-cost", type=float, default=DEFAULT_BUDGET_USD)
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--sandbox", dest="sandbox", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug-phase", choices=PHASES, help=argparse.SUPPRESS)
    return p


def configure_logging(level: int, output_dir: Path) -> None:
    """Configure logging to the console and a log file."""
    fmt = "%Y-%m-%d %H:%M:%S"
    handlers = [RichHandler(rich_tracebacks=True, markup=True)]
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", fmt))
    logging.getLogger().addHandler(file_handler)


_abort = False


def _handle_signal(signum: int, frame) -> None:  # pragma: no cover
    global _abort
    _abort = True


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _handle_signal)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the TSCE demo orchestrator."""
    args = build_arg_parser().parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    if args.sandbox and not Path("/var/run/docker.sock").exists():
        print("Docker socket missing", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO, output_dir)

    task = ResearchTask(question=args.question, id=str(uuid.uuid4()))
    status = "success"
    summary_file: str | None = None
    exit_code = 0
    start = time.time()

    timeline = output_dir / "timeline.jsonl"
    with Progress() as progress:
        task_progress = progress.add_task("pipeline", total=len(PHASES))
        for phase in PHASES:
            if _abort:
                status = "aborted"
                exit_code = 130
                break
            try:
                if phase == "plan1":
                    planner.plan(task)
                elif phase == "ingest":
                    pdf_ingest.ingest_papers(task.literature, index_dir=VECTOR_STORE_DIR)
                elif phase == "plan2":
                    planner.design_method(task)
                elif phase == "write_code":
                    run_py = script_writer.write_code(task, output_dir)
                elif phase == "sandbox":
                    sandbox_run.run(str(run_py))
                elif phase == "aggregate":
                    summary_file = str(aggregator.summarise(task, output_dir))

                if args.debug_phase == phase:
                    break

                total_tokens, total_cost = cost_tracker.totals()
                with open(timeline, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"phase": phase, "tokens": total_tokens, "cost": total_cost}) + "\n")
                if total_cost > args.max_cost:
                    logging.warning("Budget exceeded; aborting")
                    status = "budget_exceeded"
                    exit_code = 1
                    break
            except Exception as exc:  # pragma: no cover
                logging.exception("Phase %s failed", phase)
                failure = {"phase": phase, "error": str(exc)}
                (output_dir / "failure.json").write_text(json.dumps(failure))
                status = "failure"
                exit_code = 1
                break
            finally:
                progress.advance(task_progress)

    elapsed = time.time() - start
    try:
        total_tokens, total_cost = cost_tracker.totals()
    except Exception:
        total_tokens, total_cost = (0, 0.0)
    result = {
        "task_id": task.id,
        "status": status,
        "summary_file": summary_file,
        "total_cost": total_cost,
        "elapsed_s": round(elapsed, 2),
    }
    out_json = json.dumps(result)
    if args.json_out:
        Path(args.json_out).write_text(out_json)
    else:
        print(out_json)

    if _abort and exit_code == 0:
        exit_code = 130
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

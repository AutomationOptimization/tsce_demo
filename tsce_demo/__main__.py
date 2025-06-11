#!/usr/bin/env python3
import argparse, json, os, sys, tempfile, time
from agents.orchestrator import Orchestrator
from tsce_demo.utils import result_aggregator as agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TSCE demo orchestrator")
    p.add_argument("--question", required=True, help="Research question")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name")
    p.add_argument("--max-cost", type=float, default=1.0, help="Budget limit USD")
    sb = p.add_mutually_exclusive_group()
    sb.add_argument("--sandbox", dest="sandbox", action="store_true", default=True)
    sb.add_argument("--no-sandbox", dest="sandbox", action="store_false")
    p.add_argument("--json-out", default=None, help="Write result JSON to file")
    return p.parse_args()


def progress(i: int, total: int, label: str, start: float) -> None:
    bar_len = 6
    filled = int(bar_len * i / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    msg = f"[{bar}] Phase-{i}/{total} {label} ({time.time() - start:.1f} s)"
    if sys.stdout.isatty():
        print("\r" + msg, end="", flush=True)
    else:
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    phases = ["plan1", "ingest", "plan2", "write_code", "sandbox", "summarise"]
    tlog = os.path.join("logs", "orchestrator.timeline.jsonl")
    os.makedirs("logs", exist_ok=True)
    start_run = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TMP_DIR"] = tmp
        art_dir = os.path.join(tmp, agg.ART_DIR)
        os.makedirs(art_dir, exist_ok=True)
        orch = Orchestrator([args.question, "terminate"], model=args.model, output_dir=tmp)
        fail_log = os.path.join("logs", f"orchestrator_failure_{orch.run_id}.json")
        for idx, name in enumerate(phases, 1):
            phase_start = time.time()
            try:
                if name == "write_code":
                    orch.run()
                elif name == "summarise":
                    summary = agg.create_summary(args.question, tmp, bibliography="")
                # other phases are placeholders
            except Exception as exc:
                fail = {
                    "stage": name,
                    "error_type": type(exc).__name__,
                    "trace_snippet": "".join(json.dumps(str(exc))[:100]),
                    "elapsed_sec": round(time.time() - phase_start, 2),
                }
                with open(os.path.join(tmp, "failure.json"), "w", encoding="utf-8") as f:
                    json.dump(fail, f)
                with open(fail_log, "w", encoding="utf-8") as f:
                    json.dump(fail, f)
                print(
                    json.dumps(
                        {
                            "task_id": orch.run_id,
                            "status": "failure",
                            "reason": fail["error_type"],
                            "failure_log": fail_log,
                        }
                    )
                )
                if args.json_out:
                    with open(args.json_out, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "task_id": orch.run_id,
                                "status": "failure",
                                "reason": fail["error_type"],
                                "failure_log": fail_log,
                            },
                            f,
                        )
                sys.exit(10 + idx)
            phase_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(tlog, "a", encoding="utf-8") as f:
                f.write(json.dumps({"phase": name, "t_start": phase_end, "t_end": phase_end, "tokens": orch.chat.total_tokens, "cost_usd": orch.chat.total_cost_usd}) + "\n")
            progress(idx, len(phases), name, start_run)
        print()
        tokens, cost = orch.chat.totals()
        if cost > args.max_cost:
            msg = {"task_id": orch.run_id, "status": "failure", "reason": "Budget exceeded"}
            print(json.dumps(msg))
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(msg, f)
            sys.exit(1)
        result = {"task_id": orch.run_id, "status": "success", "summary_file": str(summary)}
        print(json.dumps(result))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f)


if __name__ == "__main__":
    main()


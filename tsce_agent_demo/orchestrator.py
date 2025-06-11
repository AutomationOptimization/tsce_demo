# Orchestrator CLI pipeline
from __future__ import annotations
import argparse, json, os, sys, tempfile, time, traceback, uuid, subprocess
from datetime import datetime, timezone
from pathlib import Path

from tsce_agent_demo.tsce_chat import TSCEChat
from tsce_demo.utils import result_aggregator as agg

PRICE_PER_1K = 0.005  # rough default for gpt-4o-mini


def _cost(tokens: int) -> float:
    return round(tokens / 1000 * PRICE_PER_1K, 4)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TSCE orchestrator pipeline")
    p.add_argument("--question", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-cost", type=float, default=1.0)
    sb = p.add_mutually_exclusive_group()
    sb.add_argument("--sandbox", dest="sandbox", action="store_true", default=True)
    sb.add_argument("--no-sandbox", dest="sandbox", action="store_false")
    p.add_argument("--json-out")
    return p.parse_args()


def progress(idx: int, total: int, name: str, elapsed: float) -> None:
    width = 6
    fill = int(width * idx / total)
    bar = "█" * fill + "░" * (width - fill)
    msg = f"[{bar}] Phase-{idx}/{total} {name} ({elapsed:.1f} s)"
    if sys.stdout.isatty():
        end = "\n" if idx == total else "\r"
        print(msg, end=end, flush=True)
    else:
        print(msg, flush=True)


def write_failure(tmp: Path, stage: str, exc: Exception, elapsed: float) -> None:
    info = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "trace_snippet": "".join(traceback.format_exception(exc))[:400],
        "elapsed_sec": round(elapsed, 2),
    }
    (tmp / "failure.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_id = uuid.uuid4().hex
    chat = TSCEChat(model=args.model)

    def plan1():
        r = chat(f"Plan the following research question in 3 steps:\n{args.question}")
        Path(os.environ["TMP_DIR"] + "/artifacts/plan1.txt").write_text(r.content)
        return chat.last_stats().get("total_tokens", 0), r.content

    def ingest(plan):
        r = chat(f"List two references for: {args.question}")
        Path(os.environ["TMP_DIR"] + "/artifacts/ingest.txt").write_text(r.content)
        return chat.last_stats().get("total_tokens", 0), r.content

    def plan2(data):
        r = chat(f"Refine the plan using this info:\n{data}")
        Path(os.environ["TMP_DIR"] + "/artifacts/plan2.txt").write_text(r.content)
        return chat.last_stats().get("total_tokens", 0), r.content

    def write_code(plan):
        r = chat(f"Write Python code to {plan}")
        code = r.content
        script = Path(os.environ["TMP_DIR"] + "/artifacts/analysis.py")
        script.write_text(code)
        return chat.last_stats().get("total_tokens", 0), script

    def sandbox_run(path: Path):
        proc = [sys.executable, str(path)]
        out = Path(os.environ["TMP_DIR"] + "/artifacts/run.log")
        with open(out, "w", encoding="utf-8") as f:
            subprocess.run(proc, stdout=f, stderr=subprocess.STDOUT, check=False)
        return 0, out

    def summarise(_: Path) -> tuple[int, Path]:
        summary = agg.create_summary(args.question, os.environ["TMP_DIR"], bibliography="")
        return 0, summary

    phases = [
        ("plan1", plan1),
        ("ingest", ingest),
        ("plan2", plan2),
        ("write_code", write_code),
        ("sandbox", sandbox_run),
        ("summarise", summarise),
    ]

    timeline = Path("logs/orchestrator.timeline.jsonl")
    timeline.parent.mkdir(exist_ok=True)
    cost_total = 0.0
    result = {"task_id": task_id}

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TMP_DIR"] = tmp
        Path(tmp, "artifacts").mkdir()
        start_all = time.time()
        data = None
        for idx, (name, func) in enumerate(phases, 1):
            t0 = time.time()
            try:
                tokens, data = func(data)
            except Exception as exc:
                elapsed = time.time() - t0
                write_failure(Path(tmp), name, exc, elapsed)
                result.update({"status": "failure", "reason": name})
                print(json.dumps(result))
                if args.json_out:
                    Path(args.json_out).write_text(json.dumps(result))
                sys.exit(10 + idx)
            elapsed = time.time() - t0
            cost = _cost(tokens)
            cost_total += cost
            with open(timeline, "a", encoding="utf-8") as f:
                json.dump({"phase": name, "t_start": _iso(t0), "t_end": _iso(time.time()), "tokens": tokens, "cost_usd": round(cost, 4)}, f)
                f.write("\n")
            progress(idx, len(phases), f"{name}.py", elapsed)
            if cost_total > args.max_cost:
                print("Budget exceeded", flush=True)
                write_failure(Path(tmp), name, RuntimeError("budget"), elapsed)
                result.update({"status": "failure", "reason": "budget"})
                print(json.dumps(result))
                if args.json_out:
                    Path(args.json_out).write_text(json.dumps(result))
                sys.exit(10 + idx)
        summary_path = data
        result.update({"status": "success", "summary_file": str(summary_path)})
        print(json.dumps(result))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()

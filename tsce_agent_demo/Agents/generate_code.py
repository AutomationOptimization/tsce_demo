#!/usr/bin/env python3
"""
generate_code.py
----------------
Create Python code with the **TSCE two‑pass wrapper** (see *tsce_chat.py*)
and save the model’s reply to a file.  A structured JSON trace can be
generated with `--json`.

Examples
~~~~~~~~
    # Plain run – save code only
    python generate_code.py "Write a bubble‑sort" -o bubblesort.py

    # Few‑shot + verbose console + JSON trace
    python generate_code.py "Write a quick‑sort" -s fs1.txt -s fs2.txt \
        -vv --json run.json -o quicksort.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
from typing import Any, List, TypedDict

from openai import RateLimitError  # re‑used for error trapping
from tsce_chat import TSCEChat, TSCEReply

################################################################################
# Utility helpers                                                               #
################################################################################

def utc_iso() -> str:
    """Return an ISO‑8601 UTC timestamp (timezone‑aware)."""
    return dt.datetime.now(dt.timezone.utc).isoformat("T", timespec="seconds")


def extract_code(text: str) -> str:
    """Return code within ``` blocks, else the raw text."""
    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.S)
    return "\n\n".join(b.strip() for b in blocks) if blocks else text

################################################################################
# JSON log schemas                                                              #
################################################################################
class Event(TypedDict, total=False):
    time: str
    level: str
    message: str
    data: Any

class ProcessLog(TypedDict, total=False):
    prompt: str
    few_shots: List[str]
    start_time: str
    end_time: str
    status: str
    error: str
    output: str
    latency_s: float
    events: List[Event]

################################################################################
# Few‑shot loader                                                               #
################################################################################

def load_few_shots(paths: List[str]) -> List[dict]:
    messages: List[dict] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            messages.append({"role": "system", "content": "Assistant:\n" + content})
            logging.debug("Loaded few‑shot exemplar from %s", p)
        except FileNotFoundError:
            logging.error("Few‑shot file not found: %s", p)
            sys.exit(f"✖  Few‑shot file not found: {p}")
    return messages

################################################################################
# Logging setup                                                                 #
################################################################################

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

################################################################################
# JSON writer                                                                   #
################################################################################

def write_json(path: str | None, data: ProcessLog) -> None:
    if not path:
        return
    data["end_time"] = utc_iso()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logging.info("Wrote JSON log to %s", path)
    except OSError as exc:
        logging.error("Unable to write JSON log: %s", exc)

################################################################################
# Main                                                                          #
################################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Python via TSCEChat and save the result.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("prompt", help="Prompt sent to the model (or task description)")
    ap.add_argument("-o", "--output", default="generated.py", help="Destination .py file")
    ap.add_argument("-d", "--deployment", help="Override Azure/OpenAI deployment name (passed to TSCEChat)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Upper bound for the model reply length")
    ap.add_argument("-s", "--few-shot", nargs="*", default=[], help="Path(s) to few‑shot exemplar text files")
    ap.add_argument("--json", dest="json_log", help="Write a full JSON trace to this file")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase console verbosity (-v/-vv)")
    args = ap.parse_args()

    setup_logging(args.verbose)

    # ----------------------------------------------------------------- JSON skeleton
    proc: ProcessLog = {
        "prompt": args.prompt,
        "few_shots": args.few_shot,
        "start_time": utc_iso(),
        "events": [],
    }

    def record(level: str, msg: str, **data):
        evt: Event = {"time": utc_iso(), "level": level.upper(), "message": msg}
        if data:
            evt["data"] = data
        proc["events"].append(evt)
        getattr(logging, level.lower())(msg)

    # ----------------------------------------------------------------- Build chat messages
    messages: List[dict] = [
        {"role": "system", "content": "You are a senior software engineer. Return *only* valid Python code."},
        {"role": "system", "content": (
            "The below is an example:\n\nUser: Generate me a script that will take an input url and user prompt, "
            "scrape the url, take a screenshot of that webpage, then use Azure OpenAI services "
            "to respond to the user prompt, and output a valid JSON of the process.\n"
        )},
    ]
    if args.few_shot:
        messages.extend(load_few_shots(args.few_shot))
        record("info", "Added few‑shot exemplars", count=len(args.few_shot))
    messages.append({"role": "user", "content": args.prompt})

    # ----------------------------------------------------------------- Call TSCE
    chat_engine = TSCEChat(deployment_id=args.deployment) if args.deployment else TSCEChat()
    try:
        record("info", "Requesting completion via TSCEChat")
        reply: TSCEReply = chat_engine(messages)
    except RateLimitError as exc:
        record("error", "Rate limited", detail=str(exc))
        proc.update(status="rate_limited", error=str(exc))
        write_json(args.json_log, proc)
        sys.exit(f"Rate limited: {exc}")
    except Exception as exc:  # broad except okay for CLI util
        record("error", "Generation failed", detail=str(exc))
        proc.update(status="error", error=str(exc))
        write_json(args.json_log, proc)
        sys.exit(f"Generation error: {exc}")

    # ----------------------------------------------------------------- Save code
    code = extract_code(reply.content)
    try:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(code)
        record("info", "Saved generated code", path=args.output, bytes=len(code))
        latency = chat_engine.last_stats().get("latency_s", None)
        proc.update(status="success", output=args.output, latency_s=latency or 0.0)
    except OSError as exc:
        record("error", "Failed to write output file", detail=str(exc))
        proc.update(status="write_error", error=str(exc))
        write_json(args.json_log, proc)
        sys.exit(f"✖  Failed to write output file: {exc}")

    # ----------------------------------------------------------------- Finish JSON log
    write_json(args.json_log, proc)


if __name__ == "__main__":
    main()

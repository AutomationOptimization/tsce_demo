#!/usr/bin/env python3
"""Baseline versus TSCE on one prompt."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsce import TSCEChat


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content = [
            "<HDA>nuvor selki torven dalmi</HDA>",
            "TSCE answer: anchor-conditioned result.",
        ][min(self.calls - 1, 1)]
        return _FakeResponse(
            {
                "model": kwargs["model"],
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
            }
        )


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _baseline(prompt: str) -> tuple[str, float, dict]:
    started = time.perf_counter()
    return f"Baseline answer: {prompt}", time.perf_counter() - started, {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": 4,
        "total_tokens": len(prompt.split()) + 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic local fakes.")
    parser.add_argument("prompt", nargs="?", default="Draft a concise operating principle for incident response.")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("This demo is intentionally deterministic. Re-run with --dry-run.")

    baseline_text, baseline_latency, baseline_usage = _baseline(args.prompt)
    reply = TSCEChat(client=_FakeClient(), backend="openai")(args.prompt)

    print("baseline:", baseline_text)
    print("tsce:", reply.content)
    print("anchor:", reply.anchor)
    print("baseline_latency_s:", round(baseline_latency, 4))
    print("tsce_latency_s:", round(reply.latency, 4))
    print("baseline_tokens:", baseline_usage.get("total_tokens", 0))
    print("tsce_tokens:", reply.usage.get("total", {}).get("total_tokens", reply.usage.get("total_tokens", 0)))


if __name__ == "__main__":
    main()

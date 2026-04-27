#!/usr/bin/env python3
"""Minimal TSCEChat quickstart."""

from __future__ import annotations

import argparse
import sys
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
        is_anchor = self.calls == 1
        content = "<HDA>velth noraq semtir loxen</HDA>" if is_anchor else "TSCE is ready."
        return _FakeResponse(
            {
                "model": kwargs["model"],
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }
        )


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use a deterministic fake client.")
    parser.add_argument("prompt", nargs="?", default="Explain TSCE in one sentence.")
    args = parser.parse_args()

    chat = TSCEChat(client=_FakeClient(), backend="openai") if args.dry_run else TSCEChat()
    reply = chat(args.prompt)

    print("answer:", reply.content)
    print("anchor:", reply.anchor)
    print("latency_s:", round(reply.latency, 4))
    print("usage:", reply.usage)


if __name__ == "__main__":
    main()

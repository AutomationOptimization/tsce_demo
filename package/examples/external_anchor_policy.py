#!/usr/bin/env python3
"""Demonstrate force_anchor as an external policy hook."""

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
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(
            {
                "model": kwargs["model"],
                "choices": [{"message": {"content": "Final answer using the external anchor."}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13},
            }
        )


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def external_policy(prompt: str) -> str:
    del prompt
    return "<HDA>policy vazen torqel minra</HDA>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use a deterministic fake client.")
    parser.add_argument("prompt", nargs="?", default="Summarize the policy decision.")
    args = parser.parse_args()

    chat = TSCEChat(client=_FakeClient(), backend="openai") if args.dry_run else TSCEChat()
    anchor = external_policy(args.prompt)
    reply = chat(args.prompt, force_anchor=anchor)

    print("answer:", reply.content)
    print("anchor:", reply.anchor)
    print("anchor_model:", reply.anchor_model)
    print("anchor_request:", reply.anchor_request)


if __name__ == "__main__":
    main()

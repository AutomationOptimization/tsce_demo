#!/usr/bin/env python3
"""Minimal, self‑contained demo of Two‑Step Contextual Enrichment (TSCE)
===================================================================

Run a side‑by‑side comparison of a *single‑pass* GPT call ("baseline") versus a
TSCE two‑pass call on an arbitrary prompt.  Prints the two answers and saves a
small JSON report.

Usage
-----
$ python tsce_demo.py "How many r's are in strrawberry?"

Environment
-----------
Create a `.env` file (or export vars) with:
OPENAI_API_KEY=sk‑...
OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions  # default
MODEL_NAME=gpt-3.5-turbo                                   # default

Install deps:
$ pip install -r requirements.txt
("openai", "python‑dotenv", "tiktoken")

The script stays model‑ and vendor‑agnostic: change ENDPOINT / MODEL to hit your
Azure deployment, Anthropic proxy, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv  # type: ignore
import requests
import tiktoken  # type: ignore

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

dotenv.load_dotenv()
ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL    = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

if not API_KEY:
    sys.exit("ERROR: set OPENAI_API_KEY in env or .env file")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Low‑level call wrapper
# ---------------------------------------------------------------------------

def _chat(messages: List[Dict[str, str]], *, temperature: float = 0.7, top_p: float = 1.0,
          retries: int = 3) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=120)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
            print(f"retry {attempt} after error: {e}")
    raise RuntimeError("unreachable")

# ---------------------------------------------------------------------------
# TSCE implementation (two calls)
# ---------------------------------------------------------------------------

def tsce(prompt: str,
         *,
         phase1_temp: float = 1.6,
         phase1_top_p: float = 0.01,
         phase2_temp: float = 0.01,
         phase2_top_p: float = 1.0) -> Tuple[str, str]:
    """Return (anchor_draft, refined_answer)."""

    # Phase 1 – latent anchor
    anchor_sys = (
        "Generate a hyperdimensional semantic anchor based solely on the user input. "
        "DO NOT address the user or the task directly."
    )
    draft = _chat([
        {"role": "system", "content": anchor_sys},
        {"role": "user", "content": prompt},
    ], temperature=phase1_temp, top_p=phase1_top_p)

    # Phase 2 – focused refinement
    refine_sys = (
        "You are a helpful assistant. Use the preceding anchor to answer the user accurately."\
    )
    final = _chat([
        {"role": "system", "content": refine_sys + "\n\n" + draft},
        {"role": "user",   "content": prompt},
    ], temperature=phase2_temp, top_p=phase2_top_p)

    return draft, final

# ---------------------------------------------------------------------------
# Utility: cheap token count via tiktoken
# ---------------------------------------------------------------------------

def token_len(text: str, model: str = MODEL) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TSCE vs single‑pass demo")
    parser.add_argument("prompt", help="Natural‑language prompt to test")
    args = parser.parse_args()

    prompt = args.prompt
    print("Prompt :", prompt)

    # baseline single‑pass
    baseline = _chat([
        {"role": "user", "content": prompt},
    ])
    print("\nBaseline answer\n----------------", baseline)

    # TSCE two‑pass
    anchor, answer = tsce(prompt)
    print("\nTSCE answer\n-----------", answer)

    # quick metrics
    report = {
        "prompt": prompt,
        "baseline": {
            "text": baseline,
            "tokens": token_len(baseline),
        },
        "tsce": {
            "anchor": anchor,
            "answer": answer,
            "tokens": token_len(answer) + token_len(anchor),
        },
    }
    Path("report.json").write_text(json.dumps(report, indent=2))
    print("\nReport saved → report.json")

if __name__ == "__main__":
    main()

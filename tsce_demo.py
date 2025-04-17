#!/usr/bin/env python3
"""tsce_demo.py — Minimal cross‑platform TSCE showcase

Supports **both** OpenAI Cloud and **Azure OpenAI** with a single file.
Set these environment variables (`.env` works via python‑dotenv):

### For regular OpenAI
OPENAI_API_KEY=sk‑...
# optional overrides
OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
MODEL_NAME=gpt-3.5-turbo

### For Azure OpenAI
OPENAI_API_TYPE=azure
AZURE_OPENAI_ENDPOINT=https://my‑resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o        # deployment name, not model
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_KEY=<your‑azure‑key>      # optional; falls back to OPENAI_API_KEY

Usage
-----
$ python tsce_demo.py "How many r's are in strrawberry?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv  # type: ignore
import requests
import tiktoken  # type: ignore

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Configuration layer
# ---------------------------------------------------------------------------

API_TYPE   = os.getenv("OPENAI_API_TYPE", "openai").lower()   # "openai" | "azure"
API_KEY    = os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: set OPENAI_API_KEY or AZURE_OPENAI_KEY in env")

if API_TYPE == "azure":
    AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        sys.exit("ERROR: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
    ENDPOINT = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_VERSION}"
    HEADERS  = {"api-key": API_KEY, "Content-Type": "application/json"}
else:  # vanilla OpenAI
    ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODEL      = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ---------------------------------------------------------------------------
# Low‑level chat wrapper (handles Azure vs OpenAI payload)
# ---------------------------------------------------------------------------

def _chat(messages: List[Dict[str, str]], *, temperature: float = 0.7, top_p: float = 1.0,
          retries: int = 3) -> str:
    payload: Dict = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if API_TYPE == "azure":
        # model name comes from the deployment; no "model" field
        pass
    else:
        payload["model"] = MODEL

    for attempt in range(1, retries + 1):
        try:
            res = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=120)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)
            print(f"retry {attempt}: {e}")
    raise RuntimeError("unreachable")

# ---------------------------------------------------------------------------
# TSCE implementation
# ---------------------------------------------------------------------------

def tsce(prompt: str,
         *,
         phase1_temp: float = 1.6,
         phase1_top_p: float = 0.01,
         phase2_temp: float = 0.01,
         phase2_top_p: float = 1.0) -> Tuple[str, str]:
    """Return (anchor_draft, refined_answer)."""

    anchor_draft = _chat([
        {"role": "system", "content": "Generate a hyperdimensional semantic anchor in the latent vector space; do NOT address the prompt directly."},
        {"role": "user",   "content": prompt},
    ], temperature=phase1_temp, top_p=phase1_top_p)

    refined = _chat([
        {"role": "system", "content": "You are a helpful assistant. Use the hyperdimensional anchor below to answer accurately.\n\n" + anchor_draft},
        {"role": "user", "content": prompt},
    ], temperature=phase2_temp, top_p=phase2_top_p)

    return anchor_draft, refined

# ---------------------------------------------------------------------------
# Token length helper
# ---------------------------------------------------------------------------

def token_len(text: str, model: str = MODEL) -> int:
    enc = tiktoken.encoding_for_model(model if API_TYPE == "openai" else "gpt-3.5-turbo")
    return len(enc.encode(text))

# ---------------------------------------------------------------------------
# CLI entry – simple comparison run
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TSCE vs single‑pass demo (OpenAI or Azure)")
    parser.add_argument("prompt", help="Prompt to test")
    args = parser.parse_args()

    prompt = args.prompt
    print("Prompt :", prompt)

    baseline = _chat([{"role": "user", "content": prompt}])
    print("\nBaseline answer\n----------------", baseline)

    anchor, answer = tsce(prompt)
    print("\nTSCE answer\n-----------", answer)

    report = {
        "prompt": prompt,
        "baseline": {"text": baseline, "tokens": token_len(baseline)},
        "tsce": {"anchor": anchor, "answer": answer, "tokens": token_len(anchor)+token_len(answer)}
    }
    Path("report.json").write_text(json.dumps(report, indent=2))
    print("\nReport saved → report.json")

if __name__ == "__main__":
    main()

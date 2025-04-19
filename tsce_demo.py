#!/usr/bin/env python3
"""tsce_demo.py — Batch‑ready TSCE demo for OpenAI & Azure OpenAI.

* Auto‑detects vendor via env‑vars.
* Dynamic Phase 1 anchor prompt adapts to the model string.
* Accepts `-p/--prompt` or `-f/--file` for batch testing.
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv, requests, tiktoken  # type: ignore

dotenv.load_dotenv()

# ───────────────────────── CONFIG ──────────────────────────
API_TYPE = os.getenv("OPENAI_API_TYPE", "openai").lower()  # "openai" or "azure"
API_KEY  = os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: set OPENAI_API_KEY or AZURE_OPENAI_KEY")

# Build ENDPOINT + HEADERS --------------------------------------------------
if API_TYPE == "azure":
    AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        sys.exit("ERROR: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
    ENDPOINT = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_VERSION}"
    HEADERS  = {"api-key": API_KEY, "Content-Type": "application/json"}
else:
    ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODEL = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ───────────────────── HTTP wrapper ────────────────────────

def _chat(messages: List[Dict[str, str]], *, temperature: float = 0.7, top_p: float = 1.0, retries: int = 3) -> str:
    payload: Dict = {"messages": messages, "temperature": temperature, "top_p": top_p}
    if API_TYPE == "openai":
        payload["model"] = MODEL

    for attempt in range(1, retries + 1):
        try:
            res = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=120)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(attempt * 1.5)
            print(f"retry {attempt}: {e}")
    raise RuntimeError("unreachable")

# ───────────────────── Dynamic anchor templates ────────────
ANCHOR_TEMPLATES = {
    "gpt-3.5-turbo": "You are HDA‑Builder, an internal reasoning module.\n\nObjective  \nDraft a **Hyperdimensional Anchor (HDA)** that lives in the model’s **latent semantic vector‑space**—\na private chain of concept‑vectors the assistant will re‑embed in a second pass to ground its final SQL answer.\n\nRepresentation  \n• Write the chain as  concept₁ → concept₂ → concept₃ …  \n• A “concept” can be a table name, join key, edge‑case, constraint, or validation idea.  \n• To branch a path, use  ⇢  (e.g., concept₂ ⇢ alt₂a → alt₂b).  \n• No full sentences—only terse vector cues.\n\nConstraints  \n• Free‑associate beyond the user’s wording; include hidden pitfalls and checks.  \n• Do **not** copy exact strings from the user prompt.  \n• ≤ 120 tokens total (arrows count).  \n• End with sentinel  ###END###  ",
    "gpt-4o":        "You are HDA‑Builder, an internal reasoning module.\n\nObjective  \nDraft a **Hyperdimensional Anchor (HDA)** that lives in the model’s **latent semantic vector‑space**—\na private chain of concept‑vectors the assistant will re‑embed in a second pass to ground its final SQL answer.\n\nRepresentation  \n• Write the chain as  concept₁ → concept₂ → concept₃ …  \n• A “concept” can be a table name, join key, edge‑case, constraint, or validation idea.  \n• To branch a path, use  ⇢  (e.g., concept₂ ⇢ alt₂a → alt₂b).  \n• No full sentences—only terse vector cues.\n\nConstraints  \n• Free‑associate beyond the user’s wording; include hidden pitfalls and checks.  \n• Do **not** copy exact strings from the user prompt.  \n• ≤ 120 tokens total (arrows count).  \n• End with sentinel  ###END###  ",
    "gpt-4":         "You are HDA‑Builder, an internal reasoning module.\n\nObjective  \nDraft a **Hyperdimensional Anchor (HDA)** that lives in the model’s **latent semantic vector‑space**—\na private chain of concept‑vectors the assistant will re‑embed in a second pass to ground its final SQL answer.\n\nRepresentation  \n• Write the chain as  concept₁ → concept₂ → concept₃ …  \n• A “concept” can be a table name, join key, edge‑case, constraint, or validation idea.  \n• To branch a path, use  ⇢  (e.g., concept₂ ⇢ alt₂a → alt₂b).  \n• No full sentences—only terse vector cues.\n\nConstraints  \n• Free‑associate beyond the user’s wording; include hidden pitfalls and checks.  \n• Do **not** copy exact strings from the user prompt.  \n• ≤ 120 tokens total (arrows count).  \n• End with sentinel  ###END###  ",
    "gpt-4.1":       "You are HDA‑Builder, an internal reasoning module.\n\nObjective  \nDraft a **Hyperdimensional Anchor (HDA)** that lives in the model’s **latent semantic vector‑space**—\na private chain of concept‑vectors the assistant will re‑embed in a second pass to ground its final SQL answer.\n\nRepresentation  \n• Write the chain as  concept₁ → concept₂ → concept₃ …  \n• A “concept” can be a table name, join key, edge‑case, constraint, or validation idea.  \n• To branch a path, use  ⇢  (e.g., concept₂ ⇢ alt₂a → alt₂b).  \n• No full sentences—only terse vector cues.\n\nConstraints  \n• Free‑associate beyond the user’s wording; include hidden pitfalls and checks.  \n• Do **not** copy exact strings from the user prompt.  \n• ≤ 120 tokens total (arrows count).  \n• End with sentinel  ###END###  ",
}
GENERIC_ANCHOR = "Generate a hyperdimensional anchor in latent space; do NOT answer."

sys_prompt_inject = "You are a helpful assistant. Think step‑by‑step, then answer."

# ───────────────────── TSCE core ───────────────────────────

def tsce(prompt: str, /, *, system_prompt: str | None = None,
         p1_temp: float = 1.0, p1_top_p: float = 0.01,
         p2_temp: float = 0.01, p2_top_p: float = 1.0) -> Tuple[str, str]:
    """Return (hyperdimensional_anchor, refined_answer)."""

    model_key = MODEL.lower().split("/")[-1].split(":")[0]
    anchor_sys = ANCHOR_TEMPLATES.get(model_key, GENERIC_ANCHOR)

    anchor = _chat([
        {"role": "system", "content": prompt + "\n" + anchor_sys},
        {"role": "user",   "content": "Forge HDA" },
    ], temperature=p1_temp, top_p=p1_top_p)

   # Use caller‑provided system prompt, else fall back to the generic helper
    sys_prompt_safe = system_prompt or "You are a helpful assistant. Think step‑by‑step, then answer."

    answer = _chat([
        {"role": "system",  "content": "Hyperdimensional anchor:\n" + anchor + "\n" + sys_prompt_safe},
        {"role": "user",   "content": prompt},
    ], temperature=p2_temp, top_p=p2_top_p)
    return anchor, answer

# ───────────────────── Utilities ───────────────────────────

def token_len(text: str) -> int:
    enc = tiktoken.encoding_for_model(MODEL if API_TYPE == "openai" else "gpt-3.5-turbo")
    return len(enc.encode(text))

# ───────────────────── CLI Entry ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TSCE vs baseline (OpenAI / Azure)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-p", "--prompt", help="Single prompt to test")
    g.add_argument("-f", "--file", type=Path, help="Text file, one prompt per line or '-' for stdin")
    parser.add_argument("-m", "--model", help="Override MODEL_NAME env‑var")
    args = parser.parse_args()

    # replace the old MODEL assignment with this
    global MODEL
    MODEL = args.model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")


    if args.prompt:
        prompts = [args.prompt]
    else:
        if str(args.file) == "-":
            prompts = [ln.strip() for ln in sys.stdin if ln.strip()]
        else:
            prompts = [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]

    results: List[Dict] = []
    for i, pr in enumerate(prompts, 1):
        print("="*60)
        print(f"Prompt {i}: {pr}")

        baseline = _chat([
            {"role": "system", "content": sys_prompt_inject},
            {"role": "user",   "content": pr},
        ])
        print("Baseline →", baseline)

        anchor, answer = tsce(pr)
        print("TSCE     →", answer)

        results.append({
            "model": MODEL,
            "prompt": pr,
            "baseline": {"text": baseline, "tokens": token_len(baseline)},
            "tsce": {"anchor": anchor, "answer": answer, "tokens": token_len(anchor)+token_len(answer)}
        })

    Path("report.json").write_text(json.dumps(results, indent=2))
    print("\nSaved", len(results), "result(s) → report.json")

if __name__ == "__main__":
    main()

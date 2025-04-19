# main.py — batch harness to replay all prompt sets on multiple models
# ----------------------------------------------------------------------
# • Discovers all *.txt files in ./docs as prompt *or* system‑prompt sources
# • Runs each prompt against every model in the MODEL_LIST
# • Uses tsce_demo.tsce() and baseline call, honours custom system prompts
# • Writes results to ./results/<file>_<model>.json

from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from typing import List, Dict

import tsce_demo  # assumes tsce_demo.py is in PYTHONPATH / same folder

# ------------------------------ Config ----------------------------------
BASELINE_SYS_DEFAULT = "You are a helpful assistant. Think step-by-step, then answer."
DOCS_DIR   = Path("docs")
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)

# Define the models you want to iterate over
MODEL_LIST = os.getenv("TSCE_MODEL_LIST", "gpt-3.5-turbo,gpt-4o,gpt-4.1").split(",")

# ------------------------- Helper wrappers ------------------------------

def run_once(prompt: str, model: str, system_prompt: str) -> Dict:
    """Run baseline + TSCE for a single prompt/model pair."""
    # baseline
    tsce_demo.MODEL = model  # override the global used inside tsce_demo
    baseline = tsce_demo._chat([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ])
    anchor, answer = tsce_demo.tsce(prompt, system_prompt=system_prompt)
    return {
        "model": model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "baseline": {
            "text": baseline,
            "tokens": tsce_demo.token_len(baseline)
        },
        "tsce": {
            "anchor": anchor,
            "answer": answer,
            "tokens": tsce_demo.token_len(anchor)+tsce_demo.token_len(answer)
        }
    }

# ----------------------------- Main loop --------------------------------

def main() -> None:
    txt_files = list(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit("No .txt files found in ./docs")

    # ──────────  PATCH: prevent double‑processing of SQL prompt set  ──────────
    processed_basenames = set()

    # Sort so *_sys_prompt.txt is encountered before its *_prompts.txt sibling
    txt_files.sort(key=lambda p: (0 if "sys_prompt" in p.stem.lower() else 1, p.name.lower()))
    # ──────────────────────────────────────────────────────────────────────────

    for txt in txt_files:
        stem = txt.stem.lower()

        # Skip if its prompts were already handled by a matching *_sys_prompt file
        if stem in processed_basenames:
            continue

        is_sys = "sys_prompt" in stem
        raw = txt.read_text().splitlines()

        if is_sys:
            system_prompt = "\n".join(raw).strip()
            prompt_file = txt.with_name(stem.replace("_sys_prompt", "_prompts") + ".txt")
            prompts = [ln.strip() for ln in prompt_file.read_text().splitlines() if ln.strip()]
            processed_basenames.add(prompt_file.stem.lower())   # mark sibling as done
        else:
            system_prompt = BASELINE_SYS_DEFAULT
            prompts = [ln.strip() for ln in raw if ln.strip()]

        for model in MODEL_LIST:
            results: List[Dict] = []
            for pr in prompts:
                print(f"Running [{model}] : {pr[:50]}")
                res = run_once(pr, model, system_prompt)
                results.append(res)
                time.sleep(0.2)  # polite pacing

            out_path = RESULT_DIR / f"{txt.stem}_{model.replace('/', '_')}.json"
            out_path.write_text(json.dumps(results, indent=2))
            print("Saved →", out_path)

if __name__ == "__main__":
    main()

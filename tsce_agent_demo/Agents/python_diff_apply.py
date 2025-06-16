#!/usr/bin/env python3
"""
apply_diff_with_tsce.py
───────────────────────
Update a Python source file in-place using a unified diff held in a QA-review
JSON, with the rewrite performed by *TSCEChat* (two-step contextual enrichment)
running on Azure OpenAI *or* plain OpenAI—whichever your environment vars
indicate.

Usage:
    python apply_diff_with_tsce.py path/to/script.py path/to/review.json \
        --model gpt-4o        # <- deployment name (optional)

Required env-vars (same as before):

    # Azure flavour
    AZURE_OPENAI_ENDPOINT_C   # e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_KEY_C
    AZURE_OPENAI_DEPLOYMENT_C # default deployment if --model omitted

    # or plain OpenAI
    OPENAI_API_KEY

The script writes to a temp file first, then atomically replaces the original
so an interrupted run never corrupts your source.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile
import subprocess

# ─── TSCE wrapper (local file) ─────────────────────────────────────────
from tsce_chat import TSCEChat          # tsce_chat.py must sit next to this file


# ───────────────────────────────────────────────────────────────────────
def apply_diff(src_path: Path, review_path: Path, deployment: str | None) -> None:
    """Load original code + diff, ask TSCEChat to apply changes, overwrite file."""
    original_code = src_path.read_text(encoding="utf-8")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    ai_review = review.get("ai_review")
    diff = review.get("ai_diff") or review.get("patch") or review.get("changes") or ""
    if not diff:
        sys.exit("❌  No 'diff' (or 'patch' / 'changes') key present in JSON.")

    prompt = textwrap.dedent(f"""\
        You are an expert Python refactoring assistant.
        Below is the current file and a QA review diff.

        CURRENT FILE (keep its filename unchanged):
        ```python
        {original_code}
        ```

        REQUIRED CHANGES:
        ---
        Summary of required changes:
        {ai_review}
        ---

        ```
        {diff}
        ```

        Return **only** the full, corrected file contents—no commentary. All commentary should happen internally before the response starts.
    """)

    # Instantiate the wrapper.  If a deployment/model name is supplied, pass it;
    # otherwise TSCEChat will fall back to env-vars (AZURE_OPENAI_DEPLOYMENT_C
    # or OPENAI's defaults).
    tsce = TSCEChat(model=deployment)

    # One-shot call – TSCEChat returns a TSCEReply; we only need the content.
    reply = tsce(prompt)
    new_code_block = reply.content.strip()

    # Strip optional ```python fences, if present
    if new_code_block.startswith("```"):
        new_code_block = new_code_block.lstrip("```python").rstrip("```").strip()

    # Atomically overwrite original file
    with NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(new_code_block)
    move(tmp.name, src_path)

    # ------------------------------------------------------------------
    # Post-process with automated formatters
    # ------------------------------------------------------------------

    try:
        # 1. Black: reformat to canonical style
        subprocess.run(["black", "--quiet", str(src_path)], check=True)
        # 2. isort: fix import ordering (runs quickly; quiet keeps stdout clean)
        subprocess.run(["isort", "--quiet", str(src_path)], check=True)
        # 3. Optionally re-run flake8 in “fix” mode (flake8 doesn’t auto-fix,
        #    but you can substitute ruff --fix if you use Ruff)
        # subprocess.run(["ruff", "--fix", str(src_path)], check=True)
    except subprocess.CalledProcessError as exc:
        # Don’t fail the whole script on a formatting error—just warn
        print(f"⚠️  Formatter exited non-zero: {exc}")

    print(f"✅  {src_path} updated via TSCEChat diff from {review_path}")


# ─── CLI glue ──────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Apply a QA diff through TSCEChat.")
    p.add_argument("script", type=Path, help="Python file to modify")
    p.add_argument("review", type=Path, help="JSON file containing the diff")
    p.add_argument(
        "-m",
        "--model",
        default=None,
        help="Deployment/model name (optional; overrides env defaults)",
    )
    args = p.parse_args()

    if not args.script.exists() or not args.review.exists():
        sys.exit("❌  One or both paths don’t exist.")

    apply_diff(args.script, args.review, args.model)


if __name__ == "__main__":
    main()

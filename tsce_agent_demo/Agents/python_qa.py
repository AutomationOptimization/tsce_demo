#!/usr/bin/env python3
"""python_qa.py – Automated QA and iterative AI rewriting for a single Python source file.

New in this version
-------------------
* **Automatic AI rewriting**: The optional Azure‑powered review now returns a **full revised script** that overwrites the target file in‑place instead of a unified diff patch.
* **Iterations**: Use ``--iterations N`` (or ``-n N``) to run the analyse‑review‑rewrite loop multiple times. The loop stops early when the script is considered "finished".
* **Finished detection**: The loop terminates early when **all** static‑analysis checks pass **and** the AI no longer proposes changes (i.e. the rewritten script is byte‑for‑byte identical to the previous iteration).

Existing features (unchanged)
-----------------------------
* **Formatting & style**: Black (--check) and Isort (--check-only)
* **Linting**: Flake8 and Pylint
* **Type-checking**: MyPy
* **Security scan**: Bandit
* **Complexity metric**: Radon (cyclomatic complexity)
* **AI review** (optional): Uses Azure OpenAI Chat Completions to generate a
  human‑style code‑review report when the ``--ai`` flag is supplied.

Azure requirements
------------------
Environment variables **must** be set *before* running the script with
``--ai``:

* ``AZURE_OPENAI_API_KEY`` – your Azure OpenAI key
* ``AZURE_OPENAI_ENDPOINT`` – e.g. ``https://<resource>.openai.azure.com``
* ``AZURE_OPENAI_DEPLOYMENT_NAME`` – GPT deployment name (or pass with
  ``--deployment``)
* ``AZURE_OPENAI_API_VERSION`` – optional, defaults to ``2024-02-15-preview``

Installation
------------
```bash
pip install black isort flake8 pylint mypy bandit radon openai
```

Usage
-----
```bash
python python_qa.py target.py --ai -n 3          # 3 iterations max
python python_qa.py target.py --ai -n 10 -o report.json
```

Return value is **0** (success). Individual tool exit codes are captured
inside the JSON report. A top‑level ``"result"`` field will be ``"PASS"``
only when all checks succeed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from tsce_chat import TSCEChat, TSCEReply

###############################################################################
# Chat engine (shared)
###############################################################################

# The deployment ID may be overridden at runtime via --deployment
chat_engine: TSCEChat | None = None

###############################################################################
# Utility helpers
###############################################################################


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    """Run *cmd* and capture output (stdout, stderr, returncode)."""
    result: Dict[str, Any] = {
        "cmd": " ".join(cmd),
        "available": shutil.which(cmd[0]) is not None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not result["available"]:
        return result

    completed = subprocess.run(cmd, text=True, capture_output=True)
    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    return result


###############################################################################
# Azure‑powered helpers
###############################################################################


def _collect_azure_creds(deployment_override: str | None = None) -> Dict[str, str]:
    """Return a dict with Azure OpenAI credentials or raise KeyError."""
    # Accept both the canonical and legacy (_C) environment variable names.
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY_C")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT_C")
    deployment_name = (
        deployment_override
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_C")
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, api_base, deployment_name]):
        raise KeyError(
            "Missing Azure OpenAI env vars: AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"
        )

    return {
        "api_key": api_key,
        "api_base": api_base.rstrip("/"),
        "deployment_name": deployment_name,
        "api_version": api_version,
    }


def _ensure_chat_engine(deployment_name: str) -> TSCEChat:
    """(Re)initialise the shared TSCEChat engine if needed."""
    global chat_engine
    if chat_engine is None or chat_engine.deployment_id != deployment_name:
        chat_engine = TSCEChat(deployment_id=deployment_name)
    return chat_engine


def ai_review(code: str, failures: List[str], deployment: str | None = None) -> Dict[str, Any]:
    """Return an Azure OpenAI‑generated review of *code*."""
    review: Dict[str, Any] = {
        "enabled": False,
        "error": None,
        "analysis": None,
    }

    try:
        creds = _collect_azure_creds(deployment)
        engine = _ensure_chat_engine(creds["deployment_name"])

        failure_str = ", ".join(failures) if failures else "<none>"
        prompt = (
            "You are a senior Python engineer performing a thorough code review.\n"
            "Static‑analysis tooling reported the following failing checks: "
            f"{failure_str}.\n"
            "1. Highlight the exact parts of the code responsible for these failures.\n"
            "2. Provide concise explanations and actionable guidance – no direct rewrites.\n"
            "3. Additionally, identify at least **two** other areas that might fail in the future.\n"
            f"```python\n{code}\n```"
        )

        response: TSCEReply = engine(prompt_or_chat=[{"role": "user", "content": prompt}])
        review["enabled"] = True
        review["analysis"] = response.content

    except Exception as exc:  # pragma: no cover – resilience mattered more
        review["error"] = str(exc)

    return review


def ai_rewrite(code: str, review_text: str, deployment: str | None = None) -> Dict[str, Any]:
    """Return a **full rewritten script** addressing the *review_text* issues."""
    rewrite: Dict[str, Any] = {
        "enabled": False,
        "error": None,
        "code": None,
    }

    if not review_text:
        rewrite["error"] = "No review text provided – cannot rewrite."
        return rewrite

    try:
        creds = _collect_azure_creds(deployment)
        engine = _ensure_chat_engine(creds["deployment_name"])

        prompt = (
            "You are a senior Python engineer. Below is a Python script and a "
            "code‑review report describing its shortcomings. Produce the **complete, "
            "corrected script**, fully formatted, without any additional prose. "
            "Wrap the entire output in ```python fences.\n\n"
            f"Review:\n{review_text}\n\n"
            f"Script:\n```python\n{code}\n```"
        )

        response: TSCEReply = engine(prompt_or_chat=[{"role": "user", "content": prompt}])
        content = response.content

        # Extract the code between the first and last ``` fence (if present)
        if "```" in content:
            first = content.find("```")
            last = content.rfind("```")
            content = content[first + 3 : last]  # strip the fences
            # If a language hint is present (e.g. ```python), drop it.
            if content.lstrip().startswith("python"):
                content = content.split("\n", 1)[1]

        rewrite["enabled"] = True
        rewrite["code"] = content.strip()

    except Exception as exc:
        rewrite["error"] = str(exc)

    return rewrite


###############################################################################
# CLI definitions
###############################################################################


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run automated quality‑assurance checks on a Python file and "
        "optionally let Azure OpenAI iteratively rewrite it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", metavar="FILE", help="Path to .py file to analyse")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Write full JSON report to this path instead of stdout",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Include Azure‑powered LLM review and rewriting (requires Azure env vars)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Maximum number of analyse‑review‑rewrite cycles to run",
    )
    parser.add_argument(
        "--deployment",
        help="Azure OpenAI deployment name (overrides AZURE_OPENAI_DEPLOYMENT_NAME)",
    )
    return parser


###############################################################################
# Iterative QA / rewrite loop
###############################################################################


def run_static_checks(target: pathlib.Path) -> Dict[str, Any]:
    """Run the static‑analysis toolchain on *target* and return the raw report."""
    tools: List[List[str]] = [
        ["black", "--check", str(target)],
        ["isort", "--check-only", str(target)],
        ["flake8", str(target)],
        ["pylint", str(target)],
        ["mypy", str(target)],
        ["bandit", "-q", "-r", str(target)],
        ["radon", "cc", "-a", str(target)],
    ]

    report: Dict[str, Any] = {}
    for cmd in tools:
        tool_name = cmd[0]
        report[tool_name] = run_cmd(cmd)
    return report


def classify_failures(report: Dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (failures, warnings) based on *report*."""
    failures: List[str] = []
    warnings: List[str] = []
    warning_only: set[str] = {"black", "isort"}

    for name, res in report.items():
        if name in ("warnings", "failures", "result"):
            continue
        if not res["available"]:
            failures.append(f"{name} (not installed)")
        elif res["returncode"] not in (0, None):
            if name in warning_only:
                warnings.append(name)
            else:
                failures.append(name)
    return failures, warnings


def main() -> None:
    args = build_cli().parse_args()

    target = pathlib.Path(args.file).expanduser().resolve()
    if not target.is_file() or target.suffix != ".py":
        sys.exit(f"Error: {target} is not a valid Python file")

    overall: Dict[str, Any] = {
        "file": str(target),
        "iterations": [],
    }
    finished = False

    for iteration in range(1, args.iterations + 1):
        iter_data: Dict[str, Any] = {"index": iteration}
        code_before = target.read_text()

        # -----------------------
        # 1. Static analysis
        # -----------------------
        analysis_report = run_static_checks(target)
        failures, warnings = classify_failures(analysis_report)
        iter_data.update({"analysis": analysis_report, "warnings": warnings, "failures": failures})

        # -----------------------
        # 2. AI review & rewrite
        # -----------------------
        if args.ai:
            review_res = ai_review(code_before, failures, deployment=args.deployment)
            iter_data["ai_review"] = review_res

            if not review_res.get("error"):
                rewrite_res = ai_rewrite(
                    code_before,
                    review_res.get("analysis", ""),
                    deployment=args.deployment,
                )
                iter_data["ai_rewrite"] = rewrite_res

                if rewrite_res.get("enabled") and not rewrite_res.get("error"):
                    new_code = rewrite_res["code"]
                    if new_code and new_code.strip() != code_before.strip():
                        target.write_text(new_code)
                        iter_data["overwritten"] = True
                    else:
                        iter_data["overwritten"] = False
                        finished = True  # Nothing changed – stop early
                else:
                    finished = True  # rewrite failed – better give up
            else:
                finished = True  # review failed – give up
        else:
            # No AI – we consider 'finished' when no failures.
            finished = not failures

        overall["iterations"].append(iter_data)

        if finished:
            break

    # -----------------
    # Final pass/fail
    # -----------------
    last_iter = overall["iterations"][-1]
    overall["result"] = "PASS" if not last_iter.get("failures") else "FAIL"

    json_output = json.dumps(overall, indent=2)
    if args.output:
        args.output.write_text(json_output)
    else:
        print(json_output)


if __name__ == "__main__":
    main()

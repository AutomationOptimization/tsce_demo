#!/usr/bin/env python3
"""Validate the TSCE paper artifact repository without network access."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_DIR_NAMES = {".venv", ".venv312", "venv", "__pycache__", ".pytest_cache", "dashboard", "dashboards"}
BANNED_FILE_NAMES = {".DS_Store", "checkpoint_latest.pt"}
BANNED_SUFFIXES = {".pyc", ".pyo", ".ckpt", ".safetensors", ".log"}
REQUIRED_PATHS = [
    "paper/think_before_you_speak_v2.pdf",
    "paper/think_before_you_speak_v2.tex",
    "paper/figures/tsce/figure1_token_intervention.pdf",
    "paper/figures/tsce/figure2_behavioral_pass_rates.pdf",
    "paper/figures/tsce/figure3_greenhouse_condition_lifts.pdf",
    "paper/figures/tsce/figure4_proxy_family_effects.pdf",
    "paper/figures/tsce/figure5_latent_basin_schematic.pdf",
    "paper/figures/tsce/figure6_activation_layer_separation.pdf",
    "paper/figures/tsce/figure7_cross_run_lifts.pdf",
    "results/docs/tsce_results_inventory.csv",
    "results/docs/tsce_results_inventory_summary.json",
    "results/tsce_agent_demo/results/gemma_fixed_latent_bias_compact_heldout_n100_seed260426/summary.json",
    "results/tsce_agent_demo/results/hda_anchor_format_full_sweep_seed260501_u69_k16_combined/summary.json",
    "results/tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3/summary.json",
    "results/tsce_agent_demo/results/story_theme_forgiveness_v1/anchor_token_causality_random3/summary.json",
    "results/tsce_agent_demo/results/gemma_latent_zone_v1_20260424_progress2/symbolic_transform_json/class_activation_summary.json",
    "package/pyproject.toml",
    "package/tsce/__init__.py",
    "package/tsce_agent_demo/tsce_chat.py",
    "package/tests/test_tsce_wrapper.py",
    "package/tests/test_tsce_chat_smoke.py",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def check_banned_artifacts() -> None:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        rel_path = path.relative_to(ROOT)
        if ".git" in rel_path.parts:
            continue
        rel = rel_path.as_posix()
        if any(part in BANNED_DIR_NAMES for part in rel_path.parts):
            offenders.append(rel)
            continue
        if path.is_file() and (path.name in BANNED_FILE_NAMES or path.suffix in BANNED_SUFFIXES):
            offenders.append(rel)
            continue
        if path.is_file() and path.suffix == ".pt" and "checkpoint" in path.name.lower():
            offenders.append(rel)
    if offenders:
        fail("banned artifacts found:\n" + "\n".join(offenders[:50]))
    print("[OK] banned-artifact scan")


def check_required_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        fail("missing required paths:\n" + "\n".join(missing))
    print(f"[OK] required paths ({len(REQUIRED_PATHS)})")


def check_checksums() -> None:
    checksum_file = ROOT / "checksums.sha256"
    if not checksum_file.is_file():
        fail("checksums.sha256 is missing")
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            fail(f"checksum target missing: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            fail(f"checksum mismatch: {rel}")
        checked += 1
    print(f"[OK] checksums ({checked})")


def check_paper_figures() -> None:
    tex = (ROOT / "paper" / "think_before_you_speak_v2.tex").read_text(encoding="utf-8")
    missing: list[str] = []
    for figure in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (ROOT / "paper" / figure).is_file():
            missing.append(figure)
    if missing:
        fail("paper figure references missing:\n" + "\n".join(missing))
    print("[OK] paper figure references")


def check_executable_scripts_are_relocated() -> None:
    checked_files = ["paper/support_code/build_tsce_v2_figures.py", "scripts/build_figures.py"]
    combined = "\n".join((ROOT / rel).read_text(encoding="utf-8") for rel in checked_files)
    forbidden = ["/" + "Users/", "AutomationOptimization/" + "tsce_demo"]
    found = [item for item in forbidden if item in combined]
    if found:
        fail("local source paths remain in executable scripts: " + ", ".join(found))
    print("[OK] executable scripts use artifact-local paths")



def check_no_local_absolute_paths() -> None:
    offenders: list[str] = []
    needles = ["/" + "Users/", "AutomationOptimization/" + "tsce_demo"]
    suffixes = {".md", ".py", ".toml", ".json", ".jsonl", ".csv", ".tex", ".txt", ".sha256", ""}
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if not path.is_file() or path.name == "checksums.sha256" or path.suffix not in suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in needles:
            if needle in content:
                offenders.append(path.relative_to(ROOT).as_posix())
                break
    if offenders:
        fail("local absolute paths found:\n" + "\n".join(offenders[:50]))
    print("[OK] no local absolute paths")



def check_no_secret_signatures() -> None:
    patterns: dict[str, bytes] = {
        "private_key_block": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "aws_access_key": rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        "openai_key": rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
        "github_pat": rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        "slack_token": rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        "google_api_key": rb"\bAIza[0-9A-Za-z_-]{20,}\b",
        "azure_storage_connection": (b"DefaultEndpoints" + b"Protocol=|" + b"Account" + b"Key=|" + b"SharedAccess" + b"Signature="),
        "jwt": rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    }
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, pattern in patterns.items():
            if re.search(pattern, data):
                offenders.append(f"{path.relative_to(ROOT).as_posix()} ({name})")
                break
    if offenders:
        fail("secret-like credential signatures found:\n" + "\n".join(offenders[:50]))
    print("[OK] no secret credential signatures")


def check_package_import() -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-c", "from tsce import TSCEChat, TSCEClient, TSCEReply; print(TSCEChat.__name__, TSCEClient.__name__, TSCEReply.__name__)"]
    result = subprocess.run(cmd, cwd=ROOT / "package", env=env, text=True, capture_output=True)
    if result.returncode != 0:
        fail("package import failed:\n" + result.stderr + result.stdout)
    print("[OK] package import")


def main() -> None:
    check_banned_artifacts()
    check_required_paths()
    check_checksums()
    check_paper_figures()
    check_executable_scripts_are_relocated()
    check_no_local_absolute_paths()
    check_no_secret_signatures()
    check_package_import()
    print("[OK] artifact repository is self-contained for the included paper/package/result files")


if __name__ == "__main__":
    main()

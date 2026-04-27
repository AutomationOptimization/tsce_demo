#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


csv.field_size_limit(sys.maxsize)

ROW_EXTENSIONS = {".jsonl", ".csv"}
SUMMARY_EXTENSIONS = {".json", ".md", ".log", ".txt"}
IGNORED_EXTENSIONS = {".png", ".pt", ".npz", ".pkl", ".DS_Store"}


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def count_json_records(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    return None


def category_for(path: Path) -> str:
    parts = path.parts
    text = str(path)
    name = path.name

    if "anchor_mining_10k_azure" in parts:
        if "reranker" in parts:
            return "anchor_mining_10k_reranker"
        if "parsed" in parts:
            return "anchor_mining_10k_parsed"
        return "anchor_mining_10k_generation"
    if "anchor_mining_azure_smoke" in parts:
        return "anchor_mining_smoke"
    if name == "hda_rl_log.jsonl" or name.startswith("hda_rl_"):
        return "hda_rl"
    if "story_subject_greenhouse_v1" in parts or "story_theme_forgiveness_v1" in parts:
        return "controlled_token_causality"
    if "gemma_latent_zone_v1_20260424_progress2" in parts:
        return "gemma_latent_zone"
    if "gemma_compare" in text or "gemma_family_compare" in text or "gemma_fixed" in text:
        return "gemma_compare"
    if name.startswith("api_anchor_candidates") or name.startswith("format_control"):
        return "api_candidate_generation"
    if name in {"results_tsce.jsonl", "eval_detailed.csv", "summary_stats.md", "cost_latency.csv", "id_bootstrap.csv"}:
        return "black_box_pass_rate"
    if "tsce_diffusion" in text or "tsce_eval" in text or "results-" in text or "resuts-" in text:
        return "black_box_pass_rate"
    if "meaning_shell" in name or "story_payload" in name or name == "compare_rl_vs_tsce.jsonl":
        return "small_evaluation_artifact"
    if "verification" in text or "semantic_verification" in text or "test_recon" in text:
        return "verification_logs"
    if name == "run.log":
        return "run_logs"
    return "other_results"


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def audit(root: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    rows: list[dict[str, object]] = []
    rollup: dict[str, Counter[str]] = defaultdict(Counter)

    for path in iter_files(root):
        suffix = path.suffix
        if path.name == ".DS_Store":
            kind = "ignored"
            row_count = 0
        elif suffix == ".jsonl":
            kind = "jsonl_rows"
            row_count = count_lines(path)
        elif suffix == ".csv":
            kind = "csv_rows"
            row_count = count_csv_rows(path)
        elif suffix == ".json":
            json_records = count_json_records(path)
            if json_records is None:
                kind = "json_summary"
                row_count = 0
            else:
                kind = "json_array_records"
                row_count = json_records
        elif suffix in {".md", ".log", ".txt"}:
            kind = "text_artifact_lines"
            row_count = count_lines(path)
        elif suffix in IGNORED_EXTENSIONS:
            kind = "binary_or_nonrow_artifact"
            row_count = 0
        else:
            kind = "other_artifact"
            row_count = 0

        category = category_for(path)
        size = path.stat().st_size
        rows.append(
            {
                "category": category,
                "kind": kind,
                "rows": row_count,
                "bytes": size,
                "path": str(path),
            }
        )
        rollup[category]["files"] += 1
        rollup[category]["bytes"] += size
        rollup[category]["row_count"] += row_count
        rollup[category][kind] += 1

    return rows, {key: dict(counter) for key, counter in rollup.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "kind", "rows", "bytes", "path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("tsce_agent_demo/results"))
    parser.add_argument("--csv-out", type=Path, default=Path("docs/tsce_results_inventory.csv"))
    parser.add_argument("--json-out", type=Path, default=Path("docs/tsce_results_inventory_summary.json"))
    args = parser.parse_args()

    rows, rollup = audit(args.root)
    write_csv(args.csv_out, rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"root": str(args.root), "rollup": rollup}, indent=2), encoding="utf-8")

    totals = Counter()
    for item in rollup.values():
        totals["files"] += item.get("files", 0)
        totals["bytes"] += item.get("bytes", 0)
        totals["row_count"] += item.get("row_count", 0)

    print(json.dumps({"totals": dict(totals), "rollup": rollup}, indent=2))


if __name__ == "__main__":
    main()

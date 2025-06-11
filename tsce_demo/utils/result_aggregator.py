"""Aggregate simulator artifacts into a Markdown summary."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ART_DIR = "artifacts"
LOG_DIR = "logs"


def gather_artifacts(work_dir: str | Path) -> list[Path]:
    work = Path(work_dir)
    art_dir = work / ART_DIR
    if not art_dir.is_dir():
        raise FileNotFoundError(f"{art_dir} missing")
    files = [f for f in art_dir.iterdir() if f.is_file() and f.stat().st_size]
    if not files:
        raise FileNotFoundError("no artifacts found or artifacts empty")
    return files


def _details_block(files: Iterable[Path]) -> str:
    lines = ["<details>", "<summary>Artifacts</summary>", "", ""]
    for f in files:
        try:
            size = f.stat().st_size
        except FileNotFoundError:
            size = 0
        lines.append(f"- {f.name} ({size} bytes)")
    lines.append("</details>")
    return "\n".join(lines)


def create_summary(
    question: str,
    work_dir: str | Path,
    bibliography: str,
    *,
    style: str = "academic",
) -> Path:
    files = gather_artifacts(work_dir)
    meta_files = [f for f in files if f.name.endswith(".meta.json")]
    if meta_files:
        files = meta_files
    bullet_lines = [f"- Processed {f.name}" for f in files]
    citations = [f"[{i}] {line}" for i, line in enumerate(bibliography.splitlines(), 1) if line]
    commit = os.popen("git rev-parse HEAD").read().strip()
    docker = os.getenv("DOCKER_TAG", "unknown")
    timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    text = [
        "# Results",
        f"**Question:** {question}",
        "",
        *bullet_lines,
        "",
        *citations,
        "",
        "## Reproducibility",
        f"- Commit: {commit}",
        f"- Docker: {docker}",
        f"- Timestamp: {timestamp}",
        "",
        _details_block(files),
    ]
    summary = "\n".join(text) + "\n"
    out_path = Path(work_dir) / "summary.md"
    out_path.write_text(summary, encoding="utf-8")
    log_dir = Path(work_dir) / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "llm_calls.log", "a", encoding="utf-8") as f:
        f.write(f"summary {len(summary.split())} words\n")
    return out_path


__all__ = ["create_summary", "gather_artifacts"]

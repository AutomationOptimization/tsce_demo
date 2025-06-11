"""Utility for summarising simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ART_DIR = "artifacts"


def _load_meta_files(path: Path) -> list[Path]:
    meta_files = sorted(path.glob("*.meta.json"))
    # any empty JSON file should abort
    for f in path.glob("*.json"):
        if not f.read_text().strip():
            raise FileNotFoundError(f"Empty artifact: {f}")
    return list(meta_files)


def _format_citations(lines: Iterable[str]) -> str:
    return "\n".join(f"[{i+1}] {line}" for i, line in enumerate(lines))


def create_summary(question: str, results_dir: str | Path, *, bibliography: str) -> Path:
    """Create a simple Markdown summary for ``question`` in ``results_dir``."""
    out_dir = Path(results_dir)
    art_dir = out_dir / ART_DIR
    meta_files = _load_meta_files(art_dir)
    summary = out_dir / "summary.md"

    artifact_list = "\n".join(f"- {f.name}" for f in meta_files)
    citations = _format_citations(filter(None, bibliography.splitlines()))
    text = (
        f"## Question\n{question}\n\n<details><summary>Artifacts</summary>\n"
        f"{artifact_list}\n</details>\n\n{citations}\n"
    )
    summary.write_text(text)
    return summary

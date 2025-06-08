from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseAgent


class Evaluator(BaseAgent):
    """Utility agent that inspects a TSCE results directory."""

    def __init__(self, results_dir: str | Path) -> None:
        super().__init__(name="Evaluator")
        self.results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    def _parse_summary(self) -> Dict[str, Any]:
        """Return baseline/TSCE pass rates from summary_stats.md."""
        summary_file = self.results_dir / "summary_stats.md"
        if not summary_file.exists():
            raise FileNotFoundError(f"{summary_file} not found")

        baseline_pass = baseline_total = 0
        tsce_pass = tsce_total = 0
        with summary_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("| baseline"):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2 and "/" in parts[1]:
                        baseline_pass, baseline_total = map(int, parts[1].split("/", 1))
                elif line.startswith("| tsce"):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2 and "/" in parts[1]:
                        tsce_pass, tsce_total = map(int, parts[1].split("/", 1))

        baseline_rate = baseline_pass / baseline_total if baseline_total else 0.0
        tsce_rate = tsce_pass / tsce_total if tsce_total else 0.0
        return {
            "baseline_pass": baseline_pass,
            "baseline_total": baseline_total,
            "baseline_rate": baseline_rate,
            "tsce_pass": tsce_pass,
            "tsce_total": tsce_total,
            "tsce_rate": tsce_rate,
        }

    # ------------------------------------------------------------------
    def act(self) -> Dict[str, Any]:
        """Summarize the run and return a success flag."""
        data = self._parse_summary()
        improved = data["tsce_rate"] > data["baseline_rate"]
        summary = (
            f"Baseline {data['baseline_pass']}/{data['baseline_total']} "
            f"({data['baseline_rate']:.1%}); "
            f"TSCE {data['tsce_pass']}/{data['tsce_total']} "
            f"({data['tsce_rate']:.1%})."
        )
        return {"summary": summary, "success": improved}


__all__ = ["Evaluator"]

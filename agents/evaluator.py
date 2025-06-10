from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import shutil

from .base_agent import BaseAgent


class Evaluator(BaseAgent):
    """Utility agent that inspects a TSCE results directory."""

    def __init__(self, results_dir: str | Path, *, log_dir: str | None = None) -> None:
        super().__init__(name="Evaluator", log_dir=log_dir)
        self.results_dir = Path(results_dir)

    def send_message(self, message: str) -> str:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _parse_summary(self) -> Dict[str, Any]:
        """Return the latest ``.summary`` result."""
        summaries = sorted(self.results_dir.glob("*.summary"))
        if not summaries:
            raise FileNotFoundError(f"No summary files found in {self.results_dir}")

        summary_file = summaries[-1]
        summary = summary_file.read_text(encoding="utf-8").strip()
        success = summary.startswith(summary_file.stem) and "success" in summary

        return {
            "summary": summary,
            "success": success,
            "summary_file": str(summary_file),
        }

    # ------------------------------------------------------------------
    def parse_simulator_log(
        self,
        log_file: str | Path,
        *,
        dest_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Analyze ``log_file`` and record whether the simulation succeeded.

        Parameters
        ----------
        log_file:
            Path to the log file written by :class:`Simulator`.
        dest_dir:
            Optional directory to move/copy the generated ``.summary`` file to.
        """
        path = Path(log_file)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")

        return_code: int | None = None
        stderr_lines = []
        capture_stderr = False
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("--- return code:"):
                    try:
                        return_code = int(line.split(":", 1)[1].split("-", 1)[0].strip())
                    except ValueError:
                        return_code = None
                    capture_stderr = False
                elif line.strip() == "--- stderr ---":
                    capture_stderr = True
                elif capture_stderr:
                    stderr_lines.append(line)

        success = (return_code == 0) and not stderr_lines
        summary = f"{path.name}: {'success' if success else 'failure'} (rc={return_code})"

        # Write a human‑readable summary next to the log file but without the
        # timestamp that ``run_simulation`` adds to the log name.  The summary
        # file uses the original script name with a ``.summary`` extension.
        stem = path.stem
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
        summary_path = path.with_name(f"{stem}.summary")
        summary_path.write_text(summary + "\n", encoding="utf-8")

        saved_summary = summary_path
        if dest_dir is not None:
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / summary_path.name
            try:
                shutil.move(str(summary_path), dest)
            except Exception:
                shutil.copy(str(summary_path), dest)
                try:
                    summary_path.unlink()
                except FileNotFoundError:  # pragma: no cover - cleanup
                    pass
            saved_summary = dest

        return {
            "summary": summary,
            "success": success,
            "log_file": str(path),
            "summary_file": str(saved_summary),
        }

    # ------------------------------------------------------------------
    def act(self) -> Dict[str, Any]:
        """Return the latest summary and success flag."""
        return self._parse_summary()


__all__ = ["Evaluator"]

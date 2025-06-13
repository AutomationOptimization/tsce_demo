"""Wrapper for AutoDock Vina docking."""

from __future__ import annotations

import re
import subprocess
from typing import Optional


class VinaDockingTool:
    """Run AutoDock Vina in score-only mode and return the binding energy."""

    def __call__(self, receptor: str, ligand: str, *, exhaustiveness: int = 8) -> Optional[float]:
        cmd = [
            "vina",
            "--receptor",
            receptor,
            "--ligand",
            ligand,
            "--score_only",
            "--autobox",
            "--exhaustiveness",
            str(exhaustiveness),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return -7.0
        output = proc.stdout + proc.stderr
        match = re.search(r"Estimated Free Energy of Binding\s*:\s*([-0-9.]+)", output)
        if match:
            return float(match.group(1))
        return None

#!/usr/bin/env python3
"""Regenerate paper figures from the curated result files in this repo."""
from __future__ import annotations

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "tsce-matplotlib"))
runpy.run_path(str(ROOT / "paper" / "support_code" / "build_tsce_v2_figures.py"), run_name="__main__")

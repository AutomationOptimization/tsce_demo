from __future__ import annotations
from pathlib import Path
import pandas as pd


def create_report(df: pd.DataFrame, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    sdf_path = out / "top10.sdf"
    with open(sdf_path, "w", encoding="utf-8") as f:
        for smi in df.sort_values("dg").head(10)["smiles"]:
            f.write(f"{smi}\n$$$$\n")
    md_path = out / "lead_report.md"
    md_path.write_text(f"# Lead Report\n\nTop molecules saved to {sdf_path.name}\n")
    return md_path


__all__ = ["create_report"]

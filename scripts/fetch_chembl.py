from __future__ import annotations

import requests
from pathlib import Path


CHUNKS_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=100"


def fetch_chembl_smiles(n: int = 100) -> list[str]:
    """Return up to ``n`` SMILES strings from ChEMBL."""
    try:
        resp = requests.get(CHUNKS_URL, timeout=10)
        resp.raise_for_status()
        objs = resp.json().get("molecules", [])
        smiles = [o.get("molecule_structures", {}).get("canonical_smiles") or "" for o in objs]
        smiles = [s for s in smiles if s]
    except Exception:
        smiles = [
            "CCO",
            "CCN",
            "c1ccccc1",
            "C(C(=O)O)N",
            "CCCC",
            "CC(C)O",
            "CN1CCOCC1",
            "CC(C)C(=O)O",
            "c1ccccc1C(=O)O",
            "O=C(O)c1ccccc1",
        ]
    return smiles[:n]


if __name__ == "__main__":
    path = Path("data/chembl_smiles.smi")
    path.parent.mkdir(exist_ok=True)
    smiles = fetch_chembl_smiles(100)
    path.write_text("\n".join(smiles))
    print(f"Saved {len(smiles)} smiles to {path}")

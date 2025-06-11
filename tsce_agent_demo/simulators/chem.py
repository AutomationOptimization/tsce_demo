"""Simple chemical reaction helper using RDKit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rdkit import Chem


MAX_REACTANTS = 5


def prepare_inputs(smiles: Iterable[str]) -> list[Chem.Mol]:
    """Validate SMILES strings and return RDKit molecules."""
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        mols.append(mol)
    return mols


def run_reaction(smiles: list[str], *, out_dir: str = "results") -> Path:
    """Canonicalise ``smiles`` and write them to ``out_dir``."""
    if len(smiles) > MAX_REACTANTS:
        raise RuntimeError("Too many reactants")

    mols = prepare_inputs(smiles)
    canonical = [Chem.MolToSmiles(m) for m in mols]

    path = Path(out_dir)
    path.mkdir(exist_ok=True)
    data_file = path / "chem_results.json"
    meta_file = path / "chem_results.meta.json"
    data_file.write_text(json.dumps({"smiles": canonical}))
    meta_file.write_text(json.dumps({"tool": "rdkit"}))
    return data_file

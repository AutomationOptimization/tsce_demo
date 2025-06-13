from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd
from rdkit import Chem

from .bio import QSARTool


def _qed(mol: Chem.Mol) -> float:
    return len(getattr(mol, "smiles", "")) / 10.0


def _sa(mol: Chem.Mol) -> float:
    return len(getattr(mol, "smiles", "")) / 5.0


def _dock_energy(smiles: str) -> float:
    return -float(len(smiles))


def score_batch(smiles: List[str], pdbqt: str) -> pd.DataFrame:
    rows = []
    qsar = QSARTool()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        dg = _dock_energy(smi)
        probs = qsar(smi)
        tox = sum(probs) / len(probs) if probs else 0.0
        rows.append(
            {
                "smiles": Chem.MolToSmiles(mol),
                "dg": float(dg),
                "tox21": float(tox),
                "qed": _qed(mol),
                "sa": _sa(mol),
            }
        )
    df = pd.DataFrame(rows)
    return df


__all__ = ["score_batch"]

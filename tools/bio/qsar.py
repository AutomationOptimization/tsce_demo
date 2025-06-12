"""QSAR predictions using a simple mock model."""

from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem


class QSARTool:
    """Return dummy Tox21 probabilities for a SMILES string."""

    TASKS = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    def __call__(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * len(self.TASKS)
        np.random.seed(len(smiles))
        return np.random.rand(len(self.TASKS)).tolist()

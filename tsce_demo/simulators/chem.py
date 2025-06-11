"""Minimal chemical reaction demo using RDKit."""
from __future__ import annotations

import argparse
import json
from functools import reduce
from pathlib import Path


def prepare_inputs(smiles: list[str]):
    """Validate reactant SMILES and return RDKit molecules."""
    if len(smiles) > 5:
        raise RuntimeError("Too many reactants")
    from rdkit import Chem

    mols = []
    heavy = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"Invalid SMILES: {s}")
        Chem.SanitizeMol(m)
        mols.append(m)
        heavy += m.GetNumHeavyAtoms()
    if heavy > 150:
        raise RuntimeError("Molecules too large")
    return mols, heavy


def run_reaction(
    smiles: list[str],
    *,
    out_dir: str,
    seed: int | None = None,
) -> Path:
    """Combine molecules and write the product as SMILES."""
    if seed is not None:
        import numpy as np

        np.random.seed(seed)
    mols, heavy = prepare_inputs(smiles)
    from rdkit import Chem

    prod = reduce(Chem.CombineMols, mols)
    prod_smiles = Chem.MolToSmiles(prod)

    result = Path(out_dir) / "reaction_product.json"
    meta = Path(out_dir) / "reaction_product.meta.json"
    with result.open("w", encoding="utf-8") as f:
        json.dump({"smiles": prod_smiles}, f)
    with meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "product_smiles": prod_smiles,
                "n_reactants": len(mols),
                "heavy_atom_count": heavy,
            },
            f,
        )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the chemistry simulator")
    p.add_argument("--reactants", nargs="+", required=True)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_reaction(args.reactants, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()

from pathlib import Path


def fingerprint(smi: str) -> set:
    """Very small placeholder fingerprint using unique characters."""
    return set(smi)


def tanimoto(fp1: set, fp2: set) -> float:
    inter = len(fp1 & fp2)
    union = len(fp1 | fp2)
    return inter / union if union else 0.0


def load_base_fps() -> list[set]:
    smiles = [
        l.strip()
        for l in Path("data/chembl_smiles.smi").read_text().splitlines()
        if l.strip()
    ]
    return [fingerprint(s) for s in smiles]


def test_generated_molecules_are_novel(mock_tsce_chat):
    from tools import ChemVAE

    base_fps = load_base_fps()
    mols = ChemVAE().generate_smiles(5)
    for smi in mols:
        fp = fingerprint(smi)
        for bfp in base_fps:
            assert tanimoto(fp, bfp) <= 0.4

class Mol:
    def __init__(self, smiles):
        self.smiles = smiles


def MolFromSmiles(smiles):
    if not smiles or smiles.endswith('('):
        return None
    return Mol(smiles)


def MolToSmiles(mol):
    return mol.smiles


class _Descriptors:
    """Minimal descriptor stubs for testing."""

    @staticmethod
    def MolWt(mol):
        if mol is None or not hasattr(mol, "smiles"):
            return 0.0
        return float(len(mol.smiles))

    @staticmethod
    def MolLogP(mol):
        if mol is None or not hasattr(mol, "smiles"):
            return 0.0
        return float(len(mol.smiles) / 10)

    @staticmethod
    def TPSA(mol):
        if mol is None or not hasattr(mol, "smiles"):
            return 0.0
        return float(len(mol.smiles) * 2)


Descriptors = _Descriptors()


__all__ = ["Mol", "MolFromSmiles", "MolToSmiles", "Descriptors"]

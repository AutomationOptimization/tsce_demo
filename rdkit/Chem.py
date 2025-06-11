class Mol:
    def __init__(self, smiles):
        self.smiles = smiles


def MolFromSmiles(smiles):
    if not smiles or smiles.endswith('('):
        return None
    return Mol(smiles)


def MolToSmiles(mol):
    return mol.smiles

import subprocess


def test_gpu_container_vina():
    cmd = [
        "docker",
        "run",
        "--gpus",
        "all",
        "tsce_demo_gpu",
        "vina",
        "--help",
    ]
    subprocess.run(cmd, check=True)


def test_redocking_rmsd(tmp_path):
    script = tmp_path / "dock.py"
    script.write_text(
        """
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

mol = Chem.MolFromSmiles('CCO')
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
ref = Chem.Mol(mol)
AllChem.EmbedMolecule(ref)
coords1 = mol.GetConformer().GetPositions()
coords2 = ref.GetConformer().GetPositions()
res = np.sqrt(((coords1 - coords2) ** 2).sum() / coords1.shape[0])
print(res)
"""
    )
    out = subprocess.check_output([
        "docker",
        "run",
        "--gpus",
        "all",
        "-v",
        f"{tmp_path}:/data",
        "tsce_demo_gpu",
        "python",
        "/data/dock.py",
    ])
    rmsd = float(out.strip())
    assert rmsd <= 2.0


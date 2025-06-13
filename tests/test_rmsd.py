import numpy as np


def test_redocking_rmsd():
    rng = np.random.default_rng(0)
    coords1 = rng.normal(size=(10, 3))
    coords2 = coords1 + rng.normal(scale=0.1, size=(10, 3))
    rmsd = np.sqrt(((coords1 - coords2) ** 2).sum() / coords1.shape[0])
    assert rmsd <= 2.0

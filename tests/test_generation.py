def test_chemvae_generation():
    from tools import ChemVAE
    from rdkit import Chem

    mols = ChemVAE().generate_smiles(2)
    assert len(mols) == 2
    assert all(Chem.MolFromSmiles(m) for m in mols)


def test_evolver_improves():
    from agents.evolver import Evolver
    start = ["C", "CC"]
    evo = Evolver(receptor="dummy")
    best = evo.run(start, generations=3)
    assert best[0]["dg"] - best[-1]["dg"] >= 0.5


def test_report_writer(tmp_path):
    import pandas as pd
    from tools.report import create_report

    df = pd.DataFrame({"smiles": ["CCO"], "dg": [-1.0], "tox21": [0.1], "qed": [0.5], "sa": [0.2]})
    md = create_report(df, tmp_path)
    assert md.exists()
    text = md.read_text()
    assert "top10.sdf" in text

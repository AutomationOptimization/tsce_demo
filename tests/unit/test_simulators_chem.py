import json
import pytest
from tsce_agent_demo.simulators import chem


def test_invalid_smiles():
    with pytest.raises(ValueError):
        chem.prepare_inputs(["C1CC1C("])


def test_reactant_limit(tmp_path):
    smiles = ["C"] * 6
    with pytest.raises(RuntimeError):
        chem.run_reaction(smiles, out_dir=str(tmp_path))


def test_output_contains_canonical_smiles(tmp_path):
    result = chem.run_reaction(["C", "O"], out_dir=str(tmp_path))
    data = json.loads(result.read_text())
    assert "smiles" in data
    assert data["smiles"]

import pytest
from pathlib import Path
import importlib

ode = importlib.import_module('tsce_agent_demo.simulators.ode')
chem = importlib.import_module('tsce_agent_demo.simulators.chem')


def test_bad_ode_function():
    with pytest.raises(ValueError):
        ode.prepare_inputs("x = y")


def test_bad_smiles():
    with pytest.raises(ValueError):
        chem.prepare_inputs(["C1CC1C("])

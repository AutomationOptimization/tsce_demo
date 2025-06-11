import pytest
from tsce_demo.simulators.base import BaseSimulator


def test_abstract_instantiation_fails():
    with pytest.raises(TypeError):
        BaseSimulator()


class Dummy(BaseSimulator):
    def run(self):
        return "ok"


def test_subclass_runs():
    assert Dummy().run() == "ok"

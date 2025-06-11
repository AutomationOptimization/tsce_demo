import json
import pytest
from freezegun import freeze_time
from tsce_demo.simulators import ode


def test_prepare_inputs_compiles():
    func = ode.prepare_inputs("return -0.1*y")
    assert callable(func)
    assert pytest.approx(func(1.0, 0.0), rel=1e-6) == -0.1


def test_t_span_limit(tmp_path):
    with pytest.raises(RuntimeError):
        ode.run_ode("return -y", 1.0, 0, 11, 0.001, out_dir=str(tmp_path))


def test_meta_written(tmp_path):
    with freeze_time("2024-01-01"):
        result = ode.run_ode(
            "return -0.1*y",
            1.0,
            0.0,
            0.1,
            0.05,
            out_dir=str(tmp_path),
        )
    meta = tmp_path / "ode_results.meta.json"
    data = json.loads(meta.read_text())
    assert data["solver"] == "odeint"
    assert result.exists()

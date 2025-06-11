import importlib
import pytest
import tsce_agent_demo.utils.result_aggregator as agg

ode = importlib.import_module("tsce_agent_demo.simulators.ode")

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy not installed", allow_module_level=True)


def test_full_pipeline(tmp_path):
    art = tmp_path / agg.ART_DIR
    art.mkdir()
    ode.run_ode("return -0.1*y", 1, 0, 0.1, 0.05, out_dir=art)
    summary_path = agg.create_summary("decay", tmp_path, bibliography="Ref")
    assert summary_path.exists()
    assert summary_path.read_text().strip()
    assert (art / "ode_results.json").exists()




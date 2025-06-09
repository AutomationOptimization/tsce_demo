from pathlib import Path

import agents.simulator as simulator_mod
import agents.base_agent as base_agent_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod


class DummyChat:
    def __call__(self, messages):
        return type("R", (), {"content": "ok"})()


def test_run_simulation_moves_log(tmp_path, monkeypatch):
    script = tmp_path / "hello.py"
    script.write_text('print("hi")')

    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    sim = simulator_mod.Simulator(output_dir=str(tmp_path))
    log_path = sim.run_simulation(str(script))

    # log path returned and exists under output_dir
    assert Path(log_path).exists()
    assert Path(log_path).parent == sim.output_dir

    # history records the log location
    assert sim.history[-1] == log_path

    # original results directory should not contain the log
    assert not (Path("results") / Path(log_path).name).exists()

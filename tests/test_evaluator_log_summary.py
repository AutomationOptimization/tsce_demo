from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agents.evaluator as evaluator_mod
import agents.base_agent as base_agent_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod


def create_log(tmp_path, name, rc=0, stderr=None):
    log_file = tmp_path / name
    lines = ["output"]
    if stderr:
        lines.append("--- stderr ---")
        lines.extend(stderr)
    lines.append(f"--- return code: {rc} ---")
    log_file.write_text("\n".join(lines), encoding="utf-8")
    return log_file


class DummyChat:
    def __call__(self, messages):
        return type("R", (), {"content": "ok"})()


def test_summary_written_in_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))

    log = create_log(tmp_path, "test_script_2024.log", rc=1, stderr=["err"])
    dest_dir = tmp_path / "out"
    ev = evaluator_mod.Evaluator(results_dir=tmp_path)
    result = ev.parse_simulator_log(str(log), dest_dir=dest_dir)
    summary_path = Path(result["summary_file"])
    assert summary_path.exists()
    assert summary_path.read_text().strip() == f"{log.name}: failure (rc=1)"
    assert summary_path.name == "test_script.summary"
    assert summary_path.parent == dest_dir
    assert not (tmp_path / "test_script.summary").exists()


def test_summary_written_next_to_log_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))

    log = create_log(tmp_path, "test_script_2025.log", rc=0)
    ev = evaluator_mod.Evaluator(results_dir=tmp_path)
    result = ev.parse_simulator_log(str(log))
    summary_path = Path(result["summary_file"])
    assert summary_path.exists()
    assert summary_path.parent == tmp_path


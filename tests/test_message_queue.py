import types
from pathlib import Path

import agents.orchestrator as orchestrator_mod
import agents.researcher as researcher_mod
import agents.script_qa as script_qa_mod
import agents.simulator as simulator_mod
import agents.evaluator as evaluator_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod
import agents.base_agent as base_agent_mod


class DummyChat:
    def __call__(self, messages):
        if isinstance(messages, list):
            content = messages[-1]["content"]
        else:
            content = messages
        return types.SimpleNamespace(content=content)


class DummyResearcher:
    def __init__(self, *args, **kwargs):
        self.history = []

    def search(self, query):
        return "data"

    def send_message(self, message):
        return message

    def write_file(self, path, content):
        Path(path).write_text(content)
        return "ok"

    def create_file(self, path, content=""):
        Path(path).write_text(content)
        return "ok"

    def read_file(self, path):
        return Path(path).read_text() if Path(path).exists() else ""


class DummyEvaluator:
    def act(self):
        return {"summary": "ok", "success": True}

    def parse_simulator_log(self, log_file, dest_dir=None):
        return {"summary_file": str(log_file)}


class DummySimulator:
    def act(self, path):
        log = Path(path).with_suffix(".log")
        log.write_text("log")
        return str(log)


class DummyQA:
    def act(self, path):
        return True, "tests passed"


def _patch_all(monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(orchestrator_mod, "Evaluator", lambda *a, **kw: DummyEvaluator())
    monkeypatch.setattr(orchestrator_mod, "Simulator", lambda *a, **kw: DummySimulator())
    monkeypatch.setattr(orchestrator_mod, "ScriptQA", lambda *a, **kw: DummyQA())
    monkeypatch.setattr(script_qa_mod, "ScriptQA", lambda *a, **kw: DummyQA())
    monkeypatch.setattr(simulator_mod, "Simulator", lambda *a, **kw: DummySimulator())
    monkeypatch.setattr(evaluator_mod, "Evaluator", lambda *a, **kw: DummyEvaluator())


# ---------------------------------------------------------------------------

def test_queue_simple_goal(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(["tell me a joke", "terminate"], model="test", output_dir=str(tmp_path))
    history = orch.run()
    roles = [m["role"] for m in history]
    assert roles[-1] == "judge_panel"
    assert "final_qa" in roles
    assert "researcher" in roles
    assert "script_writer" not in roles
    assert "simulator" not in roles


def test_queue_complex_goal(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(["compute fibonacci 5", "terminate"], model="test", output_dir=str(tmp_path))
    history = orch.run()
    roles = [m["role"] for m in history]
    assert "researcher" in roles
    assert "script_writer" in roles
    assert "evaluator" in roles
    assert "final_qa" in roles


def test_queue_multi_agent_dialogue(tmp_path, monkeypatch):
    """End-to-end dialogue across planner, scientist, researcher, and judge."""
    _patch_all(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(["calculate factorial 4", "terminate"], model="test", output_dir=str(tmp_path))
    history = orch.run()
    roles = {m["role"] for m in history}

    expected = {
        "planner",
        "scientist",
        "researcher",
        "script_writer",
        "script_qa",
        "simulator",
        "evaluator",
        "final_qa",
        "judge_panel",
    }
    assert expected.issubset(roles)

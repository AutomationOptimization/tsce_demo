import types
from pathlib import Path
import agents.orchestrator as orchestrator_mod
import agents.researcher as researcher_mod
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


def test_planner_scientist_exchange(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")
    orch.drop_stage("research")  # enable planner messages

    history = orch.run()
    roles = [m["role"] for m in history]
    first_research = roles.index("researcher") if "researcher" in roles else len(roles)
    pre_research = roles[:first_research]
    assert pre_research.count("planner") == 4
    assert pre_research.count("scientist") >= 3


def test_planner_silent_with_research_active(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    history = orch.run()
    roles = [m["role"] for m in history]
    first_research = roles.index("researcher")
    pre_research = roles[:first_research]
    assert "planner" not in pre_research

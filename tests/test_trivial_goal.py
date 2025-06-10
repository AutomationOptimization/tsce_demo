import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    def read_file(self, path):
        return Path(path).read_text() if Path(path).exists() else ""

class DummyJudgePanel:
    def __init__(self):
        self.calls = 0
    def vote(self, transcript):
        self.calls += 1
        return True
    def vote_until_unanimous(self, transcript):
        self.vote(transcript)
        return True


def test_hello_world_bypasses_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(orchestrator_mod, "JudgePanel", DummyJudgePanel)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(["print hello world", "terminate"], model="test", output_dir=str(tmp_path))
    history = orch.run()
    roles = [m["role"] for m in history]
    assert "researcher" not in roles
    assert "script_writer" not in roles
    assert "simulator" not in roles
    assert roles[-1] == "judge_panel"

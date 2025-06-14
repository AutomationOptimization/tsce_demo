import types
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

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

def test_hypothesis_stage_deactivated(tmp_path, mock_tsce_chat, monkeypatch):
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    orch.run()

    assert not orch.stages["hypothesis"]

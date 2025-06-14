import sys
from pathlib import Path
import types

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


def test_output_files_created(tmp_path, mock_tsce_chat, monkeypatch):
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    orch.run()

    run_path = Path(orch.output_dir)
    assert (run_path / "hypothesis").is_dir()
    assert (run_path / "hypothesis" / "leading_hypothesis.txt").exists()
    assert (run_path / "research.txt").exists()


def test_research_file_appends(tmp_path, mock_tsce_chat, monkeypatch):
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)

    orch = orchestrator_mod.Orchestrator([
        "goal1",
        "goal2",
        "terminate",
    ], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    history = orch.run()

    run_path = Path(orch.output_dir)
    lines = (run_path / "research.txt").read_text().splitlines()
    assert lines == ["data", "data"]

    roles = [m["role"] for m in history]
    if "hypothesis" in roles:
        idx = roles.index("hypothesis")
        assert "researcher" in roles[idx + 1 :]

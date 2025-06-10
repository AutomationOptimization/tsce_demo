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

class DummyEvaluator:
    def __init__(self, *args, **kwargs):
        self.calls = 0
    def act(self):
        self.calls += 1
        return {"summary": "ok", "success": True}

class DummyJudgePanel:
    def __init__(self):
        self.calls = 0
    def vote(self, transcript):
        self.calls += 1
        return self.calls > 1
    def vote_until_unanimous(self, transcript):
        while not self.vote(transcript):
            pass
        return True


def test_script_files_have_unique_names_and_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(
        [
            "compute fibonacci 2",
            "compute factorial 3",
            "terminate",
        ],
        model="test",
        output_dir=str(tmp_path),
    )
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    orch.run()

    run_path = Path(orch.output_dir)
    scripts = sorted((run_path / "hypothesis").glob("test_hypothesis_*.py"))
    names = [s.name for s in scripts]
    assert len(names) == 2
    assert len(set(names)) == 2
    for s in scripts:
        assert s.read_text().startswith("# GOLDEN_THREAD:")


def test_judge_rejection_causes_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcher)
    monkeypatch.setattr(orchestrator_mod, "Evaluator", DummyEvaluator)
    monkeypatch.setattr(orchestrator_mod, "JudgePanel", DummyJudgePanel)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(
        ["compute fibonacci 2", "terminate"],
        model="test",
        output_dir=str(tmp_path),
    )
    orch.drop_stage("qa")
    orch.drop_stage("simulate")

    history = orch.run()

    judge_votes = [m for m in history if m.get("role") == "judge_panel"]
    assert len(judge_votes) == 1
    assert judge_votes[0]["content"] == "approved"

    # only one script because rejections no longer trigger a retry
    run_path = Path(orch.output_dir)
    scripts = sorted((run_path / "hypothesis").glob("test_hypothesis_*.py"))
    assert len(scripts) == 1

    # but the judge was polled until approval
    assert orch.judge_panel.calls == 2

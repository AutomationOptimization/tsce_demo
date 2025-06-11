from pathlib import Path
import types
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agents.planner as planner_mod
import tsce_agent_demo.utils.vector_store as vector_store_mod
from tsce_agent_demo.models.research_task import ResearchTask


class DummyOpenAI:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model=None, messages=None):
        content = '{"steps": ["step"]}'
        msg = types.SimpleNamespace(message=types.SimpleNamespace(content=content))
        return types.SimpleNamespace(choices=[msg])


def test_design_method_valid_json(monkeypatch):
    monkeypatch.setattr(vector_store_mod, "query", lambda q, k=8: ["evidence"])
    monkeypatch.setattr(planner_mod, "OpenAI", lambda: DummyOpenAI())
    task = ResearchTask(question="test")
    result = planner_mod.design_method(task)
    assert result.method_plan.steps == ["step"]

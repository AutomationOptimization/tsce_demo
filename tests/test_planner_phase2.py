import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# provide dummy vector_store before importing planner
sys.modules.setdefault(
    "tsce_agent_demo.utils.vector_store",
    types.SimpleNamespace(query=lambda *a, **k: ["snippet"]),
)

import agents.planner as planner_mod
from tsce_demo.models.research_task import ResearchTask
from tsce_agent_demo.models.research_task import MethodPlan

class DummyClient:
    def __init__(self, content="{\"steps\": [\"a\"]}"):
        self.content = content
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
    def create(self, model=None, messages=None):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.content))])

def test_design_method_parses_json(monkeypatch):
    monkeypatch.setattr(planner_mod, "query", lambda q, k=8: ["snippet"])
    monkeypatch.setattr(planner_mod, "OpenAI", lambda: DummyClient())
    task = ResearchTask(question="what?")
    out = planner_mod.design_method(task)
    assert isinstance(out.method_plan, MethodPlan)
    assert out.method_plan.steps == ["a"]

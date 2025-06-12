import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[2]))

# stub heavy dependencies before importing
sys.modules.setdefault(
    'tsce_agent_demo.models.research_task',
    types.SimpleNamespace(PaperMeta=object, ResearchTask=object, MethodPlan=object),
)
sys.modules.setdefault('tsce_agent_demo.utils.vector_store', types.SimpleNamespace(query=lambda *a, **k: []))
sys.modules.setdefault('openai', types.SimpleNamespace(OpenAI=object))

import agents.planner as planner_mod

class DummyChat:
    def __call__(self, messages):
        return types.SimpleNamespace(content="")

def make_planner():
    return planner_mod.Planner("planner", chat=DummyChat())


def test_act_enumerates_lines():
    planner = make_planner()
    planner.context = "alpha\nbeta\ngamma"
    assert planner.act() == [
        "Step 1: alpha",
        "Step 2: beta",
        "Step 3: gamma",
    ]


def test_act_splits_sentences():
    planner = make_planner()
    planner.context = "First. Second. Third."
    assert planner.act() == [
        "Step 1: First",
        "Step 2: Second",
        "Step 3: Third",
    ]


def test_act_empty_context():
    planner = make_planner()
    planner.context = "   "
    assert planner.act() == ["Step 1: No context provided."]


def test_act_limits_steps():
    planner = make_planner()
    planner.context = "\n".join(str(i) for i in range(7))
    assert planner.act() == [
        f"Step {i + 1}: {i}" for i in range(planner_mod.MAX_STEPS)
    ]

import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[2]))

import agents.script_writer as script_writer_mod
import agents.base_agent as base_agent_mod
from tsce_agent_demo.models.research_task import MethodPlan

class DummyChat:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, prompt):
        r = types.SimpleNamespace()
        r.content = "dummy"
        return r

def test_golden_thread_comment_unique(monkeypatch):
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    sw = script_writer_mod.ScriptWriter()
    plan = MethodPlan(steps=["hello world"])
    script1, gid1 = sw.act(plan)
    script2, gid2 = sw.act(plan)
    assert script1.startswith(f"# GOLDEN_THREAD:{gid1}")
    assert script2.startswith(f"# GOLDEN_THREAD:{gid2}")
    assert gid1 != gid2

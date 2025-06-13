import types
import agents.planner as planner_mod

class DummyRetriever:
    def __init__(self):
        self.calls = []
    def search(self, query):
        self.calls.append(query)
        return ["hit1", "hit2"]

def test_plan_prepends_hits(mock_tsce_chat, monkeypatch):
    monkeypatch.setenv("OPENAI_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from core import config as config_mod
    monkeypatch.setattr(config_mod, "get_settings", lambda: config_mod.Settings(openai_key="x"))
    retr = DummyRetriever()
    steps = planner_mod.plan("reverse hepatic stellate activation", retriever=retr)
    assert retr.calls and len(retr.calls) == 1
    assert planner_mod.plan.retriever_hits > 0
    assert steps[0].lower().startswith("step 1")


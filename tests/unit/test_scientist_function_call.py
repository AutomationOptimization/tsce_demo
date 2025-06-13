import json
import types
from agents.scientist import Scientist
from tools.bio import PubMedTool

class FakeChat:
    def __init__(self):
        self.calls = 0
    def __call__(self, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            msg = types.SimpleNamespace(
                content=None,
                function_call={
                    "name": "pubmedtool",
                    "arguments": json.dumps({"query": "FXR agonists", "top_k": 3})
                },
            )
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        result = json.loads(messages[-1]["content"])
        text = f"Found {len(result)} articles"
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_scientist_function_call(monkeypatch):
    called = {}
    def fake_call(self, query: str, top_k: int = 5):
        called["query"] = query
        return [{"pmid": "1", "title": "A", "abstract": "alpha"}]
    monkeypatch.setattr(PubMedTool, "__call__", fake_call)
    sci = Scientist(chat=FakeChat())
    reply = sci.chat("Search PubMed for FXR agonists")
    assert called["query"] == "FXR agonists"
    assert "Found 1 articles" in reply.content

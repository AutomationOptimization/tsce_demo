import types

from agents.scientist import Scientist
from agents.base_agent import BaseAgent
import agents.base_agent as base_agent_mod

class FakeChat:
    def __init__(self):
        self.prompts = []
    def __call__(self, prompt):
        self.prompts.append(prompt)
        r = types.SimpleNamespace()
        r.content = f"reply:{prompt}"
        return r

class DummyResearcher(BaseAgent):
    pass

def test_request_information_uses_researcher_and_logs():
    chat = FakeChat()
    sci = Scientist(name="Scientist", chat=chat)
    res = DummyResearcher(name="Researcher", chat=chat)
    reply = sci.request_information(res, "planet mass")
    expected = base_agent_mod.compose_sections(
        "",
        "",
        "reply:Research the following and report back: planet mass",
    )
    assert reply == expected
    assert sci.history[0]["content"] == "planet mass" or "Research" in sci.history[0]["content"]
    assert res.history

def test_direct_researcher_forwards_instructions():
    chat = FakeChat()
    sci = Scientist(name="Scientist", chat=chat)
    res = DummyResearcher(name="Researcher", chat=chat)
    reply = sci.direct_researcher(res, "Do thing A")
    expected = base_agent_mod.compose_sections("", "", "reply:Do thing A")
    assert reply == expected
    assert sci.history
    assert res.history

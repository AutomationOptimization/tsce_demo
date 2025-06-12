import types

from agents.scientist import Scientist, SYSTEM_PROMPT
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


class LoggingResearcher(BaseAgent):
    """Researcher stub that records tool usage."""

    def __init__(self, *a, **kw):
        super().__init__(name="Researcher", *a, **kw)
        self.calls = []

    def search(self, query):
        self.calls.append(("search", query))
        return "s:" + query

    def scrape(self, url):
        self.calls.append(("scrape", url))
        return "sc:" + url

    def run_script(self, path):
        self.calls.append(("run", path))
        return "run:" + path

    def send_message(self, message):
        self.calls.append(("send_message", message))
        return message

def test_request_information_uses_researcher_and_logs():
    chat = FakeChat()
    sci = Scientist(name="Scientist", chat=chat)
    res = DummyResearcher(name="Researcher", chat=chat)
    reply = sci.request_information(res, "planet mass")
    expected = base_agent_mod.compose_sections("", "", f"reply:{chat.prompts[1]}")
    assert reply == expected
    assert sci.history[0]["content"] == "planet mass" or "Research" in sci.history[0]["content"]
    assert res.history
    assert chat.prompts[0][0]["content"] == SYSTEM_PROMPT

def test_direct_researcher_forwards_instructions():
    chat = FakeChat()
    sci = Scientist(name="Scientist", chat=chat)
    res = DummyResearcher(name="Researcher", chat=chat)
    reply = sci.direct_researcher(res, "Do thing A")
    expected = base_agent_mod.compose_sections("", "", f"reply:{chat.prompts[1]}")
    assert reply == expected
    assert sci.history
    assert res.history
    assert chat.prompts[0][0]["content"] == SYSTEM_PROMPT


def test_execute_plan_invokes_tools_and_logs():
    chat = FakeChat()
    sci = Scientist(name="Scientist", chat=chat)
    res = LoggingResearcher(chat=chat)

    plan = """Step 1: search planets\nStep 2: scrape http://example.com\nStep 3: run script.py"""
    sci.execute_plan(plan, res)

    assert ("search", "planets") in res.calls
    assert ("scrape", "http://example.com") in res.calls
    assert ("run", "script.py") in res.calls
    assert sci.history

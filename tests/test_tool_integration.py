import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tools
from agents.base_agent import BaseAgent, compose_sections
import agents.base_agent as base_agent_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod

class DummyChat:
    def __init__(self, reply):
        self.reply = reply
    def __call__(self, messages):
        return types.SimpleNamespace(content=self.reply)

class DummyAgent(BaseAgent):
    pass


def test_json_tool_call(monkeypatch):
    class Dummy:
        def __call__(self, query=""):
            return "result"
    monkeypatch.setitem(tools.TOOL_CLASSES, "dummy", Dummy)
    reply = compose_sections("t", "c", '{"tool":"dummy","args":{"query":"x"}}')
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat(reply))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat(reply))
    agent = DummyAgent("test")
    output = agent.send_message("hi")
    assert "result" in output

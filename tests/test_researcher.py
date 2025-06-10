import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest

import agents.researcher as researcher_mod
import agents.base_agent as base_agent_mod

class DummyReply:
    def __init__(self, content):
        self.content = content
        self.anchor = ""

class DummyChat:
    def __init__(self, *args, **kwargs):
        self.called_with = None
    def __call__(self, messages):
        self.called_with = messages
        if isinstance(messages, list):
            text = messages[-1]["content"]
        else:
            text = messages
        return DummyReply("ack:" + text)

def test_send_message_history(monkeypatch):
    dummy = DummyChat()
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: dummy)
    simple = DummyChat()
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: simple)
    def patched_send(self, m):
        reply = researcher_mod.Researcher.chat(self, m)
        formatted = base_agent_mod.compose_sections("", "", reply)
        self.history.append(m)
        self.history.append(formatted)
        return formatted
    monkeypatch.setattr(researcher_mod.Researcher, "send_message", patched_send)
    r = researcher_mod.Researcher(model="test-model")
    result = r.send_message("hello")
    expected = base_agent_mod.compose_sections("", "", "ack:hello")
    assert result == expected
    assert r.history[-2:] == ["hello", expected]
    assert dummy.called_with[0]["role"] == "system"

def test_env_model(monkeypatch):
    captured = {}
    def fake_chat(model=None):
        captured["model"] = model
        return DummyChat()
    monkeypatch.setattr(researcher_mod, "TSCEChat", fake_chat)
    monkeypatch.setattr(base_agent_mod, "TSCEChat", fake_chat)
    def patched_send(self, m):
        reply = researcher_mod.Researcher.chat(self, m)
        formatted = base_agent_mod.compose_sections("", "", reply)
        self.history.append(m)
        self.history.append(formatted)
        return formatted
    monkeypatch.setattr(researcher_mod.Researcher, "send_message", patched_send)
    monkeypatch.setenv("MODEL_NAME", "env-model")
    r = researcher_mod.Researcher()
    assert captured["model"] == "env-model"


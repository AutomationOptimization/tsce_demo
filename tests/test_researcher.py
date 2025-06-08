import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest

import agents.researcher as researcher_mod

class DummyReply:
    def __init__(self, content):
        self.content = content
        self.anchor = ""

class DummyChat:
    def __init__(self, *args, **kwargs):
        self.called_with = None
    def __call__(self, messages):
        self.called_with = messages
        return DummyReply("ack:" + messages[-1]["content"])

def test_send_message_history(monkeypatch):
    dummy = DummyChat()
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: dummy)
    r = researcher_mod.Researcher(model="test-model")
    result = r.send_message("hello")
    assert result == "ack:hello"
    assert r.history[-2:] == ["hello", "ack:hello"]
    assert dummy.called_with[0]["role"] == "system"

def test_env_model(monkeypatch):
    captured = {}
    def fake_chat(model=None):
        captured["model"] = model
        return DummyChat()
    monkeypatch.setattr(researcher_mod, "TSCEChat", fake_chat)
    monkeypatch.setenv("MODEL_NAME", "env-model")
    r = researcher_mod.Researcher()
    assert captured["model"] == "env-model"


import types
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault(
    "openai",
    types.SimpleNamespace(OpenAI=object, BaseClient=object, AzureOpenAI=object),
)

import tsce_agent_demo.tsce_chat as tsce_chat
import agents.base_agent as base_agent
import agents.orchestrator as orchestrator
import agents.researcher as researcher

class DummyChat:
    def __call__(self, messages):
        if isinstance(messages, list):
            content = messages[-1]["content"]
        else:
            content = messages
        return types.SimpleNamespace(content=content)

def _install(monkeypatch, dummy):
    monkeypatch.setattr(tsce_chat, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat, "TSCEChat", lambda model=None: dummy)
    monkeypatch.setattr(base_agent, "TSCEChat", lambda model=None: dummy)
    monkeypatch.setattr(orchestrator, "TSCEChat", lambda model=None: dummy)
    monkeypatch.setattr(researcher, "TSCEChat", lambda model=None: dummy)

@pytest.fixture
def mock_tsce_chat(monkeypatch):
    dummy = DummyChat()
    _install(monkeypatch, dummy)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_KEY", "test-key")
    return dummy


@pytest.fixture(autouse=True)
def set_dummy_key(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY", "test-key")


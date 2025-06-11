import types

import agents.orchestrator as orchestrator_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod
import agents.base_agent as base_agent_mod


class DummyChat:
    def __call__(self, messages):
        if isinstance(messages, list):
            content = messages[-1]["content"]
        else:
            content = messages
        return types.SimpleNamespace(content=content)


def _patch(monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())


def test_run_legacy_round_robin(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    history = orch.run_legacy()
    roles = [m["role"] for m in history]
    assert roles == ["leader", "planner", "scientist", "leader"]

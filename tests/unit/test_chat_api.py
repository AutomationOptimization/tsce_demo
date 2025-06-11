import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Provide a minimal 'openai' module so tsce_chat imports succeed
sys.modules.setdefault(
    "openai",
    types.SimpleNamespace(BaseClient=object, AzureOpenAI=object),
)

import tsce_agent_demo.tsce_chat as chat_mod

class DummyClient:
    def __init__(self):
        self.called = False
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )
    def create(self, **kwargs):
        self.called = True
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))],
            model="dummy",
            model_dump=lambda: {
                "choices": [{"message": {"content": "ok"}}],
                "model": "dummy",
                "usage": {}
            },
        )

def test_completion_uses_chat_completions():
    client = DummyClient()
    chat = chat_mod.TSCEChat(client=client)
    chat._completion([{"role": "user", "content": "hi"}])
    assert client.called

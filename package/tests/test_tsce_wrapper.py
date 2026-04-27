from __future__ import annotations

import types

import pytest

from tsce import TSCEChat, TSCEClient, TSCEReply


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class FakeCompletions:
    def __init__(self, contents=None):
        self.contents = contents or [
            "<HDA>nuvor selki torven dalmi</HDA>",
            "final answer",
        ]
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        usage = {
            "prompt_tokens": index + 2,
            "completion_tokens": index + 3,
            "total_tokens": (index + 2) + (index + 3),
        }
        return FakeResponse(
            {
                "model": kwargs.get("model"),
                "choices": [
                    {
                        "message": {"content": self.contents[index]},
                        "finish_reason": "stop",
                        "logprobs": {"content": []},
                    }
                ],
                "usage": usage,
            }
        )


class FakeOpenAIClient:
    def __init__(self, contents=None):
        self.completions = FakeCompletions(contents)
        self.chat = types.SimpleNamespace(completions=self.completions)


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, *, model, messages, stream, options):
        self.calls.append(
            {"model": model, "messages": messages, "stream": stream, "options": options}
        )
        content = "<HDA>ollam naker veldin</HDA>" if len(self.calls) == 1 else "ollama final"
        return {
            "model": model,
            "message": {"content": content},
            "prompt_eval_count": 4,
            "eval_count": 5,
            "done_reason": "stop",
        }


def test_string_prompt_input_works():
    fake = FakeOpenAIClient()
    reply = TSCEChat(client=fake, backend="openai")("Say hello.")

    assert isinstance(reply, TSCEReply)
    assert reply.content == "final answer"
    assert reply.anchor == "<HDA>nuvor selki torven dalmi</HDA>"
    assert len(fake.completions.calls) == 2
    assert fake.completions.calls[0]["messages"][1] == {"role": "user", "content": "Say hello."}


def test_openai_style_message_input_works():
    fake = FakeOpenAIClient()
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Explain rollback."},
    ]

    reply = TSCEChat(client=fake, backend="openai")(messages)

    assert reply.content == "final answer"
    anchor_messages = fake.completions.calls[0]["messages"]
    final_messages = fake.completions.calls[1]["messages"]
    assert anchor_messages[1] == messages[0]
    assert anchor_messages[2] == messages[1]
    assert final_messages[1:] == messages


def test_force_anchor_bypasses_anchor_generation():
    fake = FakeOpenAIClient(contents=["forced final"])
    reply = TSCEChat(client=fake, backend="openai")(
        "Use the provided anchor.",
        force_anchor="<HDA>external policy</HDA>",
    )

    assert reply.content == "forced final"
    assert reply.anchor == "<HDA>external policy</HDA>"
    assert reply.anchor_model == "external"
    assert reply.anchor_request["backend"] == "external"
    assert len(fake.completions.calls) == 1


def test_phase_metadata_is_attached():
    fake = FakeOpenAIClient()
    reply = TSCEChat(client=fake, backend="openai")("Report metadata.")

    assert reply.latency >= 0
    assert set(reply.latency_by_phase) == {"anchor", "final", "total"}
    assert reply.anchor_request["phase"] == "anchor"
    assert reply.final_request["phase"] == "final"
    assert reply.raw_anchor["choices"][0]["message"]["content"].startswith("<HDA>")
    assert reply.raw_final["choices"][0]["message"]["content"] == "final answer"
    assert reply.usage_by_phase["anchor"]["total_tokens"] == 5
    assert reply.usage_by_phase["final"]["total_tokens"] == 7
    assert reply.usage["total"]["total_tokens"] == 12


def test_tsce_client_returns_openai_shaped_output():
    fake = FakeOpenAIClient()
    client = TSCEClient(client=fake, backend="openai")

    response = client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "Use the adapter."}],
        temperature=0.4,
        top_p=0.8,
        metadata={"trace_id": "abc"},
    )

    assert response["choices"][0]["message"]["content"] == "final answer"
    assert response.choices[0].message.content == "final answer"
    assert response.model == "gpt-test"
    assert response.usage["total_tokens"] == 12
    assert response.tsce.anchor.startswith("<HDA>")
    assert response.tsce.metadata.trace_id == "abc"
    assert fake.completions.calls[1]["temperature"] == 0.4
    assert fake.completions.calls[1]["top_p"] == 0.8


def test_openai_backend_branch_with_fake_client():
    fake = FakeOpenAIClient()
    reply = TSCEChat(client=fake, backend="openai", model="gpt-final")("OpenAI branch.")

    assert reply.final_request["backend"] == "openai"
    assert fake.completions.calls[1]["model"] == "gpt-final"


def test_azure_backend_branch_with_fake_client():
    fake = FakeOpenAIClient()
    reply = TSCEChat(client=fake, backend="azure", deployment_id="azure-deploy")("Azure branch.")

    assert reply.final_request["backend"] == "azure"
    assert fake.completions.calls[0]["model"] == "azure-deploy"
    assert fake.completions.calls[1]["model"] == "azure-deploy"


def test_ollama_backend_branch_with_fake_client():
    fake = FakeOllamaClient()
    reply = TSCEChat(client=fake, backend="ollama", model="llama-test")("Ollama branch.")

    assert reply.content == "ollama final"
    assert reply.final_request["backend"] == "ollama"
    assert fake.calls[0]["model"] == "llama-test"
    assert fake.calls[1]["options"]["temperature"] == 0.01


def test_local_phase2_backend_can_be_exercised_with_fake(monkeypatch):
    fake = FakeOpenAIClient(contents=["<HDA>local anchor</HDA>"])
    chat = TSCEChat(client=fake, backend="openai", phase2_backend="phi3")

    def fake_phi3(self, messages, *, temperature, top_p, max_tokens):
        response = {
            "model": "local-phi3",
            "choices": [{"message": {"content": "local final"}, "finish_reason": "stop"}],
            "usage": {},
        }
        self._record_phase(
            "final",
            {
                "backend": "phi3-local",
                "phase": "final",
                "request": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
            },
            response,
        )
        return response

    monkeypatch.setattr(chat, "_completion_phi3", types.MethodType(fake_phi3, chat))
    reply = chat("Local branch.")

    assert reply.content == "local final"
    assert reply.final_model == "local-phi3"
    assert reply.final_request["backend"] == "phi3-local"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"role": "user"}],
        [{"role": "user", "content": None}],
        [{"role": "invalid", "content": "x"}],
        42,
    ],
)
def test_malformed_messages_raise_value_error(payload):
    with pytest.raises(ValueError):
        TSCEChat(client=FakeOpenAIClient(), backend="openai")(payload)


def test_messages_must_include_user_turn():
    with pytest.raises(ValueError, match="user"):
        TSCEChat(client=FakeOpenAIClient(), backend="openai")(
            [{"role": "system", "content": "No user."}]
        )

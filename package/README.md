# TSCE

TSCE (Two-Step Contextual Enrichment) is an auditable wrapper for testing anchor-first LLM calls. Phase 1 generates or accepts a hidden control anchor. Phase 2 sends the user request with that anchor as internal context and returns the final answer plus phase-level metadata.

The public package name is `tsce`; this repository also keeps `tsce_agent_demo.tsce_chat` available for paper reproducibility.

## Install

From PyPI:

```bash
pip install tsce
```

From this checkout:

```bash
python -m pip install -e .
```

## Environment

OpenAI:

```bash
export OPENAI_API_KEY="..."
```

Azure OpenAI:

```bash
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_KEY="..."
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"
export AZURE_OPENAI_DEPLOYMENT="your-deployment"
```

Ollama:

```bash
export OLLAMA_MODEL="llama3"
export OLLAMA_BASE_URL="http://localhost:11434"
```

Local Phi-3 phase-2 backend:

```bash
export TSCE_PHASE2_BACKEND="phi3"
export TSCE_PHI3_MODEL="mlx-community/Phi-3-mini-4k-instruct-4bit"
```

## 10-Line TSCEChat Example

```python
from tsce import TSCEChat

chat = TSCEChat()
reply = chat("Write a two-sentence incident response checklist.")

print(reply.content)
print(reply.anchor)
print(reply.anchor_model, reply.final_model)
print(reply.latency)
print(reply.usage_by_phase)
```

`TSCEChat` also accepts OpenAI-style messages:

```python
reply = TSCEChat()([
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain rollback risk."},
])
```

## External Anchors

Use `force_anchor` when the anchor comes from a diffusion model, RL policy, rules engine, or other external controller. This fully bypasses Phase 1 and records the anchor as external metadata.

```python
from tsce import TSCEChat

policy_anchor = "<HDA>policy vazen torqel minra</HDA>"
reply = TSCEChat()(
    "Summarize the deployment decision.",
    force_anchor=policy_anchor,
)

print(reply.content)
print(reply.anchor_model)  # external
```

## Drop-In Client Adapter

`TSCEClient` exposes `client.chat.completions.create(...)` for code that expects an OpenAI-shaped response.

```python
from tsce import TSCEClient

client = TSCEClient()
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Give me a release checklist."}],
    temperature=0.1,
    top_p=1.0,
    metadata={"trace_id": "demo"},
)

print(response.choices[0].message.content)
print(response["tsce"]["anchor"])
```

The returned object supports both dict-style and attribute-style access. The top-level shape includes `choices`, `usage`, `model`, and a `tsce` metadata block with anchor, raw phase responses, requests, usage, and latency.

## Examples

```bash
python examples/quickstart_tsce.py --dry-run
python examples/ab_test_demo.py --dry-run
python examples/external_anchor_policy.py --dry-run
```

`--dry-run` uses deterministic fake clients and makes no network calls.

## Backend Notes

- OpenAI and Azure use the official `openai` Python client.
- Azure can use `AZURE_OPENAI_DEPLOYMENT`, plus numbered `_2` and `_3` variants for simple round-robin deployment pools.
- Ollama is enabled by `OLLAMA_MODEL` or `OLLAMA_BASE_URL` and maps OpenAI-style generation options to Ollama options.
- The local Phi-3 path applies only to Phase 2; Phase 1 still needs a normal TSCE anchor unless `force_anchor` is provided.

## Tests

The wrapper tests use fake clients and do not call real model APIs:

```bash
pytest tests/test_tsce_chat_smoke.py
pytest tests/test_tsce_wrapper.py
python examples/quickstart_tsce.py --dry-run
```

## Scope

TSCE is a framework and wrapper for testing anchor-conditioned decoding. It does not claim guaranteed universal quality improvement. Treat anchors and policies as experiment artifacts, inspect the metadata, and evaluate against your own reliability criteria.

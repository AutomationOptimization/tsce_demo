## TSCE Demo 🧠⚡
Why TSCE? In many real-world tasks, LLMs either hallucinate or lose track of complex instructions when forced to answer in one shot. Two-Step Contextual Enrichment solves this by first producing an "Embedding Space Control Prompt", then guiding a second, focused generation—delivering more faithful answers with no extra training.
*A two-phase **mechanistic framework** for more reliable LLM answers — validated on OpenAI GPT-3.5/4 and open-weights Llama-3 8 B*

---

### Table of Contents
1. [What is TSCE?](#what-is-tsce)
2. [Repo Highlights](#repo-highlights)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Usage Examples](#usage-examples)
8. [How TSCE Works](#how-tsce-works)
9. [Benchmarks & Latest Results](#benchmarks--latest-results)
10. [Troubleshooting](#troubleshooting)
11. [Extending the Demo](#extending-the-demo)
12. [Contributing](#contributing)
13. [License](#license)

---

### What is TSCE? <a name="what-is-tsce"></a>
Intuition: Imagine you ask a model, “Summarize this 1,000-word legal brief.” In a single pass it might drop key clauses or veer off into a hallucination because it's sampling from a wide distribution of possible vectors. Instead, TSCE’s first pass compresses the potential vector space with an "Embedding Space Control Prompt", and then second pass is better primed to generate the summary.


| Phase | Purpose | Temp | Output |
|-------|---------|------|--------|
| **1 — Embedding Space Control Prompt** | Compresses the entire prompt into a dense latent scaffold (ESCP). | ↑ ≈ 1.0 | opaque token block |
| **2 — Focused Generation** | Re-reads *System + User + ESCP* and answers inside a narrower semantic manifold. | ↓ ≤ 0.1 | final answer |

**Outcome:** fewer hallucinations, instruction slips, and formatting errors — with no fine-tuning and only one extra call.

---

### Repo Highlights <a name="repo-highlights"></a>

| File | Purpose |
|------|---------|
| `tsce_agent_demo/` | Harness & task sets that produced the results below. |
| `tsce_agent_demo/tsce_agent_test.py` | Baseline vs TSCE, prints both answers, writes `report.json`. |
| `tsce_agent_demo/tsce_chat.py` | Main TSCE wheel |
| `tsce_agent_demo/results/` | Entropy, KL, cosine-violin plots ready to share. |
| `.env.example` | Copy → `.env`, add your keys. |
| `prompts/phase1.txt`, `prompts/phase2.txt` | Default templates for each phase |

Works with **OpenAI Cloud**, **Azure OpenAI**, or any **Ollama / vLLM** endpoint.
✨ New: we now load the Phase 1 and Phase 2 prompts from prompts/phase1.txt and prompts/phase2.txt, making it easy to swap in your own prompt templates.

### How TSCE Works <a name="how-tsce-works"></a>

1. **Phase 1 – Embedding Space Control Prompt (ESCP) Construction:** compresses embedding space and generates an embedding space control prompt based on the user's input.
2. **Phase 2 – Guided Answering:** reads the control prompt with your original prompt to craft the final response.

#### Trade-off Considerations
Compressing natural language always risks dropping nuance, but our benchmarks show that on multi-step reasoning tasks TSCE still gains +30 pp on GPT-3.5 and yields 76 % success on Llama-3 vs. 69 % baseline—so the escp’s focus outweighs the compression loss.

---

### Benchmarks & Latest Results <a name="benchmarks--latest-results"></a>

| Model | Task Suite | One-Shot | **TSCE** | Token × |
|------|-------------|---------|-------|--------|
| GPT-3.5-turbo | math ∙ calendar ∙ format | 49 % | 79 % | 1.9× |
| GPT-4.1 | em-dash & policy tests | 50 % viol. | 6 % viol. | 2.0× |
| Llama-3 8B | mixed reasoning pack | 69 % | 76 % | 1.4× |

> *ESCP alone lifts GPT-3.5 by +30 pp; on the smaller Llama, the embedding space control prompt unlocks CoT (+16 pp).*

Note: TSCE uses two passes, so raw joules/token cost ≈2× single-shot; we compare against a zero-temp, single-shot oracle.
**Key plots** (see `figures/`):  
* `entropy_bar.png` — 6× entropy collapse  
* `kl_per_position.png` — KL > 10 nats after token 20  
* `cosine_violin.png` — answers cluster tighter with an embedding space control prompt

---

### Prerequisites <a name="prerequisites"></a>

* Python 3.8 +
* OpenAI API key **or** Azure OpenAI deployment key
* Git *(virtualenv optional)*

---

### Installation <a name="installation"></a>

```bash
git clone https://github.com/<your-username>/tsce_demo.git
cd tsce_demo
cd tsce_agent_demo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your creds
---

### Configuration <a name="configuration"></a>

#### OpenAI Cloud

```env
OPENAI_API_KEY=sk-********************************
# optional
OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
MODEL_NAME=gpt-3.5-turbo
```

#### Azure OpenAI

```env
OPENAI_API_TYPE=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o           # your deployment name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_KEY=<azure-key>             # or reuse OPENAI_API_KEY
```

*Leave unused keys blank.*

### HuggingFace Spaces secrets

Create a `.streamlit/secrets.toml` file so Streamlit can read your API keys when
deployed on a HuggingFace Space:

```toml
OPENAI_API_KEY = "${OPENAI_API_KEY}"
AZURE_OPENAI_ENDPOINT = "${AZURE_OPENAI_ENDPOINT}"
AZURE_OPENAI_DEPLOYMENT = "${AZURE_OPENAI_DEPLOYMENT}"
AZURE_OPENAI_API_VERSION = "${AZURE_OPENAI_API_VERSION}"
AZURE_OPENAI_KEY = "${AZURE_OPENAI_KEY}"
```

Add the corresponding secrets in the Space **Settings → Secrets** tab. They will
be exposed as environment variables for the app at runtime.

---

### Quick Start <a name="quick-start"></a>

```bash
python tsce_agent_test.py
```

Sample output:

```
==========
>>> ENTERING EMBEDDING ANALYTICS
>>> EMBEDDINGS DONE, choosing scatter solver
>>> running t-SNE with {'method': 'barnes_hut', 'perplexity': 30, 'n_iter': 1000}
...
```

For an interactive UI that lets you compare the baseline and TSCE answers, run:

```bash
streamlit run streamlit_chat.py
```
---

### Troubleshooting <a name="troubleshooting"></a>

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` | Wrong or expired key; ensure the key matches the endpoint type. |
| Hangs > 2 min | Slow model; tweak `timeout` in `_chat()` or lower temperature. |
| `ValueError: model not found` | Set `MODEL_NAME` (OpenAI) **or** `AZURE_OPENAI_DEPLOYMENT` (Azure) correctly. |

---

### Extending the Demo <a name="extending-the-demo"></a>

* **Batch runner** — loop over a prompt list, save aggregate CSV.  
* **Visualization** — embed t‑SNE plot code from the white‑paper (convex hulls, arrows).  
* **Guard‑rails** — add a self‑critique third pass for high‑risk domains.  
* **Streamlit UI** — drop‑in interactive playground (ask → escp → answer).  

**Open Questions & Next Steps**
- Recursive ESCP? Does running Phase 1 on its own escp improve or compound errors?
- Automated Prompt Tuning: Explore integrating dspy for auto-optimizing your prompt templates.
- Benchmark Strategy: We welcome new task sets—suggest yours under benchmark/tasks/.

Pull requests welcome!

---

### Contributing <a name="contributing"></a>

1. Fork the repo  
2. `git checkout -b feature/my-feature`  
3. Commit & push  
4. Open a PR

Please keep new code under MIT license and add a line to `README.md` if you extend functionality.

---

### License <a name="license"></a>

This project is licensed under the **MIT License** — free for commercial or private use.  See [LICENSE](./LICENSE) for full text.

---

*Happy controlling!* 🚀
## TSCE Demo 🧠⚡
Why TSCE? In many real-world tasks, LLMs either hallucinate or lose track of complex instructions when forced to answer in one shot. Two-Step Contextual Enrichment solves this by first producing an "Embedding Space Control Prompt", then guiding a second, focused generation—delivering more faithful answers with no extra training.
*A two-phase **mechanistic framework** for more reliable LLM answers — validated on OpenAI GPT-3.5/4 and open-weights Llama-3 8 B*

---

### Table of Contents
1. [What is TSCE?](#what-is-tsce)
2. [Repo Highlights](#repo-highlights)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Usage Examples](#usage-examples)
8. [How TSCE Works](#how-tsce-works)
9. [Benchmarks & Latest Results](#benchmarks--latest-results)
10. [Troubleshooting](#troubleshooting)
11. [Extending the Demo](#extending-the-demo)
12. [Contributing](#contributing)
13. [License](#license)

---

### What is TSCE? <a name="what-is-tsce"></a>
Intuition: Imagine you ask a model, “Summarize this 1,000-word legal brief.” In a single pass it might drop key clauses or veer off into a hallucination because it's sampling from a wide distribution of possible vectors. Instead, TSCE’s first pass compresses the potential vector space with an "Embedding Space Control Prompt", and then second pass is better primed to generate the summary.


| Phase | Purpose | Temp | Output |
|-------|---------|------|--------|
| **1 — Embedding Space Control Prompt** | Compresses the entire prompt into a dense latent scaffold (ESCP). | ↑ ≈ 1.0 | opaque token block |
| **2 — Focused Generation** | Re-reads *System + User + ESCP* and answers inside a narrower semantic manifold. | ↓ ≤ 0.1 | final answer |

**Outcome:** fewer hallucinations, instruction slips, and formatting errors — with no fine-tuning and only one extra call.

---

### Repo Highlights <a name="repo-highlights"></a>

| File | Purpose |
|------|---------|
| `tsce_agent_demo/` | Harness & task sets that produced the results below. |
| `tsce_agent_demo/tsce_agent_test.py` | Baseline vs TSCE, prints both answers, writes `report.json`. |
| `tsce_agent_demo/tsce_chat.py` | Main TSCE wheel |
| `tsce_agent_demo/results/` | Entropy, KL, cosine-violin plots ready to share. |
| `.env.example` | Copy → `.env`, add your keys. |
| `tsce_agent_demo/data/<prompts>.txt` | Default templates |
| `docs/tsce_DRAFT.pdf` | Current draft of the TSCE research paper |

Works with **OpenAI Cloud**, **Azure OpenAI**, or any **Ollama / vLLM** endpoint.
✨ New: we now load the Phase 1 and Phase 2 prompts from prompts/phase1.txt and prompts/phase2.txt, making it easy to swap in your own prompt templates.

### How TSCE Works <a name="how-tsce-works"></a>

1. **Phase 1 – Embedding Space Control Prompt (ESCP) Construction:** compresses embedding space and generates an embedding space control prompt based on the user's input.
2. **Phase 2 – Guided Answering:** reads the control prompt with your original prompt to craft the final response.

#### Trade-off Considerations
Compressing natural language always risks dropping nuance, but our benchmarks show that on multi-step reasoning tasks TSCE still gains +30 pp on GPT-3.5 and yields 76 % success on Llama-3 vs. 69 % baseline—so the escp’s focus outweighs the compression loss.

---

### Benchmarks & Latest Results <a name="benchmarks--latest-results"></a>

| Model | Task Suite | One-Shot | **TSCE** | Token × |
|------|-------------|---------|-------|--------|
| GPT-3.5-turbo | math ∙ calendar ∙ format | 49 % | 79 % | 1.9× |
| GPT-4.1 | em-dash & policy tests | 50 % viol. | 6 % viol. | 2.0× |
| Llama-3 8B | mixed reasoning pack | 69 % | 76 % | 1.4× |

> *ESCP alone lifts GPT-3.5 by +30 pp; on the smaller Llama, the embedding space control prompt unlocks CoT (+16 pp).*

Note: TSCE uses two passes, so raw joules/token cost ≈2× single-shot; we compare against a zero-temp, single-shot oracle.
**Key plots** (see `figures/`):  
* `entropy_bar.png` — 6× entropy collapse  
* `kl_per_position.png` — KL > 10 nats after token 20  
* `cosine_violin.png` — answers cluster tighter with an embedding space control prompt

---

### Prerequisites <a name="prerequisites"></a>

* Python 3.8 +
* OpenAI API key **or** Azure OpenAI deployment key
* Git *(virtualenv optional)*

---

### Installation <a name="installation"></a>

```bash
git clone https://github.com/<your-username>/tsce_demo.git
cd tsce_demo
cd tsce_agent_demo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your creds
---

### Configuration <a name="configuration"></a>

#### OpenAI Cloud

```env
OPENAI_API_KEY=sk-********************************
# optional
OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
MODEL_NAME=gpt-3.5-turbo
```

#### Azure OpenAI

```env
OPENAI_API_TYPE=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o           # your deployment name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_KEY=<azure-key>             # or reuse OPENAI_API_KEY
```

*Leave unused keys blank.*

---

### Quick Start <a name="quick-start"></a>

```bash
python tsce_agent_test.py
```

Sample output:

```
==========
>>> ENTERING EMBEDDING ANALYTICS
>>> EMBEDDINGS DONE, choosing scatter solver
>>> running t-SNE with {'method': 'barnes_hut', 'perplexity': 30, 'n_iter': 1000}
...
```

For an interactive UI that lets you compare the baseline and TSCE answers, run:

```bash
streamlit run streamlit_chat.py
```
---

### Troubleshooting <a name="troubleshooting"></a>

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` | Wrong or expired key; ensure the key matches the endpoint type. |
| Hangs > 2 min | Slow model; tweak `timeout` in `_chat()` or lower temperature. |
| `ValueError: model not found` | Set `MODEL_NAME` (OpenAI) **or** `AZURE_OPENAI_DEPLOYMENT` (Azure) correctly. |

---

### Extending the Demo <a name="extending-the-demo"></a>

* **Batch runner** — loop over a prompt list, save aggregate CSV.  
* **Visualization** — embed t‑SNE plot code from the white‑paper (convex hulls, arrows).  
* **Guard‑rails** — add a self‑critique third pass for high‑risk domains.  
* **Streamlit UI** — drop‑in interactive playground (ask → escp → answer).  

**Open Questions & Next Steps**
- Recursive ESCP? Does running Phase 1 on its own escp improve or compound errors?
- Automated Prompt Tuning: Explore integrating dspy for auto-optimizing your prompt templates.
- Benchmark Strategy: We welcome new task sets—suggest yours under benchmark/tasks/.

Pull requests welcome!

---

### Contributing <a name="contributing"></a>

1. Fork the repo  
2. `git checkout -b feature/my-feature`  
3. Commit & push  
4. Open a PR

Please keep new code under MIT license and add a line to `README.md` if you extend functionality.

---

### License <a name="license"></a>

This project is licensed under the **MIT License** — free for commercial or private use.  See [LICENSE](./LICENSE) for full text.

---

*Happy controlling!* 🚀

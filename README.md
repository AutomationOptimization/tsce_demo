## TSCE Demo 🧠⚡
Why TSCE? In many real-world tasks, LLMs either hallucinate or lose track of complex instructions when forced to answer in one shot. Two-Step Contextual Enrichment solves this by first compressing your entire prompt into a "hyperdimensional anchor," then guiding a second, focused generation—delivering more faithful answers with no extra training.
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
Intuition: Imagine you ask a model, “Summarize this 1,000-word legal brief.” In a single pass it might drop key clauses. Instead, TSCE’s first pass distills it into a tiny latent scaffold (“anchor”), and the second pass expands that into a summary—so you get both brevity and completeness.


| Phase | Purpose | Temp | Output |
|-------|---------|------|--------|
| **1 — Hyper-Dimensional Anchor** | Compresses the entire prompt into a dense latent scaffold (HDA). | ↑ ≈ 1.3 | opaque token block |
| **2 — Focused Generation** | Re-reads *System + User + HDA* and answers inside a narrower semantic manifold. | ↓ ≤ 0.7 | final answer |

**Outcome:** fewer hallucinations, instruction slips, and formatting errors — with no fine-tuning and only one extra call.

---

### Repo Highlights <a name="repo-highlights"></a>

| File | Purpose |
|------|---------|
| `tsce_demo.py` | Baseline vs TSCE, prints both answers, writes `report.json`. |
| `tsce_core.py` | 120-LoC reference implementation (backend-agnostic). |
| `benchmark/` | Harness & task sets that produced the results below. |
| `figures/` | Entropy, KL, cosine-violin plots ready to share. |
| `.env.example` | Copy → `.env`, add your keys. |
| `prompts/phase1.txt`, `prompts/phase2.txt` | Default templates for each phase |

Works with **OpenAI Cloud**, **Azure OpenAI**, or any **Ollama / vLLM** endpoint.
✨ New: we now load the Phase 1 and Phase 2 prompts from prompts/phase1.txt and prompts/phase2.txt, making it easy to swap in your own prompt templates.

### How TSCE Works <a name="how-tsce-works"></a>

1. **Phase 1 – Anchor Construction:** compresses the entire prompt into an opaque anchor.
2. **Phase 2 – Guided Answering:** reads the anchor with your original prompt to craft the final response.

#### Trade-off Considerations
Compressing natural language always risks dropping nuance, but our benchmarks show that on multi-step reasoning tasks TSCE still gains +30 pp on GPT-3.5 and yields 76 % success on Llama-3 vs. 69 % baseline—so the anchor’s focus outweighs the compression loss.

---

### Benchmarks & Latest Results <a name="benchmarks--latest-results"></a>

| Model | Task Suite | One-Shot | **TSCE** | Token × |
|------|-------------|---------|-------|--------|
| GPT-3.5-turbo | math ∙ calendar ∙ format | 49 % | 79 % | 1.9× |
| GPT-4.1 | em-dash & policy tests | 50 % viol. | 6 % viol. | 2.0× |
| Llama-3 8B | mixed reasoning pack | 69 % | 76 % | 1.4× |

> *Anchor alone lifts GPT-3.5 by +30 pp; on the smaller Llama, the anchor unlocks CoT (+16 pp).*

Note: TSCE uses two passes, so raw joules/token cost ≈2× single-shot; we compare against a zero-temp, single-shot oracle.
**Key plots** (see `figures/`):  
* `entropy_bar.png` — 6× entropy collapse  
* `kl_per_position.png` — KL > 10 nats after token 20  
* `cosine_violin.png` — answers cluster tighter with an anchor

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
python tsce_demo.py "How many r's are in strrawberry?"
```

Sample output:

```
Prompt : How many r's are in strrawberry?

Baseline answer
---------------- There are 2 r's in the word…

TSCE answer
----------- The word "strrawberry" contains 4 r's.

Report saved → report.json
```

For an interactive UI that lets you compare the baseline and TSCE answers, run:

```bash
streamlit run streamlit_chat.py
```

### Quick Demo
```bash
python tsce_demo.py "What are the safety protocols for lithium-ion batteries?"
```
Sample output:

Anchor: ⟨HDA: 0x3f2a…⟩
TSCE Answer: Lithium-ion batteries require…

![Example output](figures/output.png)

---

### Usage Examples <a name="usage-examples"></a>

```bash
python tsce_demo.py "Rewrite this sentence without any em-dashes — can you?"
python tsce_demo.py "Generate a SQL query that counts users per country."
python tsce_demo.py "Explain quantum tunnelling to a 10‑year‑old in 3 bullet points."
```

`report.json` includes token counts and the hidden anchor for post‑hoc analysis.

---

### Troubleshooting <a name="troubleshooting"></a>

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` | Wrong or expired key; ensure the key matches the endpoint type. |
| Hangs > 2 min | Slow model; tweak `timeout` in `_chat()` or lower temperature. |
| `ValueError: model not found` | Set `MODEL_NAME` (OpenAI) **or** `AZURE_OPENAI_DEPLOYMENT` (Azure) correctly. |
| Anchor leaks to user | Verify Phase 2 system message starts with `Hyperdimensional anchor:\n` and low temp. |

---

### Extending the Demo <a name="extending-the-demo"></a>

* **Batch runner** — loop over a prompt list, save aggregate CSV.  
* **Visualization** — embed t‑SNE plot code from the white‑paper (convex hulls, arrows).  
* **Guard‑rails** — add a self‑critique third pass for high‑risk domains.  
* **Streamlit UI** — drop‑in interactive playground (ask → anchor → answer).  

**Open Questions & Next Steps**
- Recursive Anchors? Does running Phase 1 on its own anchor improve or compound errors?
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

*Happy anchoring!* 🚀

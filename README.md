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
7. [Orchestrator Demo](#orchestrator-demo)
8. [Pipeline Overview](#pipeline-overview)
9. [Usage Examples](#usage-examples)
10. [How TSCE Works](#how-tsce-works)
11. [Benchmarks & Latest Results](#benchmarks--latest-results)
12. [Troubleshooting](#troubleshooting)
13. [Extending the Demo](#extending-the-demo)
14. [Contributing](#contributing)
15. [License](#license)

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
| `results/` | Entropy, KL, cosine-violin plots ready to share. |
| `tsce_agent_demo/run_orchestrator.py` | Command-line interface for the pipeline |
| `tsce_agent_demo/inspect_tsce_layers.py` | Layer variance tool using transformer-lens |
| `tsce_agent_demo/tsce_heval_test.py` | Evaluate the HaluEval benchmark |
| `.env.example` | Copy → `.env`, add your keys. |
| `prompts/phase1.txt`, `prompts/phase2.txt` | Default templates for each phase |
| `agents/hypothesis.py` | Record agreed hypothesis and emit `TERMINATE` |

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
git clone https://github.com/<your-username>/tsce_agent_demo.git
cd tsce_agent_demo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env          # then edit .env with your creds
# If you need a ready-to-use environment, pull the sandbox image:
```bash
docker run --rm -it ghcr.io/<owner>/tsce_sandbox:latest
```
---

### Configuration <a name="configuration"></a>

#### OpenAI Cloud

```env
OPENAI_API_KEY=sk-********************************
# optional
OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
MODEL_NAME=gpt-3.5-turbo
LOG_DIR=logs  # optional agent conversation logs
```

#### Azure OpenAI

```env
OPENAI_API_TYPE=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o           # your deployment name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_KEY=<azure-key>             # or reuse OPENAI_API_KEY
```

*Leave unused keys blank.* Set ``LOG_DIR`` if you want to persist agent history logs.

---

### Quick Start <a name="quick-start"></a>

```bash
$ tsce-agent-demo "What is the best catalyst for ..."
{"task_id": "...", "status": "success", "summary_file": ".../summary.md"}
```

For an interactive UI that lets you compare the baseline and TSCE answers, run:

```bash
streamlit run streamlit_chat.py
```

### Orchestrator Demo <a name="orchestrator-demo"></a>

The orchestrator now routes chat messages through each agent. Instantiate it
with your goals and iterate over ``run()`` to inspect the conversation. The
Planner proposes a step-by-step plan, then the Leader tasks the Scientist to
carry it out:

```python
from agents import Orchestrator

goals = ["Print hello world", "TERMINATE"]
orc = Orchestrator(goals, log_dir="logs")
for msg in orc.run():
    print(msg["role"], msg["content"])
```
Logs are saved under ``logs/`` by default; set ``LOG_DIR`` in your ``.env`` to
change this.


Sample ``hello world`` session:

```
leader Print hello world
planner Use Python's ``print``
scientist Looks good
hypothesis TERMINATE
script_writer print("Hello, world!")
simulator Hello, world!
evaluator success
judge_panel approved
```

The run concludes only when the nine-member ``JudgePanel`` unanimously approves
the evaluator's summary.

### Pipeline Overview <a name="pipeline-overview"></a>

The orchestrator runs a queue of specialized agents in sequence:

1. **Leader** – pulls the next goal from the list.
2. **Planner** – drafts a step-by-step plan.
3. **Scientist** – executes the plan while consulting the Planner.
4. **Researcher** – gathers data from the web or filesystem.
5. **Hypothesis** – Scientist and Researcher agree on a written hypothesis. When
   ``TERMINATE`` is logged, planning stops and research begins.
6. **ScriptWriter** – generates executable Python code.
7. **ScriptQA** – optional lint/unit-test pass over the script.
8. **Simulator/Evaluator** – runs the script and summarizes the results.
9. **JudgePanel** – nine judges vote until the evaluation is unanimously
   approved.

See [docs/pipeline.md](docs/pipeline.md) for a deeper walk through of each
stage.

### CLI Usage

Run the orchestrator directly from the command line:

```bash
python -m tsce_agent_demo --question "Print hello world"
```

Additional utilities:

* ``tsce_agent_demo/inspect_tsce_layers.py`` – analyse layer-wise variance on local models.
* ``tsce_agent_demo/tsce_heval_test.py`` – run the HaluEval benchmark with TSCE.

### Built-in Tools

Agents may call helper utilities by returning a JSON object in the ``Speak`` section:

```json
{"tool": "google_search", "args": {"query": "python", "num_results": 2}}
```

Tool names:

* ``google_search`` – quick Google result titles
* ``web_scrape`` – fetch a web page and strip HTML
* ``create_file`` – create a new text file
* ``read_file`` – read a text file
* ``edit_file`` – overwrite an existing file
* ``delete_file`` – remove a file
* ``run_script`` – execute a Python script

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

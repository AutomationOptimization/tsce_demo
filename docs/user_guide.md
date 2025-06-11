# TSCE Demo User Guide

This page collects setup notes and helpful tips for running the demo agents.

## Installation

### Pip
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/tsce_demo.git
   cd tsce_demo
   ```
2. Create a virtual environment and install requirements
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

### Conda
1. Create and activate an environment
   ```bash
   conda create -n tsce python=3.10
   conda activate tsce
   ```
2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```

### Docker
A simple Dockerfile is provided. Build and run:
```bash
docker build -t tsce_demo .
docker run --rm -it tsce_demo
```
If you are on an ARM machine (e.g. Apple Silicon) see the FAQ below.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI Cloud |
| `OPENAI_ENDPOINT` | Override the chat completion URL |
| `MODEL_NAME` | Default model name when using OpenAI |
| `LOG_DIR` | Directory for conversation logs |
| `OPENAI_API_TYPE` | Set to `azure` for Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | Name of your deployment |
| `AZURE_OPENAI_API_VERSION` | API version string |
| `AZURE_OPENAI_KEY` | Key for Azure OpenAI (optional if using `OPENAI_API_KEY`) |
| `OLLAMA_MODEL` | Model name for an Ollama backend |
| `OLLAMA_BASE_URL` | Base URL for Ollama |
| `NCBI_API_KEY` | Enables PubMed literature search |

Copy `.env.example` to `.env` and populate any keys you plan to use.

## First Five‑Minute Run

1. Ensure your API credentials are set in `.env`.
2. Run the orchestrator with a question:
   ```bash
   python -m tsce_demo --question "How does TSCE reduce hallucinations?"
   ```
   The script walks through several phases and prints a JSON result when done.
3. Inspect the generated summary file reported in the output.
4. If a phase fails, a JSON error report is saved under `logs/` using the
   run ID, e.g. `logs/orchestrator_failure_<id>.json`.

## Troubleshooting FAQ

| Problem | Remedy |
|---------|-------|
| **Docker build fails on ARM** | Install `libboost-all-dev` and other build tools before the RDKit wheel is built. See `docs/docker.md` for more notes. |
| **`RDKit` wheel cannot find `GLIBCXX`** | Ensure your GCC/GLIBC packages are up to date. On Ubuntu, run `sudo apt-get install libstdc++6`. |
| **401 Unauthorized** | Wrong or expired API key; verify it matches the endpoint type. |
| **`ValueError: model not found`** | Set `MODEL_NAME` or `AZURE_OPENAI_DEPLOYMENT` correctly in your environment. |


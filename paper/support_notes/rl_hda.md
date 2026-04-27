RL: Train a Small Policy to Generate HDA

Goal
- Learn a lightweight policy that produces the phase‑1 “HDA” string.
- Use TSCE phase‑2 + existing evaluators as a pass/fail reward.

What’s included
- `tsce_agent_demo/tsce_chat.py`: now accepts `force_anchor` to skip phase‑1 and run only phase‑2 with a supplied HDA.
- `tsce_agent_demo/train_hda_rl.py`: minimal REINFORCE trainer with a char‑level LSTM policy that emits a 20×8‑char, space‑separated HDA.

Quick start
1) Ensure API env vars are set (choose one backend):
   - OpenAI: `OPENAI_API_KEY`
   - Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT`
   - Ollama: `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL`

2) Install Python deps (if needed):
   `pip install -r requirements.txt`

3) Run training (200 steps over mixed tasks):
   `python -m tsce_agent_demo.train_hda_rl --steps 200 --task-kind auto`

Artifacts
- Checkpoints: `models/hda_policy.pt`
- Logs: `results/hda_rl_log.jsonl`

Notes
- The policy and RL are intentionally small and simple to keep runtime light. You can increase steps, adjust entropy bonus, or widen the HDA shape later.
- The default evaluator pool is reused from the existing benchmark (math, calendar, formatting, schema, md2latex). Set `--task-kind math` for a focused curriculum.
- Costs: phase‑2 calls use your configured backend each step; start with small `--steps`.


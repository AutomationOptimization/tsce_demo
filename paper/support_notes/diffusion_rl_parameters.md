# Diffusion RL Parameter Reference
End-to-end knobs for training, evaluating, and serving the diffusion-based TSCE RL policy. Defaults are shown exactly as defined in code.

## Training CLI — `tsce_agent_demo/train_hda_rl.py`
Run with `python -m tsce_agent_demo.train_hda_rl ...`.

### Run + optimizer
| Flag | Default | Purpose |
| --- | --- | --- |
| `--seed` | `42` | Seeds Python/NumPy/Torch; honors `HARD_DETERMINISM` to force deterministic Torch kernels when set. |
| `--steps` | `200` | Training steps (one RL rollout + update each). |
| `--task-kind` | `auto` | Task curriculum: rotate across task kinds with per-kind success-rate weighting, or pin to `math`/`gsm8k`/`gsm_hard`/`schema`/`formatting`. |
| `--phase2-backend` | `default` | TSCE phase-2 backend for final answer generation (`default` or `phi3`). |
| `--temperature` | `0.9` | Base anchor sampling temperature (used as the baseline bandit arm). |
| `--max-chars` | `128` | Max character budget for anchors (`max_new_tokens` in sampler). |
| `--lr` | `1e-4` | AdamW learning rate. |
| `--weight-decay` | `0.0` | AdamW weight decay. |
| `--entropy` | `0.006` | Entropy bonus coefficient; multiplied by `1 + EXPLORE_BOOST * (1 - pass_rate)` to encourage exploration when success is low. |
| `--kl-coef` | `0.02` | KL weight added to the loss if the sampler returns `kl_sum`; otherwise ignored with a warning. |

### Reconstruction pretrain (Stage 0)
| Flag | Default | Purpose |
| --- | --- | --- |
| `--pretrain-recon` | `false` | Enable Stage-0 reconstruction (anchor → prompt) pretraining. |
| `--recon-steps` | `8000` | Planned number of Stage-0 steps (capped by `--recon-stage-max-steps`). |
| `--lambda-recon` | `0.0` | Weight on reconstruction loss/bonuses during Stage 0. |
| `--lambda-recon-aux` | `0.0` | Auxiliary recon weight (currently ignored after Stage 0). |
| `--lambda-overlap` | `0.0` | Penalty weight on anchor/prompt token overlap during recon. |
| `--lambda-len` | `0.0` | Penalty weight on anchor length (chars) during recon. |
| `--lambda-ent` | `0.0` | Entropy bonus weight for anchors during recon. |
| `--anchor-dropout` | `0.15` | Token dropout rate applied to anchors before recon decoding. |
| `--recon-priority-scale` | `2.0` | Multiplier on recon reward delta (before clamping). |
| `--recon-priority-cap` | `1.0` | Absolute clamp on scaled recon reward delta. |
| `--recon-stage-max-steps` | `2000` | Hard cap on Stage-0 steps; `0` disables the cap. |
| `--recon-gate-loss` | `0.6` | Gate: mean recon loss must be below this to unlock. |
| `--recon-gate-match` | `0.4` | Gate: mean recon token-match fraction must exceed this. |
| `--recon-gate-entropy` | `0.8` | Gate: mean anchor entropy must exceed this. |
| `--recon-gate-anchor-cos` | `0.75` | Gate: mean anchor/prompt cosine must stay below this. |
| `--recon-gate-window` | `400` | Window for recon gate statistics. |
| `--recon-gate-min-samples` | `100` | Minimum recon samples before gate checks. |
| `--recon-gate-patience` | `2000` | Steps to wait before timing out the gate. |
| `--recon-plateau-patience` | `500` | Steps to wait without recon loss improvement before treating as plateau. |
| `--recon-plateau-improve` | `0.1` | Required fractional improvement to avoid plateau. |
| `--recon-lock-match` | `0.2` | Optional lock: require mean recon match ≥ this to end Stage 0; otherwise exit reason is `lock_not_met`. |

### Sampling & bandit arms
| Flag | Default | Purpose |
| --- | --- | --- |
| `--base-top-k` | `50` | Top-k filter for the baseline sampling arm. |
| `--base-top-p` | `0.95` | Top-p filter for the baseline sampling arm. |
| `--base-cfg-scale` | `2.0` | Classifier-free guidance scale applied to the policy and the baseline arm. |
| `--auto-tune-sampling` | `false` | Enable multi-armed bandit tuning over temperature/top-k/top-p/cfg-scale combinations. |
| `--sample-temperatures` | `0.7,0.9,1.1` | Candidate temperatures when auto-tune is on. |
| `--sample-top-k` | `20,50,120` | Candidate top-k values when auto-tune is on. |
| `--sample-top-p` | `0.8,0.92,0.97` | Candidate top-p values when auto-tune is on. |
| `--sample-cfg-scale` | `1.0,2.0,3.0` | Candidate CFG scales when auto-tune is on. |
| `--sample-arm-limit` | `12` | Maximum number of distinct sampling arms to keep. |
| `--bandit-epsilon` | `0.1` | Epsilon-greedy exploration probability across arms. |
| `--bandit-beta` | `0.3` | EMA factor for bandit reward updates per arm. |
| `--bandit-ucb` | `0.5` | UCB bonus scale when picking arms (balances exploration with reward EMA). |

### Reward shaping & sampling strategy
| Flag | Default | Purpose |
| --- | --- | --- |
| `--w-task` | `1.0` | Scales base task utility and streak bonus inside `positive_reward`. |
| `--anchor-div-weight` | `0.45` | Weight for anchor diversity bonus; annealed with pass rate via `MAX_DIVERSITY_SCALE`. |
| `--answer-div-weight` | `0.1` | Weight for answer diversity; annealed down as pass rate rises (floor at 20% of base). |
| `--fid-weight` | `0.1` | Weight for chain-of-thought fidelity (capped to 0.5). |
| `--anchor-align-weight` | `0.12` | Bonus weight for cosine similarity to mean pass anchors. |
| `--repeat-penalty-weight` | `0.4` | Weight applied to token repetition penalty when enabled. |
| `--repeat-threshold` | `0.7` | Minimum unique-token ratio before repeat penalty starts. |
| `--anchor-len-penalty` | `0.0001` | Per-token length penalty added to the negative reward. |
| `--batch-k` | `1` | Number of anchors sampled per task; if `>1`, SCST-style selection/bonus is used. |
| `--select-top` | `true` | When `batch-k > 1`, pick the highest-reward sample; disable to keep the first sample. |
| `--anchor-repeat-penalty` | `false` | Toggle applying the repeat penalty to anchors. |

### Baselines, curriculum, and stopping
| Flag | Default | Purpose |
| --- | --- | --- |
| `--baseline-beta` | `0.85` | EMA factor for the per-kind advantage baseline (smaller = more reactive). |
| `--baseline-reset-interval` | `200` | Steps between halving baselines to prevent drift (`0` disables). |
| `--ma-window` | `35` | Window length for binary pass-rate tracking (drives annealing and early stop). |
| `--immediate-retry-burst` | `3` | Number of immediate retries before re-queuing a failed task (`0` disables burst retries). |
| `--success-rate-beta` | `0.9` | EMA smoothing for per-kind success rates used in auto curriculum. |
| `--success-rate-min-weight` | `0.05` | Minimum sampling weight per kind to avoid starving high-success tasks. |
| `--target-passes` | `18` | Target number of passes within the moving window when using `--until-target`. |
| `--until-target` | `false` | Stop once the moving window accumulates `target_passes`. |

### Embedding, logging, and checkpoints
| Flag | Default | Purpose |
| --- | --- | --- |
| `--embed-backend` | `e5` | Embedding backend for shaping/alignment (`hf`, `e5`, `hashed`, `universal`). |
| `--embed-model` | `intfloat/e5-small-v2` | Embedding model name for the selected backend. |
| `--history` | `32` | Rolling window length for anchor/answer vectors used in diversity and alignment shaping. |
| `--cond-backend` | `hashed` | Conditioner backend for prompt embeddings (`hashed` lexical or `hf` semantic). |
| `--cond-model` | `distilbert-base-uncased` | HF encoder name when `--cond-backend=hf`. |
| `--cond-dim` | `256` | Conditioning embedding dimension. |
| `--save-dir` | `models` | Subdir (under `out-dir` unless absolute) where checkpoints are written. |
| `--log` | `results/hda_rl_log.jsonl` | Log path (placed under `out-dir` unless absolute). |
| `--eval-every` | `0` | If `>0`, run periodic pass@1 eval every N steps. |
| `--eval-n` | `20` | Number of eval tasks per eval sweep. |
| `--eval-temp` | `0.8` | Anchor sampling temperature during eval when auto-tune is off; when auto-tune is on, uses the arm’s params. |
| `--out-dir` | `""` | Root output folder; defaults to `results/<timestamp>` when empty. |
| `--resume` | `""` | Path to checkpoint to resume; restores policy, optimizer, baselines, and pass window. |

### Training-time environment toggles
- `HARD_DETERMINISM`: truthy enables deterministic Torch algorithms and disables CuDNN autotune.
- `TSCE_PHASE2_RETRIES` / `TSCE_PHASE2_RETRY_BACKOFF`: retry count (default `3`) and backoff seconds (default `2.0`) for TSCE phase-2 calls during training.
- Thought-panel (optional reward bonus) requires Azure creds: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, and either `THOUGHT_PANEL_DEPLOYMENT` or `AZURE_OPENAI_DEPLOYMENT` (optional `AZURE_OPENAI_API_VERSION`, default `2024-12-01-preview`). Missing creds silently disable the panel.

## Evaluation CLI — `tsce_agent_demo/eval_diffusion_ckpt.py`
Compare a diffusion RL checkpoint against baseline TSCE.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--ckpt` | `""` | Path to `checkpoint_latest.pt` (auto-picks newest when empty). |
| `--n` | `100` | Number of tasks to evaluate. |
| `--task-kind` | `auto` | Task pool (`auto` rotates across math/gsm8k/gsm_hard/schema/formatting). |
| `--device` | `mps` | Device for diffusion policy (`cuda`/`mps`/`cpu`). |
| `--eval-temp` | `1.0` | Anchor sampling temperature. |
| `--cfg-scale` | `2.0` | Classifier-free guidance scale for sampling. |
| `--top-k` | `50` | Top-k filter. |
| `--top-p` | `0.95` | Top-p filter. |
| `--max-chars` | `600` | Character cap for anchors. |
| `--out-dir` | `""` | Output dir (defaults to `results/<timestamp>`). |
| `--tsce-deployment` | `""` | Optional Azure deployment/model override for TSCE phase-2. |

## Anchor verification — `tsce_agent_demo/verify_diffusion_anchor.py`
Local sanity-check of a checkpoint’s anchor generation.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--ckpt` | `""` | Checkpoint path (auto-picks latest in `models/` when empty). |
| `--device` | `cpu` | Device for the diffusion policy (`cpu`/`cuda`/`mps`). |
| `--n` | `10` | Number of anchors to generate. |
| `--temperature` | `0.9` | Sampling temperature. |
| `--cfg-scale` | `2.5` | Classifier-free guidance scale. |
| `--top-k` | `50` | Top-k filter. |
| `--top-p` | `0.95` | Top-p filter. |
| `--max-chars` | `200` | Character cap for anchors. |
| `--semantic` | `false` | Run optional semantic similarity analysis (requires `sentence-transformers`). |
| `--verbose` | `false` | Print per-anchor details. |
| `--output` | `""` | Optional JSON output path. |

## TerminalBench runner — `tsce_agent_demo/terminalbench_runner.py`
Evaluates the diffusion policy + TSCE on TerminalBench tasks.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--ckpt` | (required) | Diffusion checkpoint path. |
| `--out-dir` | `results/terminalbench_eval` | Where to write the report. |
| `--device` | `cuda` | Device for the diffusion model (`cuda`/`cpu`/`mps`). |
| `--eval-temp` | `0.7` | Sampling temperature. |
| `--cfg-scale` | `2.0` | CFG scale. |
| `--top-k` | `50` | Top-k filter. |
| `--top-p` | `0.95` | Top-p filter. |
| `--max-chars` | `200` | Anchor length cap. |
| `--task-set` | `terminalbench_default` | TerminalBench task set identifier. |
| `--tsce-deployment` | `""` | Optional Azure deployment/model override for TSCE phase-2. |

## Benchmark harness diffusion overrides — `tsce_agent_demo/tsce_agent_test.py`
These CLI flags set environment variables that enable the diffusion anchor policy inside the benchmark harness when a checkpoint is provided.

| CLI flag | Env it sets | Default | Purpose |
| --- | --- | --- | --- |
| `--diffusion-ckpt` | `DIFFUSION_CKPT` | `""` | Path to diffusion checkpoint; when set, anchors are drawn from the diffusion policy. |
| `--diffusion-temperature` / `--diff-temp` | `DIFF_TEMP` | `0.9` | Anchor sampling temperature. |
| `--diffusion-final-temp` / `--diff-final-temp` | `DIFF_FINAL_TEMP` | `0.01` | Final-phase temperature used when TSCE consumes the diffusion anchor. |
| `--diffusion-cfg` / `--diff-cfg` | `DIFF_CFG` | `2.5` | CFG scale for sampling. |
| `--diffusion-topk` / `--diff-topk` | `DIFF_TOPK` | `50` | Top-k filter. |
| `--diffusion-topp` / `--diff-topp` | `DIFF_TOPP` | `0.95` | Top-p filter. |
| `--diffusion-max-chars` / `--diff-max-chars` | `DIFF_MAX_CHARS` | `200` | Anchor character cap. |
| `--diffusion-device` / `--diff-device` | `DIFFUSION_DEVICE` | `cpu` | Device for loading the diffusion checkpoint. |

## Diffusion anchor API — `tsce_agent_demo/diffusion_api.py`
FastAPI wrapper to serve anchors.

- Environment:
  - `TSCE_DIFFUSION_CKPT`: checkpoint path (falls back to newest `checkpoint_latest.pt` under `results/*`).
  - `TSCE_DEVICE`: device string (`cuda`/`mps`/`cpu`, auto-selected when unset).
- Request body parameters:
  - `prompt` (required): text to condition the diffusion policy.
  - `temperature` (default `0.8`): sampling temperature.
  - `max_chars` (default `600`): character budget for the anchor.

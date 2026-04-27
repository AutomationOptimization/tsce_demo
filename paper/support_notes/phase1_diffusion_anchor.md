# TSCE phase-1 anchors, diffusion fit, and the RL training loop

This note stitches together the base TSCE theory from `docs/Think_Before_You_Speak.pdf`, the HDA/ESCP prompt shape in `tsce_agent_demo/tsce_chat.py`, and the diffusion-based RL trainer in `tsce_agent_demo/train_hda_rl.py`. It explains why the diffusion policy is a good fit for phase-1 anchors and how the training pipeline leverages it.

## TSCE in one page (ESCP / anchor theory)
- Two-step conditioning: Phase 1 samples an Embedding Space Control Prompt (ESCP, called HDA in code) from the same model: A ~ Pθ(A | X, S_escp). Phase 2 conditions on that anchor: Y ~ Pθ(Y | X, A, S_final). Because H(Y | X, A) ≤ H(Y | X), the anchor contracts the output manifold before the user-visible decode.
- Empirical effect (paper): +24–44 pp task lift vs single-pass on GPT-3.5/4 and Llama-3; convex-hull area in embedding space shrinks ~18%; word-level entropy drops ~0.4 bits (Phase 1) and stays ~0.33 bits lower after Phase 2; violation rate in “no em-dash” drops from ~50% to ~6%.
- Geometric view: baseline answers, anchors, and TSCE finals form three clusters; anchors collapse into a tight latent basin, finals re-expand but remain inside the baseline hull. ESCP tokens are opaque to humans but act as an attractor in the model’s hidden space.
- Inference-time, model-agnostic: the anchor is just a system-side string; no finetuning of the base LLM is required.

## Mechanistic update from the greenhouse probe
- The anchor itself is the intervention. The core paper claim does not require extracting a portable activation vector from the anchor and patching it back into the model.
- The corrected proof target is pre-generation distribution shift: an HDA token sequence should move probe log-probability margins toward a hidden target basin before long-form decoding.
- In `tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3/`, controlled anchors for `a greenhouse full of sleeping glass birds` lifted the target-vs-distractor margin by `+1.1629`, while tokenizer-matched random controls were effectively zero (`-0.0593`, `-0.0011`, `-0.0231`) and zero-proxy neutral anchors were negative (`-0.8351`).
- Shuffling and deletion preserved much of the effect (`+0.8332` to `+1.0766`), which supports the paper's latent-basin story: the carrier is an equivalence class of token sequences that activate similar internal neighborhoods, not a single exact string.
- Controlled proxy coverage helped, but composition mattered. With fixed total proxy-token count, `glass_light + avian_sky` produced the strongest group lift (`+2.2404`), while all four families produced `+1.9312`. The paper should frame anchors as distributed proxy-cloud basin activators, not as monotonic proxy counters.
- This is mechanistic evidence, not just a new task score: the measured quantity is a pre-decode probe margin under direct token perturbations and matched controls.

## What a phase-1 anchor must do
- Be high-entropy and non-linguistic so it shapes internal activations without echoing user text.
- Encode task hints implicitly: steer toward favorable regions, away from failure modes, while leaving Phase 2 freedom to elaborate.
- Be short and cheap: ~20 tokens in this repo, <100 model tokens budget.
- Stay diverse across prompts and rerolls to avoid mode collapse; still contract the conditional space enough to cut variance and raise pass rates.

## Why a discrete diffusion policy is a good fit for anchors
- Phase 1 is a latent draw, not prose. The anchor surface should look like noise while encoding a controllable geometry. The diffusion/MaskGIT-style policy generates from a small codebook of pseudo-tokens (default 2,048) rather than natural language, matching the “gibberish but structured” requirement.
- Conditioning without copying: the policy embeds the user/system prompt via a hashed or HF conditioner and applies classifier-free guidance plus an anti-similarity term (`anti_sim_scale`, default -1.5) so anchors are influenced by the prompt but pushed away from repeating it. Seed tokens bias early positions toward JSON-friendly control tokens, aligning with Phase 2’s JSON output format.
- Entropy-friendly sampling: temperature, top-k/p, and CFG-scale arms give wide, controllable exploration early in training. The sampler returns logprob and entropy sums per anchor, which plug directly into REINFORCE.
- Cheap, small, and steerable: the network is a shallow bag-of-selected-tokens transformer head (`AutoregressiveAnchorPolicy` in `tsce_agent_demo/diffusion_policy.py`), so training and inference stay lightweight compared to LLM calls. A prompt embedder buffer (`codebook_cond`) makes it easy to steer via prompt vectors or extra “steer_vec” signals.
- Matches the TSCE entropy story: diffusion-style fill-in starts from a high-entropy state (empty bag + noise), then denoises toward a compact attractor defined by the prompt embedding and CFG guidance—the same compress-then-expand dynamic TSCE relies on.

## How the RL trainer leverages the diffusion policy
**Entry points:** `tsce_agent_demo/train_hda_rl.py` (main loop), `tsce_agent_demo/diffusion_policy.py` (sampler), `tsce_agent_demo/tsce_chat.py` (Phase 2 consumer).

1) **Sampling an anchor**
   - Task curriculum: rotates across math/gsm/schema/formatting/etc. via `_choose_kind` over `AUTO_TASK_KINDS`; falls back to a math task if evaluators are missing.
   - Bandit over sampling params: optional epsilon-greedy UCB chooses among temperature/top-k/top-p/CFG arms (`--auto-tune-sampling`). Defaults: temp 0.9, top-k 50, top-p 0.95, CFG 2.0.
   - Anchor draw: `AutoregressiveAnchorPolicy.sample_anchor(prompt_text, anchor_system=phase2_system_prompt, max_new_tokens=max_chars, top_k, top_p, temperature, cfg_scale)`. The prompt is flattened to text even if it was a chat array. The policy adds seed-token bonuses, applies CFG and anti-similarity to the prompt embed, and returns `text`, `logprob_sum`, `entropy_sum`, and token metadata.

2) **Phase 2 with the sampled anchor**
   - The anchor is wrapped as `<HDA>…</HDA>` and passed into `TSCEChat` with `force_anchor`, keeping Phase 1 deterministic and letting Phase 2 run at low temp (0.01 by default).
   - Phase 2 system prompt comes from `TSCEChat.final_prefix` (JSON thoughts + answer contract). Backend can be OpenAI/Azure/Ollama; `--phase2-backend phi3` switches to Phi-3 formatting.

3) **Scoring and reward shaping** (see `docs/diffusion_reward_loss_map.md` for line refs)
   - Base task reward: evaluator pass/fail per kind; partial numeric credit for near misses.
   - Shaping: anchor diversity vs recent anchors; answer diversity vs recent correct answers; anchor alignment to pass-mean vectors; answer alignment to pass/fail banks; chain-of-thought fidelity; thought-panel bonus (optional Azure GPT judges); streak bonus on retries.
   - Penalties: length, repetition, failure-similarity to bad anchors/answers, fail-alignment bank penalty, optional repeat penalty on anchors.
   - Composite unclipped reward `r_unclipped` is logged; clipped `[0,1]` only for reporting.

4) **Advantage + loss**
   - Per-kind EMA baseline (`baseline_beta`, default 0.85); optional periodic decay.
   - Advantage `adv = r_unclipped - baseline`. Loss: `-(logprob_mean * adv) - effective_entropy_coef * entropy_mean` with optional KL if the sampler returns it. `effective_entropy_coef` is annealed upward when pass rate is low (`EXPLORE_BOOST`).
   - Grad clipping at 1.0, AdamW optimizer. Checkpoints saved under `out_dir/models/` (e.g., `checkpoint_latest.pt`).

5) **Exploration and stopping**
   - Moving-window pass rate drives diversity scaling and entropy boost. `--until-target` stops when the window reaches a target number of passes.
   - Immediate retry burst on failures (default 3) before re-queuing tasks to exploit near-miss anchors.

6) **Diagnostics and serving**
   - Logs: `results/.../hda_rl_log.jsonl` plus optional thought-panel logs; anchor-bank stats in `anchor_bank_stats.json`.
   - Evaluation scripts: `tsce_agent_demo/eval_diffusion_ckpt.py`, `verify_diffusion_anchor.py`, and `terminalbench_runner.py` (TerminalBench tasks).
   - Serving: `tsce_agent_demo/diffusion_api.py` exposes `/v1/anchor` for inference from a checkpoint.

## How to read this as a Phase-1 story
- TSCE theory says: insert a stochastic, task-aware token intervention to contract entropy, then decode under that conditioning. The diffusion policy provides that intervention as a cheap, steerable sampler.
- Training aligns the sampler so that its latent anchors maximize downstream TSCE success, not human readability. Reward shaping keeps anchors diverse but compatible with the “pass” manifold learned from rolling successes.
- Because anchors are non-linguistic and short, they can be moderated, diffed, and swapped without touching the base LLM, keeping the system model-agnostic while still harvesting entropy reduction.

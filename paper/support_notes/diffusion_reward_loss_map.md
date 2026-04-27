# Diffusion RL Reward & Loss Map

This repo shapes the diffusion anchor policy with REINFORCE in `tsce_agent_demo/train_hda_rl.py`, backed by the discrete diffusion sampler in `tsce_agent_demo/diffusion_policy.py`. Below is a map of every place where reward, shaping, baselines, and loss are defined.

## Base task scoring and success checks
- `score_reply` routes phase-2 answers to the task-specific evaluators and returns the base score (usually `0/1`, float for creative writing) `tsce_agent_demo/train_hda_rl.py:270-312`.
- `partial_numeric_reward` gives smooth partial credit on numeric-ish tasks (letter_counting/math/gsm/table_reasoning) by normalizing absolute error and capping at `0.75` `tsce_agent_demo/train_hda_rl.py:625-652`.
- `_is_success` compares the evaluator score to per-kind thresholds (defaults `>=0.999`, creative writing `0.85`) `tsce_agent_demo/train_hda_rl.py:655-657`.

## Shaping helpers (reused inside the reward builder)
- Repetition/entropy diagnostics: `repetition_stats` and `repeat_penalty` compute token-level uniqueness penalties `tsce_agent_demo/train_hda_rl.py:702-724`.
- Diversity bonus: cosine distance from recent anchors/answers via `diversity_bonus` `tsce_agent_demo/train_hda_rl.py:726-737`.
- CoT fidelity: heuristic structure score (numbered steps, math checks, inline JSON) capped to `<=0.5` when used `tsce_agent_demo/train_hda_rl.py:742-795`.
- Failure similarity: hinge penalty against recent failed anchors (`failure_similarity_penalty`) and failed answers (`answer_repeat_penalty`) `tsce_agent_demo/train_hda_rl.py:798-846`.
- Thought panel: optional Azure GPT judges returning a quality score averaged over sampled personas `_score_thoughts_panel*` `tsce_agent_demo/train_hda_rl.py:547-609`.
- Anchor bank: rolling store for pass-aligned mean vectors used for pass-alignment shaping `tsce_agent_demo/train_hda_rl.py:853-925`.

## Reward assembly per sampled anchor
All reward components are computed inside `_evaluate_sample` `tsce_agent_demo/train_hda_rl.py:1382-1530`:
- **Task utility** `r_task`: evaluator pass/fail with optional partial numeric credit; `is_ok` treats partial `>=0.6` as “good enough” for diversity.
- **Streak bonus**: `PASS_STREAK_BONUS` when a pass occurs after retries `tsce_agent_demo/train_hda_rl.py:1404-1407`.
- **Embeddings**: anchors, scalar answers, and full answers embedded for alignment/diversity.
- **Pass alignment bonus**: cosine(answer, pass_mean_final_vec) from `AnchorBank`, scaled by current pass rate `c_succ` and `PASS_ALIGN_WEIGHT` `tsce_agent_demo/train_hda_rl.py:1425-1454`.
- **Fail alignment penalty**: cosine to failed answers multiplied by `FAIL_ALIGN_WEIGHT` `tsce_agent_demo/train_hda_rl.py:1425-1454`.
- **Anchor alignment bonus**: cosine(anchor, mean pass anchor) scaled by `anchor_align_weight` `tsce_agent_demo/train_hda_rl.py:1455-1468`.
- **Diversity**: anchor diversity vs rolling history and answer diversity vs recent correct answers (disabled for numeric kinds). Both scaled by `diversity_scale` and `w_answer_div_effective`, which are annealed with pass rate `c_succ` `tsce_agent_demo/train_hda_rl.py:1469-1475` and `1600-1603`.
- **Thought quality**: optional thought-panel bonus `THOUGHT_PANEL_WEIGHT * score` `tsce_agent_demo/train_hda_rl.py:1477-1488`, `1524-1527`.
- **CoT fidelity bonus**: `fid_weight * cot_fid_score` (capped at `0.5`) `tsce_agent_demo/train_hda_rl.py:1485-1488`, `1506-1514`.
- **Penalties**:
  - Repeat-token penalty (`repeat_penalty_weight`) toggled by `--anchor-repeat-penalty` `tsce_agent_demo/train_hda_rl.py:1500-1502`, `1516-1522`.
  - Failure-similarity penalty to failed anchors (`FAIL_SIM_WEIGHT`, scaled by success rate) `tsce_agent_demo/train_hda_rl.py:1503-1505`, `1516-1522`.
  - Answer repeat penalty on failed answers (`ANSWER_REPEAT_WEIGHT`) `tsce_agent_demo/train_hda_rl.py:1490-1496`, `1516-1522`.
  - Length penalty (`anchor_len_penalty * token_count`) `tsce_agent_demo/train_hda_rl.py:1498`, `1516-1522`.
  - Fail-alignment penalty from the bank (`FAIL_ALIGN_WEIGHT`) `tsce_agent_demo/train_hda_rl.py:1425-1454`, `1516-1522`.
- **Composite reward**: `positive_reward = w_task * r_task + w_task*streak + anchor_div + answer_div + fid + anchor_align + pass_align`; `penalties` summed separately; `r_unclipped = positive_reward + thought_panel_bonus - penalties` `tsce_agent_demo/train_hda_rl.py:1506-1529`.
- Rewards are clipped to `[0,1]` only for logging; training uses unclipped values `tsce_agent_demo/train_hda_rl.py:1915-1922`.

## Batch-level adjustments (multi-sample SCST-style)
- When `batch_k > 1`, each sample is evaluated; either the top reward is chosen or the first sample is kept based on `--select-top` `tsce_agent_demo/train_hda_rl.py:1647-1651`.
- Additional `batch_div_bonus` equals the mean pairwise diversity of correct answers and is added to `r_unclipped` before advantage computation `tsce_agent_demo/train_hda_rl.py:1653-1699`.
- Batch diversity stats and per-sample rewards are logged for analysis `tsce_agent_demo/train_hda_rl.py:1685-1699`, `1919-1978`.

## Annealing and exploration controls
- Effective entropy coefficient scales with `(1 + EXPLORE_BOOST * (1 - c_succ))`, encouraging exploration when pass rate is low `tsce_agent_demo/train_hda_rl.py:1599-1602`.
- Diversity weights scale up early via `MAX_DIVERSITY_SCALE` and down as success improves `tsce_agent_demo/train_hda_rl.py:1600-1603`.
- Answer-diversity weight bottoms at `20%` of the base weight to avoid collapse `tsce_agent_demo/train_hda_rl.py:1602-1603`.
- Pass-rate window (`ma_window`) tracks binary outcomes only; thresholds drive annealing and early stopping `tsce_agent_demo/train_hda_rl.py:1275-1276`, `1788-1800`, `2057-2059`.

## Advantage baselines
- Per-kind EMA baseline updated as `baseline = beta * baseline + (1-beta) * r_unclipped`; advantage `adv = r_unclipped - baseline` `tsce_agent_demo/train_hda_rl.py:1755-1758`.
- Optional soft reset halves baselines every `baseline_reset_interval` steps to prevent drift `tsce_agent_demo/train_hda_rl.py:1575-1583`.

## Loss construction
- Token-mean logprob and entropy are derived from the sampled anchor (`logprob_sum`, `entropy_sum`) produced by the diffusion policy `tsce_agent_demo/train_hda_rl.py:1759-1765`.
- Core objective: `loss = -(logprob_mean * adv) - effective_entropy_coef * entropy_mean` (REINFORCE with entropy bonus) `tsce_agent_demo/train_hda_rl.py:1767-1768`.
- Optional KL: if the sampler supplies `kl_sum`, the mean KL is added with weight `kl_coef`; otherwise KL is disabled with a warning `tsce_agent_demo/train_hda_rl.py:1770-1778`.
- Gradients are clipped to `1.0` before optimizer step `tsce_agent_demo/train_hda_rl.py:1780-1785`.

## Diffusion sampler hooks (source of logprob/entropy)
- `AutoregressiveAnchorPolicy.sample_anchor` generates the anchor, accumulates `logprob_sum` and `entropy_sum` per token, and returns them for the loss `tsce_agent_demo/diffusion_policy.py:274-371`.
- Classifier-free guidance and anti-similarity to the prompt shape logits; seed tokens bias the first positions toward JSON-friendly strings `tsce_agent_demo/diffusion_policy.py:235-272`, `303-343`.

## Logging/diagnostics tied to reward shaping
- Every training step logs the unclipped reward, task score, each shaping term, penalty, entropy, diversity, and alignment diagnostics in `hda_rl_log.jsonl` `tsce_agent_demo/train_hda_rl.py:1915-1997`.
- Rolling anchor/answer vectors are stored to drive diversity and alignment shaping and to populate the `AnchorBank` for pass-mean guidance `tsce_agent_demo/train_hda_rl.py:1812-1843`, `853-925`.

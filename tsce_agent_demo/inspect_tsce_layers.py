#!/usr/bin/env python3
"""
inspect_tsce_layers.py
------------------------------------------
Layer-wise analysis of Two-Step Contextual
Enrichment (TSCE) on Mistral-7B with no
external APIs—uses transformer-lens.

USAGE
  python inspect_tsce_layers.py agentic_prompts.txt       # full 50 prompts
  python inspect_tsce_layers.py agentic_prompts.txt --max_n 10
  python inspect_tsce_layers.py agentic_prompts.txt --device cpu
"""

import argparse, time, math, json, statistics as stats
from pathlib import Path
import torch, transformer_lens as tl
from transformers import AutoTokenizer
from tqdm import tqdm

# ================================─ CLI ==================================
p = argparse.ArgumentParser()
p.add_argument("prompt_file")
p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1")
p.add_argument("--device", default="cuda", help="cuda | cpu | mps")
p.add_argument("--dtype", default="float16", help="float16 | bfloat16 | float32")
p.add_argument("--max_n", type=int, default=None)
p.add_argument("--save_json", default="tsce_layer_log.json")
args = p.parse_args()

# =============================- load model ============================─
dtype_map = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)
print(f"Loading {args.model} → {args.device}  ({args.dtype})")
# -- wrap in inference-mode to save memory & disable grads
with torch.inference_mode():
    model = tl.HookedTransformer.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        fold_ln=True,
        trust_remote_code=True,
        # load_in_8bit=True,
    )
tok = AutoTokenizer.from_pretrained(args.model)

# ========================─ helper: build HDA ==========================─
ANCHOR_SYS = (
    "You are a latent-space compression module. "
    "Return a single dense sentence that captures the FULL semantics of the "
    "user’s request with no additional context or explanation:"
)

def build_hda(prompt: str) -> str:
    """Phase-1 generation → returns the anchor sentence."""
    toks = model.to_tokens([ANCHOR_SYS + "\n\n" + prompt], prepend_bos=True)
    with torch.inference_mode():
        out = model.generate(toks, max_new_tokens=64, temperature=0.0)
    anchor = tok.decode(out[0]).split(tok.eos_token)[0].strip()
    return anchor

# ============─ helper: capture residual variance layer-by-layer ========
def resid_variances(text: str):
    tokens = model.to_tokens(text, prepend_bos=True)
    _, cache = model.run_with_cache(
        tokens,
        stop_at_layer=None,
        names_filter=lambda n: n.endswith("hook_resid_post"),
    )
    return [
        cache[f"blocks.{l}.hook_resid_post"]
        .float()
        .var(unbiased=False)
        .item()
        for l in range(model.cfg.n_layers)
    ]

def answer_ok(output: str, gold: str) -> bool:
    return gold.lower() in output.lower()

# ========================─ load prompt set ============================─
prompts_raw = [ln.strip() for ln in Path(args.prompt_file).read_text().splitlines() if ln.strip()]
if args.max_n: prompts_raw = prompts_raw[:args.max_n]
prompts = []
for ln in prompts_raw:
    if "|ANSWER|" in ln:
        q, a = [s.strip() for s in ln.split("|ANSWER|", 1)]
        prompts.append((q, a))
print(f"Running {len(prompts)} prompts")

# ========================─ main loop ==================================─
rows, deltas = [], []
for idx, (q, gold) in tqdm(enumerate(prompts, 1), total=len(prompts)):
    # -- baseline (forward + cache)
    base_tokens = model.to_tokens(q, prepend_bos=True)
    with torch.inference_mode():
        base_start = time.time()
        base_out_tokens = model.generate(base_tokens, max_new_tokens=64, temperature=0.0)
    base_time = time.time() - base_start
    base_out  = tok.decode(base_out_tokens[0])
    base_ok   = answer_ok(base_out, gold)
    base_vars = resid_variances(q)

    # TSCE pass
    anchor = build_hda(q)
    tsce_prompt = anchor + "\n\n" + q
    tsce_tokens = model.to_tokens(tsce_prompt, prepend_bos=True)
    with torch.inference_mode():
        tsce_start = time.time()
        tsce_out_tokens = model.generate(tsce_tokens, max_new_tokens=64, temperature=0.0)
    tsce_time = time.time() - tsce_start
    tsce_out  = tok.decode(tsce_out_tokens[0])
    tsce_ok   = answer_ok(tsce_out, gold)
    tsce_vars = resid_variances(tsce_prompt)

    # delta per layer (tsce − baseline)
    layer_delta = [t - b for t, b in zip(tsce_vars, base_vars)]
    deltas.append(layer_delta)

    tqdm.write(f"{idx:02d}/{len(prompts)}  base={'✅' if base_ok else '❌'}  "
               f"tsce={'✅' if tsce_ok else '❌'}  "
               f"Δvar@L0={layer_delta[0]:+.3f}")

    rows.append(dict(
        prompt=q[:50]+"…",
        gold=gold,
        baseline_ok=base_ok,
        tsce_ok=tsce_ok,
        base_time=round(base_time,2),
        tsce_time=round(tsce_time,2),
        layer_delta=layer_delta,
    ))

# ========================─ summary stats ==============================─
acc_base = sum(r["baseline_ok"] for r in rows)/len(rows)*100
acc_tsce = sum(r["tsce_ok"] for r in rows)/len(rows)*100
mean_deltas = [stats.mean(col) for col in zip(*deltas)]

print("\n===== LAYER VARIANCE Δ (mean across prompts) =====")
for i, d in enumerate(mean_deltas):
    print(f"L{i:02d}: {d:+.4f}")

print("\n===== ACCURACY =====")
print(f"Baseline : {acc_base:5.1f} %")
print(f"TSCE     : {acc_tsce:5.1f} %   (Δ {acc_tsce-acc_base:+.1f} pp)")

# optional JSON dump
Path(args.save_json).write_text(json.dumps(rows, indent=2))
print(f"\nFull per-prompt log → {args.save_json}")

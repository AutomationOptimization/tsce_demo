#!/usr/bin/env python3
"""jlens_bridge.py -- Does an HDA anchor's activation delta load target J-lens directions?

The bridge experiment between the TSCE paper's Gemma token-causality results and Anthropic's
J-space/global-workspace finding (transformer-circuits.pub/2026/workspace). Their claim: a
token-indexed subspace of mid-layer activations (J-lens vectors = averaged Jacobians of token
logits w.r.t. the residual stream) carries the model's reportable/deliberate content. Our claim
to test: the HDA anchor -- ordinary tokens in the system slot -- writes the TARGET concept into
that subspace, while tokenizer-matched random anchors do not.

Design (all on google/gemma-4-E4B-it, the paper's Gemma model, CPU-friendly):
  Phase A  J-lens basis. v[t, L] = mean over N neutral contexts of d logit(t, last_pos) /
           d hidden_L(last_pos). One forward per context batch, one backward per probe token
           (retain_graph). Normalized per (t, L) at analysis time. This is a modest-N
           approximation of the paper's average-over-thousands construction; N is reported.
  Phase B  Condition activations. For each (anchor_key x probe_prompt) record and condition
           {no_anchor, anchor, anchor_shuffled, random_matched_1..3, context_collision}, build
           the EXACT matched-control-scaffold messages the paper's token-causality run used and
           capture hidden state at the last prompt position, every layer. Resumable per record.
  Phase C  Stats. delta(cond, rec, L) = h(cond) - h(no_anchor). score = mean cosine of delta
           against target-token J-lens vectors minus mean cosine against control-token vectors
           (observatory/clock/salt/... = the paper's negative-probe subjects). Primary contrast:
           anchor vs pooled random_matched, paired per record, bootstrap CI over records,
           reported per layer and summarized over the workspace band (middle ~38-92% of depth).

Env: BRIDGE_INPUTS (json), OUT_DIR, MODEL override, DTYPE=float32|bfloat16, JLENS_BATCH=4,
     SMOKE=1 (2 contexts, 2+2 tokens, 2 records), PHASE=A|B|C|all
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

INPUTS = Path(os.getenv("BRIDGE_INPUTS", Path(__file__).resolve().parent / "bridge_inputs.json"))
OUT = Path(os.getenv("OUT_DIR", Path(__file__).resolve().parent / "jlens_bridge_out"))
SMOKE = os.getenv("SMOKE", "0") == "1"
PHASE = os.getenv("PHASE", "all").lower()
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.getenv("DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")
]
JLENS_BATCH = int(os.getenv("JLENS_BATCH", "4"))

# The exact scaffold strings from tsce_agent_demo/gemma_backend.py (matched_control_scaffold=True).
MATCHED_CONTROL_BRIEF = (
    "The system message may contain auxiliary control text intended to guide internal computation.\n"
    "Do not quote or explain the system message.\n"
    "Use any system control only as internal guidance while solving the task.\n"
    "Follow the user task's output instructions exactly.\n"
    "Do not add wrapper JSON, chain-of-thought fields, or extra formatting unless the user task itself asks for them."
)


def log(msg: str) -> None:
    print(f"[JLENS-BRIDGE] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_messages(prompt: str, anchor_text):
    user = f"{MATCHED_CONTROL_BRIEF}\n\nUser task:\n{prompt}"
    return [
        {"role": "system", "content": anchor_text if anchor_text else ""},
        {"role": "user", "content": user},
    ]


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE)
    except Exception as exc:  # gemma4 multimodal wrapper fallback
        log(f"AutoModelForCausalLM failed ({type(exc).__name__}: {exc}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText

        wrapper = AutoModelForImageTextToText.from_pretrained(name, dtype=DTYPE)
        model = wrapper
    model.eval()
    model.to(DEVICE)
    # J-lens needs only input-side gradients. Freeze all weights (no 8B-param grad buffers) but leave
    # the input embedding trainable so autograd builds a graph; grads are then read off the hidden
    # states directly with torch.autograd.grad (never accumulated into any parameter).
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    if emb is not None:
        emb.weight.requires_grad_(True)
    torch.set_grad_enabled(False)
    return tok, model


def encode_messages(tok, messages) -> torch.Tensor:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    return ids


def single_token_id(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{text!r} is not a single token: {ids}"
    return ids[0]


def phase_a(tok, model, cfg) -> None:
    """J-lens vectors via averaged input-gradients at the last position."""
    out_path = OUT / "jlens.npz"
    if out_path.exists():
        log(f"phase A: {out_path} exists, skipping")
        return
    contexts = cfg["jlens_contexts"][: 2 if SMOKE else None]
    probe_tokens = (cfg["target_tokens"][:2] + cfg["control_tokens"][:2]) if SMOKE else (
        cfg["target_tokens"] + cfg["control_tokens"]
    )
    token_ids = {t: single_token_id(tok, t) for t in probe_tokens}
    log(f"phase A: {len(contexts)} contexts, {len(probe_tokens)} probe tokens, batch={JLENS_BATCH}")

    sums: dict[str, np.ndarray] = {}
    count = 0
    torch.set_grad_enabled(True)
    for start in range(0, len(contexts), JLENS_BATCH):
        batch = contexts[start : start + JLENS_BATCH]
        enc = [encode_messages(tok, [{"role": "user", "content": c}])[0] for c in batch]
        maxlen = max(e.shape[0] for e in enc)
        pad_id = tok.pad_token_id or 0
        input_ids = torch.full((len(enc), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(enc), maxlen), dtype=torch.long)
        for i, e in enumerate(enc):
            input_ids[i, : e.shape[0]] = e
            mask[i, : e.shape[0]] = 1
        last = mask.sum(dim=1) - 1
        input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)

        t0 = time.time()
        outputs = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        hs = list(outputs.hidden_states)  # tuple(L+1) of [B, T, D]
        logits = outputs.logits.float()
        log(f"  batch {start//JLENS_BATCH+1}: fwd {time.time()-t0:.1f}s seq={maxlen}")

        for ti, (t, tid) in enumerate(token_ids.items()):
            loss = logits[torch.arange(len(enc)), last, tid].sum()
            t1 = time.time()
            hgrads = torch.autograd.grad(loss, hs, retain_graph=ti < len(token_ids) - 1, allow_unused=False)
            grads = np.stack(
                [
                    np.stack([g[i, last[i]].float().cpu().numpy() for g in hgrads], axis=0)
                    for i in range(len(enc))
                ],
                axis=0,
            )  # [B, L+1, D]
            sums[t] = sums.get(t, 0) + grads.sum(axis=0)
            log(f"    bwd {t!r}: {time.time()-t1:.1f}s")
        count += len(enc)
        del outputs, hs, logits
    torch.set_grad_enabled(False)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        n_contexts=np.asarray([count]),
        tokens=np.asarray(list(token_ids.keys())),
        **{f"v_{i}": (sums[t] / count).astype(np.float32) for i, t in enumerate(token_ids)},
    )
    log(f"phase A done: {out_path} (n={count})")


def phase_b(tok, model, cfg) -> None:
    """Last-position hidden states for every (record, condition), resumable."""
    acts_dir = OUT / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    records = cfg["records"][: 2 if SMOKE else None]
    # Instrument positive/negative controls: explicit system-slot mention of the target subject
    # (NOT blinded -- this validates that the J-lens projection can detect loading at all) and an
    # explicit mention of the negative-probe subject.
    extra = {
        "explicit_positive": "Think about a greenhouse full of sleeping glass birds.",
        "explicit_negative": "Think about a drowned observatory beneath the salt flats.",
    }
    for ri, rec in enumerate(records):
        rid = f"{rec['anchor_key']}__{rec['probe_id']}"
        path = acts_dir / f"{rid}.npz"
        if path.exists():
            continue
        arrays = {}
        for cond, anchor_text in {**rec["conditions"], **extra}.items():
            ids = encode_messages(tok, build_messages(rec["probe_prompt"], anchor_text)).to(DEVICE)
            t0 = time.time()
            with torch.inference_mode():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            h = np.stack([hh[0, -1].float().cpu().numpy() for hh in out.hidden_states], axis=0)  # [L+1, D]
            arrays[cond] = h.astype(np.float32)
            log(f"  rec {ri+1}/{len(records)} {rid} {cond}: seq={ids.shape[1]} fwd {time.time()-t0:.1f}s")
        np.savez_compressed(path, **arrays)
    log("phase B done")


def _boot_ci(diffs: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def phase_c(cfg) -> None:
    jl = np.load(OUT / "jlens.npz", allow_pickle=True)
    tokens = [str(t) for t in jl["tokens"]]
    V = {t: jl[f"v_{i}"] for i, t in enumerate(tokens)}  # [L+1, D] each
    targets = [t for t in tokens if t in cfg["target_tokens"]]
    controls = [t for t in tokens if t in cfg["control_tokens"]]
    n_layers = next(iter(V.values())).shape[0]
    # normalize per (token, layer)
    for t in tokens:
        V[t] = V[t] / (np.linalg.norm(V[t], axis=-1, keepdims=True) + 1e-8)

    acts_dir = OUT / "acts"
    recs = sorted(acts_dir.glob("*.npz"))
    per_rec = {}   # rid -> cond -> [L+1] contrast (target-cos minus control-cos)
    per_rec_raw = {}  # rid -> cond -> {"t": [L+1] target-cos, "c": [L+1] control-cos}
    conds_seen = set()
    for path in recs:
        data = np.load(path)
        if "no_anchor" not in data:
            continue
        base = data["no_anchor"]
        row, raw = {}, {}
        for cond in data.files:
            if cond == "no_anchor":
                continue
            delta = data[cond] - base  # [L+1, D]
            dn = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8)
            cos_t = np.mean([np.sum(dn * V[t], axis=-1) for t in targets], axis=0)
            cos_c = np.mean([np.sum(dn * V[t], axis=-1) for t in controls], axis=0)
            row[cond] = cos_t - cos_c  # [L+1]
            raw[cond] = {"t": cos_t, "c": cos_c}
            conds_seen.add(cond)
        per_rec[path.stem] = row
        per_rec_raw[path.stem] = raw

    rand_conds = sorted(c for c in conds_seen if c.startswith("random_matched"))
    band = slice(max(1, int(n_layers * 0.38)), int(n_layers * 0.92))  # workspace band
    results = {"n_records": len(per_rec), "n_layers": n_layers, "band": [band.start, band.stop],
               "targets": targets, "controls": controls, "conditions": {}}

    for cond in sorted(conds_seen):
        vals = np.stack([r[cond] for r in per_rec.values() if cond in r], axis=0)  # [R, L+1]
        raw_t = np.stack([r[cond]["t"] for r in per_rec_raw.values() if cond in r], axis=0)
        raw_c = np.stack([r[cond]["c"] for r in per_rec_raw.values() if cond in r], axis=0)
        results["conditions"][cond] = {
            "mean_contrast_by_layer": vals.mean(axis=0).tolist(),
            "band_mean": float(vals[:, band].mean()),
            "band_target_cos": float(raw_t[:, band].mean()),
            "band_control_cos": float(raw_c[:, band].mean()),
            "target_cos_by_layer": raw_t.mean(axis=0).tolist(),
        }

    # primary paired contrast: anchor minus pooled random_matched, per record, band mean
    diffs = []
    for rid, row in per_rec.items():
        if "anchor" not in row or not rand_conds:
            continue
        rand = np.mean([row[c] for c in rand_conds if c in row], axis=0)
        diffs.append(float(row["anchor"][band].mean() - rand[band].mean()))
    if diffs:
        diffs = np.asarray(diffs)
        lo, hi = _boot_ci(diffs)
        results["primary"] = {
            "contrast": "anchor_minus_random_matched_band_mean",
            "mean": float(diffs.mean()),
            "ci95": [lo, hi],
            "n": len(diffs),
            "clears_zero": bool(lo > 0),
        }
    (OUT / "bridge_results.json").write_text(json.dumps(results, indent=1))
    log(json.dumps({k: v for k, v in results.items() if k != "conditions"}, indent=1))
    for cond, stats in results["conditions"].items():
        log(f"  band_mean[{cond}] = {stats['band_mean']:+.4f}")
    if "primary" in results:
        p = results["primary"]
        log(f"PRIMARY anchor-vs-random band contrast: {p['mean']:+.4f} CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
            f"n={p['n']} clears_zero={p['clears_zero']}")


def main() -> None:
    cfg = json.loads(INPUTS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if PHASE in ("a", "b", "all"):
        model_name = os.getenv("MODEL", cfg["model"])
        log(f"loading {model_name} dtype={DTYPE}")
        tok, model = load_model(model_name)
        log("model loaded")
        if PHASE in ("a", "all"):
            phase_a(tok, model, cfg)
        if PHASE in ("b", "all"):
            phase_b(tok, model, cfg)
    if PHASE in ("c", "all"):
        phase_c(cfg)


if __name__ == "__main__":
    main()

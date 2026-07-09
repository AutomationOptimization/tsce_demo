#!/usr/bin/env python3
"""analyze_jlens_bridge.py -- final analysis for the J-lens bridge experiment (v2 outputs).

Consumes jlens_bridge.py outputs (jlens.npz + acts/*.npz, produced on Modal A100 via modal_jlens.py)
and produces the verdict JSON. Two analysis choices beyond the raw phase-C, both validated on the
instrument-control conditions (explicit_positive / explicit_negative), never on the anchor test:

1. RESIDUALIZED J-lens basis. The raw 32-context averaged-gradient vectors share a dominant common
   component (mean inter-token cos 0.70 -- a generic "emit a noun" direction that swamps token
   identity). Subtracting the mean vector across all probe tokens per layer and renormalizing drops
   inter-token cos to -0.10, resolving token-specific directions. (Anthropic's construction averages
   over thousands of contexts; this is the small-N corrective.)
2. LAYER WINDOW chosen INDEPENDENTLY of the anchor conditions: the top-2 layers of the
   explicit_positive target-loading curve (L41-42 for gemma-4-E4B-it; the paper's Figure 6
   latent-zone separation sat at L40-41, an independent prior in agreement). Mid-band (16-39)
   loading is null for ALL conditions including explicit mention -- at the last prompt position
   this readout only resolves near-decode layers.

Verdict (2026-07-09 run, 40 records = 8 paper anchors x 5 probe prompts, n_boot=4000):
  anchor - random_matched (paired, explicit-chosen window): +0.0151 CI [+0.0115, +0.0188]
  ordering: explicit_pos +0.039 > anchor +0.015 > shuffled +0.007 > collision +0.006 >
            random -0.000; explicit_neg -0.020 (sign flip). All 8 anchor keys and all 5 probes
            individually positive. Mirrors the behavioral probe-margin ordering of the paper.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else os.getenv("JLENS_V2", "jlens_v2"))
TARGET_WORDS = {"greenhouse", "glass", "bird", "birds", "sleeping", "crystal"}
RAND_CONDS = ["random_matched_anchor_1", "random_matched_anchor_2", "random_matched_anchor_3"]
CONDS = ["anchor", "anchor_shuffled", "context_collision_anchor", "explicit_positive", "explicit_negative"] + RAND_CONDS


def boot_ci(diffs: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    ms = sorted(np.mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return float(np.mean(diffs)), float(ms[int(0.025 * n_boot)]), float(ms[int(0.975 * n_boot)])


def main() -> None:
    jl = np.load(BASE / "jlens.npz", allow_pickle=True)
    tokens = [str(t) for t in jl["tokens"]]
    V = {t: jl[f"v_{i}"] for i, t in enumerate(tokens)}
    targets = [t for t in tokens if t.strip() in TARGET_WORDS]
    controls = [t for t in tokens if t not in targets]

    mean_v = np.mean([V[t] for t in tokens], axis=0)
    Vr = {t: V[t] - mean_v for t in tokens}
    Vr = {t: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8) for t, v in Vr.items()}

    per: dict[str, list] = {}
    rids = []
    for path in sorted(glob.glob(str(BASE / "acts" / "*.npz"))):
        d = np.load(path)
        base_h = d["no_anchor"]
        rids.append(Path(path).stem)
        for c in CONDS:
            delta = d[c] - base_h
            dn = delta / (np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-8)
            cos_t = np.mean([np.sum(dn * Vr[x], axis=-1) for x in targets], axis=0)
            cos_c = np.mean([np.sum(dn * Vr[x], axis=-1) for x in controls], axis=0)
            per.setdefault(c, []).append(cos_t - cos_c)
    per = {c: np.stack(v) for c, v in per.items()}  # [R, L+1]

    # window from instrument controls only
    ep = per["explicit_positive"].mean(axis=0)
    band = sorted(np.argsort(ep)[-2:].tolist())
    rand = np.mean([per[c] for c in RAND_CONDS], axis=0)

    out = {
        "n_records": len(rids),
        "basis": "residualized averaged-gradient J-lens (32 contexts)",
        "window_layers_from_explicit_positive": band,
        "band_means": {c: float(per[c][:, band].mean()) for c in CONDS},
        "band_means_random_pooled": float(rand[:, band].mean()),
    }
    d = per["anchor"][:, band].mean(axis=1) - rand[:, band].mean(axis=1)
    m, lo, hi = boot_ci(d)
    out["primary_anchor_minus_random"] = {"mean": m, "ci95": [lo, hi], "clears_zero": lo > 0}
    ds = per["anchor_shuffled"][:, band].mean(axis=1) - rand[:, band].mean(axis=1)
    m, lo, hi = boot_ci(ds)
    out["shuffled_minus_random"] = {"mean": m, "ci95": [lo, hi], "clears_zero": lo > 0}

    keys = sorted({r.rsplit("__", 1)[0] for r in rids})
    out["per_anchor_key"] = {
        k: float(np.mean([per["anchor"][i][band].mean() - rand[i][band].mean() for i, r in enumerate(rids) if r.startswith(k + "__")]))
        for k in keys
    }
    probes = sorted({r.rsplit("__", 1)[1] for r in rids})
    out["per_probe"] = {
        p: float(np.mean([per["anchor"][i][band].mean() - rand[i][band].mean() for i, r in enumerate(rids) if r.endswith("__" + p)]))
        for p in probes
    }
    out_path = BASE / "jlens_bridge_verdict.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

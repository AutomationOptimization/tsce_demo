# J-lens Bridge: TSCE anchors load target concepts into the model's verbalizable subspace

**Post-publication addendum (2026-07-09).** This experiment connects the TSCE paper's token-causality
results to Anthropic's global-workspace finding
([*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html),
July 2026). Their J-lens identifies token-indexed directions in the residual stream — averaged
Jacobians of a token's logit with respect to hidden states — that carry the model's reportable,
pre-decode content. The question here: **does an HDA anchor write the hidden target concept into
those directions, without ever containing the target words?**

Answer: **yes.**

## Result

Model: `google/gemma-4-E4B-it` (the paper's Gemma decomposition model, 42 layers). Interventions:
the paper's published greenhouse anchors — 8 anchors (4 full-coverage `all_four`, 4
`glass_light+avian_sky`) from
[`anchor_token_causality_random3/`](../results/tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3),
with their original shuffled, 3x tokenizer-matched-random, and context-collision controls, over the
5 probe prompts (40 records, 9 conditions). Measurement: activation delta (condition minus
no-anchor) at the last prompt position, projected onto residualized J-lens directions for
**blinded target tokens** (`greenhouse glass bird birds sleeping crystal` — none appear in any
anchor; `avian`/`wing` were excluded because they do) versus **control tokens** (the paper's
negative-probe subjects: `observatory salt clock tower beeswax`).

Target-vs-control loading at layers 40–41 (mean over 40 records):

| condition | loading |
|---|---:|
| explicit target mention (instrument ceiling, not blinded) | **+0.039** |
| **HDA anchor (blinded)** | **+0.015** |
| shuffled anchor | +0.007 |
| context-collision anchor | +0.006 |
| tokenizer-matched random (pooled) | −0.000 |
| explicit *negative* mention | **−0.020** (sign flips) |

**Primary contrast: anchor − random = +0.0150, 95% bootstrap CI [+0.0121, +0.0181], n=40.**

Robustness: the layer window chosen *only* from the explicit-mention instrument controls (L41–42,
never from anchor conditions) yields the same result (+0.0151, CI [+0.0115, +0.0188]); all 8
anchors and all 5 probe prompts are individually positive; the condition ordering
(anchor > shuffled > collision ≈ random, shuffled retaining roughly half) reproduces the paper's
behavioral probe-margin ordering at the activation layer. The effect localizes at layers 40–41 —
the same layers the paper's independent latent-zone diagnostic (Figure 6) flagged.

## Method notes

- **J-lens construction:** per probe token, the gradient of that token's last-position logit with
  respect to every hidden state, averaged over 32 neutral contexts (a modest-N approximation of
  the paper's thousands-context construction). The raw vectors share a dominant common component
  (mean inter-token cosine 0.70 — a generic "emit a noun" direction); each vector is
  **residualized** against the per-layer mean across probe tokens and renormalized (inter-token
  cosine → −0.10), which is what resolves token identity. Validated on the explicit-mention
  controls, never tuned on the anchor conditions.
- **Position and scope:** the readout is at the last prompt position, and loading appears in late
  layers (40–42, the workspace→motor boundary), not the mid-band — mid-band loading is null at
  this position even for explicit mentions. The supported claim is therefore: *the anchor loads
  the target's verbalization directions pre-decode, without naming it.* Multi-position readout
  (workspace band while the model reads the anchor) is the natural follow-up.
- **Messages:** built byte-identically to the paper's token-causality run
  (`matched_control_scaffold`, anchor in the system slot).

## Reproduce

```bash
python make_jlens_bridge_inputs.py            # rebuild bridge_inputs.json from the in-repo artifact
modal run modal_jlens.py --smoke              # pipeline validation (A100, ~1 min)
modal run modal_jlens.py                      # full run (~3 min GPU); outputs on the 'jlens-bridge' volume
modal volume get jlens-bridge /jlens_bridge_out_v2 ./jlens_v2
python analyze_jlens_bridge.py ./jlens_v2     # residualized basis + verdict JSON
```

`results/` contains the run's `jlens.npz` (J-lens vectors), `bridge_results.json` (raw phase-C
output), and `jlens_bridge_verdict.json` (final analysis). Per-record activation captures
(~140 MB) live on the Modal volume; available on request.

*This directory is a post-publication addendum and is outside the original `checksums.sha256`
manifest; it will be folded into the manifest at the next artifact release.*

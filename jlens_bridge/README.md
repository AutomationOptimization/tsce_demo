# J-lens Bridge: the anchors were writing into the workspace all along

**2026-07-09.** On July 6, Anthropic published
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) —
the finding that language models carry their reportable, pre-decode content in a small set of
token-indexed directions in the residual stream (read out by their **J-lens**: averaged Jacobians
of a token's logit with respect to hidden states), and that this workspace mediates deliberate
reasoning while automatic processing runs without it.

I have been building the black-box counterpart of that mechanism since 2023. This addendum runs
their instrument against my published anchors and closes the loop: **TSCE's blinded HDA anchors —
ordinary tokens in the system slot, never containing the target words — load the hidden target
concept into the model's verbalizable token directions before it speaks.** Tokenizer-matched
random strings load nothing.

## Three years, one claim, four levels of evidence

The claim has never changed: *a compact, opaque token prefix shifts the model's pre-decode state
toward a target basin the prefix never names.* What changed is how directly it could be measured.

| when | level | evidence |
|---|---|---|
| **2023** | production behavior | TSCE two-pass anchoring deployed in Automation Optimization client work (documented lineage, [paper §2.1](../paper/think_before_you_speak_v2.tex)) |
| **Apr–May 2025** | public release | [`tsce` on PyPI](https://pypi.org/project/tsce/) (0.1.1, 2025-04-22), [HN](https://news.ycombinator.com/item?id=43899325) [posts](https://news.ycombinator.com/item?id=43991918), this repo |
| **Apr 2026** | controlled causality | the paper: 17-run outcome summary (62.9% → 81.2% paired); Gemma decomposition (149/300 rescues, scaffold ablation to 4.8%); **pre-decode token-causality**: task-relevant anchors move target probe margins (+1.16 greenhouse, +2.58 forgiveness) while tokenizer-matched random controls sit at ~0; **Figure 6**: anchor-quality separation in Gemma activations at layers 40–41 |
| **Jul 2026** | subspace identification | **this addendum**: the same anchors' activation deltas align with the target's J-lens directions — the workspace coordinates Anthropic just published — at the same layers 40–41 |

Anthropic found the room from inside the house. These anchors have been posting messages through
the mail slot for three years — and the paper's Figure 6 had already circled the right wall before
anyone had a name for it.

## The measurement

Model: `google/gemma-4-E4B-it` (the paper's Gemma decomposition model, 42 layers). Interventions:
the paper's published greenhouse anchors, verbatim — 8 anchors (4 full-coverage `all_four`, 4
`glass_light+avian_sky`) from
[`anchor_token_causality_random3/`](../results/tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3),
with their original shuffled, 3x tokenizer-matched-random, and context-collision controls, over
the 5 probe prompts (40 records, 9 conditions), messages built byte-identically to the paper's
token-causality run. Measurement: activation delta (condition minus no-anchor) at the last prompt
position, projected onto residualized J-lens directions for **blinded target tokens**
(`greenhouse glass bird birds sleeping crystal` — none appear in any anchor; `avian`/`wing` were
excluded precisely because they do) versus **control tokens** (the paper's negative-probe
subjects: `observatory salt clock tower beeswax`).

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

Three things worth noticing in that table:

1. **The ordering is the paper's behavioral ordering, re-derived at the activation layer.**
   Anchor > shuffled > collision ≈ random, with shuffled retaining roughly half — the same
   pattern the 2026 probe-margin experiments produced from pure black-box measurement. Two
   independent instruments, two years apart, one mechanism.
2. **The blinded anchor reaches ~38% of the explicit-mention ceiling without ever saying the
   words.** That is the channel: composed proxy evidence (`eagles velvet lumina…`) doing a
   measurable fraction of what literal naming does.
3. **The effect lives at layers 40–41 — the layers Figure 6 flagged in April from a completely
   different diagnostic** (class-vector alignment), before Anthropic's paper existed. The
   activation geometry was already pointing here.

Robustness: the layer window chosen *only* from the explicit-mention instrument controls (never
from anchor conditions) yields the identical result (+0.0151, CI [+0.0115, +0.0188]); all 8
anchors and all 5 probe prompts are individually positive.

## Method notes (read before quoting)

- **J-lens construction:** per probe token, the gradient of that token's last-position logit with
  respect to every hidden state, averaged over 32 neutral contexts — a modest-N approximation of
  Anthropic's thousands-context construction. The raw vectors share a dominant common component
  (mean inter-token cosine 0.70 — a generic "emit a noun" direction); each vector is
  **residualized** against the per-layer mean across probe tokens and renormalized (inter-token
  cosine → −0.10), which is what resolves token identity. Validated on the explicit-mention
  controls, never tuned on the anchor conditions.
- **Position and scope:** the readout is at the last prompt position, and loading appears in late
  layers (40–42, the workspace→motor boundary), not the mid-band — mid-band loading is null at
  this position even for explicit mentions. The supported sentence is therefore: *the anchor
  loads the target's verbalization directions pre-decode, without naming it* — not "writes into
  the mid-layer workspace." Multi-position readout (the workspace band while the model reads the
  anchor) is the natural follow-up, along with testing whether the subword/misspelling carrier
  channel loads composed concepts the single-token J-lens can't index.

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

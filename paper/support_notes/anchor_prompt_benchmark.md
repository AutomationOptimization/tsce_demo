# Anchor Prompt Benchmark

## Purpose
Rank anchor prompt families by the criteria that matter for TSCE:
- downstream pass rate,
- robustness against shuffled and matched-random controls,
- measurable hidden-state / logit movement,
- anchor quality under the selector.

## Benchmark protocol

Model:
- `google/gemma-4-E2B-it`

Runtime:
- Lambda GH200
- `--device cuda`
- `--dtype bfloat16`
- `--attn-implementation eager`

Task sample:
- `task-kind auto`
- smoke pass size: `n=8`

Anchor search settings:
- prompt families: `opaque_control`, `latent_bias`, `task_abstract`, `keyboard_drift`
- selector: `heuristic`
- candidate count: `8`
- anchor temperature: `1.25`
- top-p: `0.95`

Conditions:
- `baseline`
- `readable_control`
- `tsce_anchor`
- `random_matched`
- `shuffled_anchor`

## Primary decision rule

Prefer the family whose `tsce_anchor` rows:
- beat `random_matched` and `shuffled_anchor`,
- remain competitive with or better than `readable_control`,
- show non-zero hidden-state / logit movement,
- and maintain high selector-valid rate with low prompt overlap.

## Why this protocol is stricter than earlier passes
- It keeps the prior failures in scope instead of treating the current prompt rewrite as a fresh start.
- It compares multiple prompt families in one run.
- It records candidate-level selector metadata instead of only saving the chosen anchor text.
- It evaluates both behavioral and mechanistic signal.

## Results
Run directory:
- `tsce_agent_demo/results/gemma_family_compare_e2b_smoke_2026-04-21`

### Overall pooled summary
- `baseline`: `37.5%`
- `readable_control`: `50.0%`
- `tsce_anchor`: `34.4%`
- `random_matched`: `28.1%`
- `shuffled_anchor`: `37.5%`

Interpretation:
- pooled across all families, opaque TSCE anchors are not yet better than readable controls
- pooled TSCE anchors do beat matched-random controls
- pooled TSCE anchors do not beat shuffled controls, so the conservative overall claim remains `not_yet`

### Family-by-family summary

| Prompt Family | Baseline | Readable | TSCE | Random | Shuffled | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `latent_bias` | 37.5% | 50.0% | 50.0% | 37.5% | 37.5% | Best family. Beats both opaque controls. Matches readable control on this smoke pass. |
| `task_abstract` | 37.5% | 50.0% | 37.5% | 25.0% | 50.0% | Mixed. Some useful anchors, but still too semantically legible and one invalid candidate slipped through before the final selector patch. |
| `opaque_control` | 37.5% | 50.0% | 25.0% | 25.0% | 37.5% | Too much readable/control vocabulary leakage. |
| `keyboard_drift` | 37.5% | 50.0% | 25.0% | 25.0% | 25.0% | Collapsed into literal task words like `shift`, `focus`, `valid`, `path`. |

### Best family
`latent_bias`

Why it won:
- highest `tsce_anchor` pass rate: `50.0%`
- perfect selector valid rate: `1.0`
- zero prompt overlap in the selected anchors
- compact average length: `4.5` whitespace tokens
- non-zero logit and attention movement relative to baseline

Per-family claim status:
- `latent_bias`: `supports_human_opaque_control = yes`
- `latent_bias`: `supports_soft_prompt_wording = not_yet`

That is the right read for this smoke pass. The best opaque family can outperform the opaque controls, but it still does not beat readable controls.

### Representative selected anchors

`latent_bias`:
- `<HDA>fluxdrift nexus voidkern cipherwisp tranceleap</HDA>`
- `<HDA>synapseflux nexusvortex cypherdrift echoform</HDA>`
- `<HDA>fluxgle riftspun pyroxal gleamwr</HDA>`

`opaque_control` failure mode:
- `<HDA>shift nexus inference closure premise negation antecedent</HDA>`
- `<HDA>shift nexus locus proxy inertia subtle vector drift</HDA>`

`keyboard_drift` failure mode:
- `<HDA>shift focus state align date time slot clear</HDA>`
- `<HDA>shift next valid path</HDA>`

### What the smoke pass changed
- The best family was not the initial hybrid default `opaque_control`; it was the more synthetic `latent_bias` family.
- Strong anchors on Gemma 4 E2B are shorter than the prompt originally asked for.
- Compact pseudo-lexical handles worked better than abstract readable control words.
- The benchmark exposed a selector bug in `task_abstract`: one brace-wrapped invalid candidate was selected because it scored above valid candidates. The selector has since been patched to always prefer valid candidates when any are available.

### Current recommendation
- Make `latent_bias` the implementation default.
- Keep `opaque_control` only as a broader hybrid ablation, not as the preferred prompt.
- Tighten the synthetic family further around short pseudo-lexical codewords and remove remaining English leakage such as `nexus`, `shift`, and `flux` if a follow-up benchmark shows those tokens are acting as filler rather than signal.

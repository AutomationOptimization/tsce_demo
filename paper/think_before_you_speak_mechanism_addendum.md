# Mechanistic Addendum to Think Before You Speak

## Point of the paper
`docs/Think_Before_You_Speak.pdf` argues that TSCE improves generation by inserting a hidden, task-aware token intervention before the user-visible answer. The paper's mechanism is not that the anchor is readable prose, and not that it can be replaced by an arbitrary hand-injected activation vector. The mechanism is two-step conditioning:

1. Phase 1 produces an ESCP/HDA anchor.
2. Phase 2 conditions on that anchor before decoding the answer.
3. The anchor contracts or tilts the conditional distribution into a narrower semantic basin.

In the paper's language, the ESCP "actively shapes the model's early activations" and acts as a "persistent attractor." The corrected proof target is therefore:

> TSCE anchors are token-sequence interventions that, without explicitly naming the target, shift the model's conditional distribution toward a hidden semantic basin. This shift is measurable before long-form generation.

## Mechanistic nature of the claim
The paper is not only making a behavioral claim that two calls outperform one call. It is making a mechanistic claim about why the second call changes: the anchor is a causal input intervention that changes the model's pre-decode conditional state.

That means the strongest evidence should have four properties:

1. It measures something before long-form generation, not only final answer quality.
2. It compares the real anchor to no-anchor and matched random-token controls.
3. It perturbs the anchor tokens directly through shuffling, deletion, and replacement.
4. It shows that target-proxy token composition predicts the direction or size of the distribution shift.

The greenhouse token-causality bench has exactly that shape. Its dependent variable is a target-vs-distractor probe margin computed before story generation, so the result is evidence about the model's conditional distribution, not just about whether a generated story happened to pass a rubric.

## What the greenhouse experiment adds
The controlled greenhouse experiment tests the paper's central mechanistic claim at the right level. It does not ask whether a mean activation delta can be patched into the model. It asks whether ordinary anchor tokens, passed through the model's normal input channel, change next-token/probe probabilities before visible story generation.

Target concept:

> a greenhouse full of sleeping glass birds

Completed artifact:

`tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3/`

Core result:

- `no_anchor` margin: `-3.7140`
- full controlled anchors mean lift: `+1.1629`
- shuffled anchors mean lift: `+0.9398`
- head/mid/tail deletion lifts: `+0.8332`, `+1.0441`, `+1.0766`
- tokenizer-matched random controls: `-0.0593`, `-0.0011`, `-0.0231`
- neutral anchors with zero proxy coverage: `-0.8351`

This is direct evidence for the Think Before You Speak mechanism: the anchor changes the model's conditional distribution before it speaks. The target phrase is not placed in the anchor, yet the positive target probes gain probability relative to negative story-subject probes.

That distinction matters. A behavioral benchmark can show TSCE works. This bench shows the token intervention is doing work at the distributional layer the paper claims to affect.

## Why this is stronger than vector patching
The failed patch-in recovery runs tested a different hypothesis: whether a portable mean activation delta can replace the token-sequence mechanism. That is not required by the paper. TSCE is model-agnostic precisely because the intervention is a token sequence passed through the native token pathway.

The corrected interpretation is:

- The anchor itself is the intervention.
- The intervention works through normal token processing.
- The measurable object is conditional distribution shift, not successful transplantation of a detached vector.
- Activation patching remains a secondary diagnostic, not the core proof.

## What the proxy-family controls show
The controlled grid fixed length and fixed total proxy-token count for all non-neutral anchors, then varied semantic proxy-family composition:

- `0` family coverage: `-0.8351`
- `1` family coverage: `+0.8271`
- `2` family coverage: `+1.4721`
- `3` family coverage: `+1.3424`
- `4` family coverage: `+1.9312`

This supports the broad paper claim that anchors work by activating a latent basin through distributed token evidence. It also refines the claim: coverage is predictive, but not monotonic under fixed proxy budget. Composition matters.

The strongest controlled group was `glass_light + avian_sky` at `+2.2404`, above `all_four` at `+1.9312`. Family-presence deltas also show that `glass_light` and `avian_sky` are the most important proxy neighborhoods for this target:

- `glass_light`: `+0.7247`
- `avian_sky`: `+0.7078`
- `garden_green`: `+0.3247`
- `sleep_quiet`: `+0.1412`

The paper should therefore avoid saying "more proxy families always cause more lift." The defensible claim is sharper:

> TSCE anchors steer by composing indirect proxy neighborhoods that move the model into the target basin; both coverage and family geometry matter.

## How to state this in the paper
Use this framing:

> The ESCP need not name the answer or encode a human-readable instruction. In controlled token-causality experiments, anchors composed of indirect semantic proxies shift pre-generation probe margins toward a hidden target subject, while tokenizer-matched random controls do not. This supports the view of TSCE as token-mediated latent conditioning: the anchor acts as a hard-token attractor that contracts the model's conditional distribution before visible decoding.

For the mechanistic claim, use this tighter sentence:

> The evidence is mechanistic rather than merely behavioral because the measured effect occurs before long-form decoding, survives token-order and deletion perturbations, and disappears under tokenizer-matched random controls.

Avoid these framings:

- "The anchor is just a vector we can extract and patch elsewhere."
- "Any semantically weird token soup works."
- "Exact token order is the carrier."
- "Coverage alone explains the effect."

The empirical story is closer to an equivalence class: shuffling and deletion often preserve lift because several token sequences land in the same latent basin, but matched random strings do not.

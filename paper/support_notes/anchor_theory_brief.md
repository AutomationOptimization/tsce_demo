# Anchor Theory Brief

## Bottom line
TSCE anchors should be treated as discrete control codes.

Every token in a prompt perturbs hidden states, so the phrase "latent control" is not special because anchors somehow bypass normal token processing. The meaningful distinction is the objective: a TSCE anchor is authored purely to steer the model's internal trajectory before the visible answer, not to communicate a readable instruction to a human. In that sense, TSCE is closer to a hard-token control prefix than to ordinary prompting.

This connects directly to `docs/Think_Before_You_Speak.pdf`: the paper's central claim is that the ESCP/HDA anchor is a hidden token intervention that contracts or tilts the conditional distribution before the model speaks. The anchor is the intervention. It should not be treated merely as a way to extract a portable activation vector.

## What anchors are
- A short token sequence inserted to bias the second pass toward a narrower, more useful region of the model's existing behavior.
- A discrete approximation to a soft prompt or prefix.
- An inference-time intervention that should be judged by downstream task lift plus measurable internal movement.

## What anchors are not
- Not a readable task summary.
- Not roleplay, metaphor theater, sigils, or symbol soup.
- Not direct activation injection.
- Not proof that hard tokens are as expressive as arbitrary continuous activation edits.

## Relation to prior theory

### 1. Prefix tuning / soft prompts
Prefix tuning and prompt tuning show that a frozen model can be steered by prepended task-specific vectors that behave like virtual tokens. This is the cleanest prior for TSCE: the anchor is trying to occupy a similar control role, but inside the native token alphabet instead of a continuous prefix.

Practical implication:
- TSCE anchors are best understood as hard-token control prefixes.
- Continuous prefixes remain more expressive than hard tokens, so TSCE should be framed as an approximation, not as an unrestricted substitute for direct latent intervention.

### 2. Theory of prompting / prefix tuning
Recent theory argues that prompting and prefix tuning can bias outputs and elicit capabilities already present in the pretrained model, but cannot create new attention patterns the way full weight updates can. That maps well onto TSCE. A good anchor should not be expected to grant new reasoning primitives; it should bias selection among behaviors the model already knows how to execute.

Practical implication:
- The anchor should compress search over existing capabilities.
- Good evaluation targets are variance reduction, formatting reliability, and error-mode suppression, not "new skill creation."

### 3. Discrete prompt optimization
Universal adversarial triggers and AutoPrompt both show that short discrete token sequences can strongly steer model behavior even when the resulting strings are opaque, input-agnostic, or not human-authored. This matters because it removes the main objection to TSCE on theoretical grounds: ordinary tokens can absolutely function as control handles rather than human-readable instructions.

Practical implication:
- Opaque anchors are not automatically "crap anchors."
- Search quality matters more than human readability.

### 4. Activation engineering
Activation engineering is the stronger intervention class: it perturbs hidden activations directly during the forward pass. TSCE does not do that. Instead, it uses tokens to induce a hidden-state shift upstream through the standard model interface.

Practical implication:
- TSCE should be described as token-mediated activation steering, not direct activation editing.
- The right mechanistic test is whether the anchor causes stable hidden-state and logit movement relative to controls.

### 5. Opaque autoprompt analysis
Recent work on machine-generated opaque prompts suggests these prompts are often not uniformly meaningful. They tend to contain a small number of influential tokens, some filler, and many tokens that can be pruned without losing most of the effect. That aligns with the repo history: shorter anchors consistently outperform long, overloaded ones.

Practical implication:
- Do not reward length for its own sake.
- Favor compact anchors with a few high-leverage tokens and minimal filler.

## Local empirical evidence from this repo

The repo's own anchor analysis is more informative than generic prompt-engineering advice.

### Greenhouse token-causality result

The strongest current mechanistic evidence is the greenhouse subject artifact:

`tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3/`

Target concept:

`a greenhouse full of sleeping glass birds`

This experiment tests the corrected Think Before You Speak proof target: an anchor token sequence shifts pre-generation probe probabilities toward a hidden semantic basin without naming the target phrase in the anchor.

This matters because the paper's claim is mechanistic, not merely behavioral. Final story quality can show downstream utility, but a pre-generation probe-margin shift shows the anchor changed the conditional distribution before the model produced the visible answer.

Key results:
- `no_anchor` margin: `-3.7140`
- controlled anchors mean lift: `+1.1629`
- shuffled anchors mean lift: `+0.9398`
- head/mid/tail deletion lifts: `+0.8332`, `+1.0441`, `+1.0766`
- tokenizer-matched random controls: `-0.0593`, `-0.0011`, `-0.0231`
- zero-proxy neutral anchors: `-0.8351`

Controlled proxy-family coverage, with total proxy-token count held fixed:
- `0` families: `-0.8351`
- `1` family: `+0.8271`
- `2` families: `+1.4721`
- `3` families: `+1.3424`
- `4` families: `+1.9312`

The important refinement is that family composition matters. The strongest group was `glass_light + avian_sky` at `+2.2404`, above `all_four` at `+1.9312`. So the defensible paper claim is not "more proxy families always monotonically improve lift." It is:

> TSCE anchors steer by composing indirect proxy neighborhoods that move the model into the target basin; both distributed coverage and family geometry matter.

This is a better fit to the paper than vector patching. Patch-in recovery tests whether a detached mean activation delta can replace the token pathway. TSCE's actual claim is token-mediated conditioning through the normal model interface.

From `tsce_agent_demo/anchor_analysis/2025-09-10_00-18-34/`:
- `n=201`, `ok=140`, `ok_rate=69.7%`.
- Successful anchors averaged `9.94` whitespace tokens; failures averaged `11.74`.
- Token-count bucket success rates were:
  - `<=6`: `89.4%`
  - `7-10`: `70.4%`
  - `11-14`: `55.2%`
  - `15+`: `66.7%`
- Anchors with repetition ratio `<=0.10` reached `80.2%` success.
- Anchors with alpha ratio `>=0.90` reached `85.7%` success.
- Historical winners used no digits and no punctuation.

Representative winning anchors:
- `qvthric xrvane dkzlor thqvane vrqen xrlith qvdkz xrvith lorthic vqzric qvthqen`
- `qwertyui asdfghjk lkjhgfds zxcvbnmq poiuytre asdfghjkl qwertyuiop`
- `qvthxk vqzjrc lqvthg jxkqen rjvane`

These winners share four properties:
- short to mid-length,
- mostly alphabetic,
- low repetition,
- semantically opaque or pseudo-lexical.

They do not look like the older HDAG prompt wanted them to look. The old prompt emphasized metaphor density, symbol density, long length, and visual novelty. The repo's best anchors overwhelmingly contradict that design.

## Working theory for prompt design

The anchor prompt should optimize for this shape:
- compact,
- mostly alphabetic,
- low prompt overlap,
- low repetition,
- non-sentential,
- opaque enough that the text is clearly serving control rather than explanation.

That leads to a practical design rule:
- ask the model to internally search for several candidates,
- explicitly ban readable summaries and instruction verbs,
- target roughly `5-10` whitespace tokens for the default family,
- then select by heuristic quality rather than trusting a single sample.

## Strong claims we can defend
- TSCE anchors are a legitimate form of token-level steering.
- They are best framed as discrete control prefixes or hard-token soft-prompt approximations.
- Opaque surface form is not incidental; it is a direct consequence of optimizing for steering instead of human readability.
- Historical evidence in this repo favors short, alphabetic, low-repetition anchors and disfavors long symbolic constructions.
- Controlled greenhouse evidence is mechanistic: anchors shift hidden target probe margins before long-form generation, while tokenizer-matched random controls do not.
- Good anchors are better understood as an equivalence class of indirect token sequences that land in a latent basin, not as exact strings whose order must be preserved.

## Claims we should not overstate
- We do not have grounds to claim that hard-token anchors are fully equivalent to arbitrary direct activation interventions.
- We do not have grounds to claim that every token in an opaque anchor contributes equally.
- We should not assume human unreadability alone implies a good anchor; downstream lift and internal movement still decide.
- We should not claim that proxy-family coverage is strictly monotonic; the greenhouse grid shows coverage helps, but family composition can dominate.
- We should not make vector-patching recovery the main proof target; it tests a stronger and different hypothesis than the paper requires.

## References

Local repo references:
- `docs/Think_Before_You_Speak.pdf`
- `docs/think_before_you_speak_mechanism_addendum.md`
- `docs/phase1_diffusion_anchor.md`
- `docs/rl_hda.md`
- `docs/diffusion_reward_loss_map.md`
- `tsce_agent_demo/results/story_subject_greenhouse_v1/controlled_proxy_family_grid_v1/anchor_token_causality_random3/property_analysis/anchor_token_property_analysis.md`
- `tsce_agent_demo/anchor_analysis/2025-09-10_00-18-34/report.md`
- `tsce_agent_demo/anchor_analysis/2025-09-10_00-18-34/tables/token_uplift_overall.csv`
- `tsce_agent_demo/anchor_analysis/2025-09-10_00-18-34/tables/anchors_features.csv`

External references:
- Li and Liang, 2021, "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
- Lester et al., 2021, "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Petrov et al., 2024, "When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations"
- Wallace et al., 2019, "Universal Adversarial Triggers for Attacking and Analyzing NLP"
- Shin et al., 2020, "AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts"
- Turner et al., 2024, "Steering Language Models with Activation Engineering"
- Genewein et al., 2025, "Understanding Prompt Tuning and In-Context Learning via Meta-Learning"
- Rakotonirina et al., 2025, "Evil twins are not that evil: Qualitative insights into machine-generated prompts"

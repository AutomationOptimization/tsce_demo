# Anchor Prompt Spec

## Objective
Generate a TSCE anchor that behaves like a discrete soft prompt: strong enough to steer the second pass, compact enough to avoid filler, and opaque enough that the text is clearly serving control rather than human-readable instruction.

## Empirical constraints from repo history

From the repo's anchor-analysis pass (`n=201`):
- Overall success rate: `69.7%`
- Successful anchors mean token count: `9.94`
- Failed anchors mean token count: `11.74`
- `<=6` tokens was the strongest bucket at `89.4%`
- Repetition ratio `<=0.10` improved success to `80.2%`
- Alpha ratio `>=0.90` improved success to `85.7%`
- Historical winners used no digits and no punctuation

This means the target is not "more elaborate." The target is compact, clean, low-overlap control text.

## Prompt families

### `opaque_control`
Hybrid family.

Target shape:
- `5-10` word-like control tokens
- mostly alphabetic
- can mix rare abstract words, mutated compounds, and synthetic lexemes
- must feel unordered and non-sentential

Why it exists:
- best match to the historical winners
- broad enough to let Gemma land on either pseudo-lexical or oblique-real-word anchors

### `latent_bias`
Synthetic family. Current default.

Target shape:
- `4-8` invented alphabetic codewords
- pseudo-lexical strings, keyboard-adjacent fragments, or plausible nonsense
- minimal ordinary English

Why it exists:
- stress-tests the strongest interpretation of anchors as explicit latent handles
- closest to the historical `qvthric`-style winners
- best-performing family in the Gemma 4 E2B smoke benchmark

### `task_abstract`
Abstract family.

Target shape:
- `6-10` rare nouns, adjectives, or oblique compounds
- can use real words
- must remain indirect, unordered, and non-sentential

Why it exists:
- allows semantic bias without collapsing into a readable summary
- useful when pure pseudo-lexeme prompts drift too far from the task manifold

### `keyboard_drift`
Keyboard family.

Target shape:
- `5-9` alphabetic strings that resemble keyboard drift or key-neighbor clusters
- intentional and varied, not random smash

Why it exists:
- directly tests the historical `qwertyui` / `asdfghjk` pattern seen in the repo
- useful as a separate ablation because the surface form is unusually specific

## Selector / validator

Anchors are not accepted on a single draw. The backend now:
- samples multiple candidates per prompt,
- canonicalizes them into `<HDA>...</HDA>`,
- extracts feature metadata,
- scores each candidate,
- and selects the best one heuristically.

### Features used by the selector
- token count
- unique token count
- repetition ratio
- average token length
- alpha ratio
- prompt-overlap count
- control-word overlap count
- function-word count
- non-alpha character count
- symbol-only token count
- lowercase violations
- readable-clause heuristic
- validity flag plus invalid reasons

### Hard failure patterns
These make a candidate invalid:
- copied prompt words
- control or instruction words such as `answer`, `compute`, `schema`, `output`
- function words that make the anchor read like a clause
- digits
- non-alpha characters
- symbol-only tokens
- obvious sentence punctuation
- repetition above the `>2` token-repeat limit

### Scoring priorities
The selector rewards:
- landing in the family token window,
- high alpha ratio,
- high uniqueness,
- compactness,
- low readability.

It penalizes:
- prompt overlap,
- instruction vocabulary,
- repetition,
- punctuation or symbols,
- sentence-like structure,
- extra filler length.

## Banned patterns
- readable task summaries
- direct instruction verbs
- answer leakage
- copied schema words
- markdown, JSON, XML, LaTeX, braces, hashes, slashes
- digits
- quotes
- roleplay framing
- symbolic or glyph-heavy constructions

## Good anchor shape
Examples from repo history:
- `qvthric xrvane dkzlor thqvane vrqen xrlith`
- `qwertyui asdfghjk zxcvbnml qwertyop asdfghqz zxcvbnqw`
- `qvthxk vqzjrc lqvthg jxkqen rjvane`

Recent acceptable Gemma outputs after the prompt rewrite:
- `abjectly nexus flux cipher mimicry parallax subtle tethering metric frame`
- `inundation deluge fraction dispersal quantum metric nexus aperture parallax`

The second class is weaker than the historical pseudo-lexeme winners because the words are still semantically legible, but they are directionally correct: compact, low-overlap, and non-sentential.

## Bad anchor shape
- `compute schema validate internal structure output format`
- long metaphor-dense sigil strings
- any string with copied task nouns or answer tokens
- long constructions whose only purpose is to satisfy an arbitrary minimum length

## Default knobs

Current implementation defaults:
- prompt family: `latent_bias`
- selector: `heuristic`
- candidate count: `6`
- anchor temperature: `1.25`
- top-p: `0.95`

Benchmark setting:
- candidate count: `8`
- prompt families: `all`

## Practical rule
If an anchor looks like a clever mini-instruction, it is probably wrong.

If it looks like compact, clean, low-overlap control text and survives the selector, it is at least shaped correctly and deserves downstream evaluation.

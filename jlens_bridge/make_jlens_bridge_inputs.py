#!/usr/bin/env python3
"""Build bridge_inputs.json for jlens_bridge.py from the paper's greenhouse token-causality artifact.

Pulls the EXACT anchor strings the paper run used (per anchor_key x probe x condition) so the
J-lens bridge measures the same interventions Figure/Table token-causality reported, on the same
model (google/gemma-4-E4B-it), with the same matched-control scaffold.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent.parent / (
    "results/tsce_agent_demo/results/story_subject_greenhouse_v1/"
    "controlled_proxy_family_grid_v1/anchor_token_causality_random3"
)
OUT = Path(__file__).resolve().parent / "bridge_inputs.json"

# Headline group (full 4-family coverage) + the strongest pair group from the proxy-family grid.
ANCHOR_KEYS = [
    "all_four_001", "all_four_002", "all_four_003", "all_four_004",
    "coverage_2_glass_light_avian_sky_001", "coverage_2_glass_light_avian_sky_002",
    "coverage_2_glass_light_avian_sky_003", "coverage_2_glass_light_avian_sky_004",
]
CONDITIONS = [
    "no_anchor", "anchor", "anchor_shuffled",
    "random_matched_anchor_1", "random_matched_anchor_2", "random_matched_anchor_3",
    "context_collision_anchor",
]

# Single-token status verified against the gemma-4-E4B-it tokenizer (262k vocab).
TARGET_TOKENS = [" greenhouse", "glass", "bird", "birds", "sleeping", "crystal"]  # avian/wing dropped: appear verbatim in anchors (echo, not inference)
CONTROL_TOKENS = [" observatory", "salt", "clock", "tower", " beeswax"]  # ledger dropped: appears in a control anchor

# Diverse neutral contexts for J-lens averaging (modest stand-in for the paper's
# "average over thousands of contexts"; none evoke targets or controls).
JLENS_CONTEXTS = [
    "Explain how to file a tax extension in one paragraph.",
    "What is a good beginner routine for learning the piano?",
    "Summarize the rules of chess for a child.",
    "Give me a recipe idea using lentils and rice.",
    "How do I politely decline a meeting invitation?",
    "Describe the water cycle briefly.",
    "What should I check before buying a used car?",
    "Explain the difference between RAM and storage.",
    "Write a two-sentence product description for a backpack.",
    "How does compound interest work?",
    "Suggest three stretches for lower back pain.",
    "What is the capital of Australia and one fact about it?",
    "Explain what an API is to a non-programmer.",
    "Give tips for keeping houseplants alive in winter.",
    "How do I convert a PDF to a text file?",
    "Describe how vaccines train the immune system.",
    "What makes sourdough different from regular bread?",
    "Outline a simple monthly budget template.",
    "How should I prepare for a phone interview?",
    "Explain time zones in two sentences.",
    "What are common symptoms of dehydration?",
    "Suggest a weekend itinerary for a small mountain town.",
    "How does a refrigerator keep food cold?",
    "Write a polite reminder email about an unpaid invoice.",
    "What is the difference between a debit and credit card?",
    "Explain photosynthesis at a middle-school level.",
    "Give three tips for improving sleep quality.",
    "How do I set up a new email signature?",
    "Describe the plot structure of a typical mystery novel.",
    "What tools do I need to change a bicycle tire?",
    "Explain inflation using a grocery store example.",
    "How can I reduce phone screen time?",
]


def main() -> None:
    rows = [json.loads(l) for l in (ARTIFACT / "token_causality_rows.jsonl").read_text().splitlines() if l.strip()]
    records = {}
    for r in rows:
        key = r["anchor_key"]
        if key not in ANCHOR_KEYS or r["condition"] not in CONDITIONS:
            continue
        rid = f"{key}__{r['probe_id']}"
        rec = records.setdefault(rid, {
            "anchor_key": key,
            "probe_id": r["probe_id"],
            "probe_prompt": r["probe_prompt"],
            "conditions": {},
        })
        rec["conditions"][r["condition"]] = r.get("anchor_text") or None
    recs = sorted(records.values(), key=lambda x: (x["anchor_key"], x["probe_id"]))
    missing = [
        (r["anchor_key"], r["probe_id"], c)
        for r in recs for c in CONDITIONS if c not in r["conditions"]
    ]
    if missing:
        raise SystemExit(f"missing conditions: {missing[:5]} (+{len(missing)-5} more)" if len(missing) > 5 else f"missing conditions: {missing}")
    out = {
        "model": "google/gemma-4-E4B-it",
        "source_artifact": str(ARTIFACT),
        "target_tokens": TARGET_TOKENS,
        "control_tokens": CONTROL_TOKENS,
        "jlens_contexts": JLENS_CONTEXTS,
        "records": recs,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(recs)} records x {len(CONDITIONS)} conditions, "
          f"{len(TARGET_TOKENS)} target / {len(CONTROL_TOKENS)} control tokens, {len(JLENS_CONTEXTS)} jlens contexts")


if __name__ == "__main__":
    main()

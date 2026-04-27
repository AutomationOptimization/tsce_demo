#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "tsce-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ARTIFACT_ROOT / "results"
OUT_DIR = ARTIFACT_ROOT / "paper" / "figures" / "tsce"

BEHAVIORAL_SUMMARY = RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-300-tsce_gpt-35-turbo" / "summary_stats.md"
GREENHOUSE_ANALYSIS = (
    RESULTS_ROOT
    / "tsce_agent_demo"
    / "results"
    / "story_subject_greenhouse_v1"
    / "controlled_proxy_family_grid_v1"
    / "anchor_token_causality_random3"
    / "property_analysis"
    / "anchor_token_property_analysis.json"
)
GEMMA_CLASS_ACTIVATION = (
    RESULTS_ROOT
    / "tsce_agent_demo"
    / "results"
    / "gemma_latent_zone_v1_20260424_progress2"
    / "symbolic_transform_json"
    / "class_activation_summary.json"
)


METHOD_LABELS = {
    "baseline": "Baseline",
    "cot": "CoT",
    "refine": "Self-Refine",
    "tsce": "TSCE",
    "cot+tsce": "CoT+TSCE",
    "refine+tsce": "Refine+TSCE",
}

SUMMARY_RUNS = [
    ("Run 200-A", RESULTS_ROOT / "archive" / "2025-05-30_17-08-01" / "summary_stats.md"),
    ("Run 200-B", RESULTS_ROOT / "archive" / "2025-05-30_17-13-11" / "summary_stats.md"),
    ("Run 1000", RESULTS_ROOT / "archive" / "2025-05-30_22-14-31" / "summary_stats.md"),
    ("GPT low-temp", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-gpt-lowt" / "summary_stats.md"),
    ("GPT no-HDA", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-gpt-nohda" / "summary_stats.md"),
    ("GPT p2temp1", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-gpt-vart-p2temp1" / "summary_stats.md"),
    ("GPT p2top0", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-gpt-vart-p2tempt1top0" / "summary_stats.md"),
    ("GPT 100", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-tsce-gpt" / "summary_stats.md"),
    ("Llama 8B", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-100-tsce_llama-8b" / "summary_stats.md"),
    ("GPT 300", RESULTS_ROOT / "tsce_agent_demo" / "results" / "results-300-tsce_gpt-35-turbo" / "summary_stats.md"),
    ("GPT temp 0.01", RESULTS_ROOT / "tsce_agent_demo" / "results" / "resuts-100-gpt-vart-0.01top" / "summary_stats.md"),
    ("GPT temp 1", RESULTS_ROOT / "tsce_agent_demo" / "results" / "resuts-100-gpt-vart-1top" / "summary_stats.md"),
    ("Diffusion 30", RESULTS_ROOT / "tsce_agent_demo" / "results" / "tsce_diffusion_2025-12-03_10-55-38" / "summary_stats.md"),
    ("Diffusion 300a", RESULTS_ROOT / "tsce_agent_demo" / "results" / "tsce_diffusion_2025-12-03_11-03-35" / "summary_stats.md"),
    ("Diffusion 300b", RESULTS_ROOT / "tsce_agent_demo" / "results" / "tsce_diffusion_2025-12-03_11-17-13" / "summary_stats.md"),
    ("TSCE eval", RESULTS_ROOT / "tsce_agent_demo" / "results" / "tsce_eval" / "summary_stats.md"),
    ("Deterministic eval", RESULTS_ROOT / "tsce_agent_demo" / "results" / "tsce_eval_deterministic" / "summary_stats.md"),
]


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _parse_pass_table(path: Path) -> List[Tuple[str, int, int, float]]:
    rows: List[Tuple[str, int, int, float]] = []
    pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)/(\d+)\s*\|")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        method = match.group(1).strip()
        passed = int(match.group(2))
        total = int(match.group(3))
        rows.append((method, passed, total, passed / total))
    return rows


def _best_tsce_row(rows: List[Tuple[str, int, int, float]]) -> Tuple[str, int, int, float]:
    candidates = [
        row
        for row in rows
        if row[0].lower() != "baseline" and ("tsce" in row[0].lower() or "diff" in row[0].lower())
    ]
    if not candidates:
        raise ValueError("no TSCE/HDA candidate rows found")
    return max(candidates, key=lambda row: row[3])


def _bar_labels(ax: plt.Axes, bars: Iterable[plt.Rectangle], fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_figure_1() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.set_axis_off()
    nodes = [
        ("User prompt\n$X$", 0.10, 0.58),
        ("Phase 1\ngenerate HDA $A$", 0.33, 0.58),
        ("Pre-decode\nconditional shift", 0.58, 0.58),
        ("Phase 2\ngenerate answer $Y$", 0.84, 0.58),
    ]
    colors = ["#e8f1f2", "#f4d35e", "#ee964b", "#d9ead3"]
    for (label, x, y), color in zip(nodes, colors):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=color, edgecolor="#1d3557", linewidth=1.2),
        )
    for left, right in zip(nodes, nodes[1:]):
        ax.annotate(
            "",
            xy=(right[1] - 0.105, right[2]),
            xytext=(left[1] + 0.105, left[2]),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#1d3557"),
        )
    ax.text(
        0.58,
        0.21,
        "Mechanistic test: perturb HDA tokens and measure probe/logit movement before long-form decoding.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "figure1_token_intervention")


def build_figure_2() -> None:
    rows = _parse_pass_table(BEHAVIORAL_SUMMARY)
    order = ["baseline", "cot", "refine", "tsce", "cot+tsce", "refine+tsce"]
    by_method = {method: (passed, total, rate) for method, passed, total, rate in rows}
    labels = [METHOD_LABELS[item] for item in order]
    rates = [100.0 * by_method[item][2] for item in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = ["#8d99ae", "#a8dadc", "#a8dadc", "#e76f51", "#f4a261", "#f4a261"]
    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, rates, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar in bars:
        width = float(bar.get_width())
        ax.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.0f}%", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pass rate")
    ax.set_xlim(0, 88)
    ax.set_title("Black-box reliability: 300-task GPT-3.5 evaluation")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, "figure2_behavioral_pass_rates")


def _greenhouse() -> Dict[str, object]:
    return json.loads(GREENHOUSE_ANALYSIS.read_text(encoding="utf-8"))


def build_figure_3() -> None:
    data = _greenhouse()
    summary = data["condition_summary"]
    condition_order = [
        ("anchor", "HDA"),
        ("anchor_shuffled", "Shuffled"),
        ("anchor_head_deleted", "Head del."),
        ("anchor_mid_deleted", "Mid del."),
        ("anchor_tail_deleted", "Tail del."),
        ("anchor_neutral_replaced", "Neutral repl."),
        ("random_matched_anchor_1", "Rand 1"),
        ("random_matched_anchor_2", "Rand 2"),
        ("random_matched_anchor_3", "Rand 3"),
    ]
    labels = [label for _, label in condition_order]
    values = [float(summary[key]["mean_lift"]) for key, _ in condition_order]
    colors = ["#e76f51"] * 6 + ["#8d99ae"] * 3

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, values, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    for bar in bars:
        width = float(bar.get_width())
        label_x = width + 0.03 if width >= 0 else width - 0.03
        ha = "left" if width >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va="center", ha=ha, fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Margin lift vs. no HDA")
    ax.set_xlim(-0.2, 1.3)
    ax.set_title("Mechanistic token-causality controls")
    ax.grid(axis="x", alpha=0.22)
    _save(fig, "figure3_greenhouse_condition_lifts")


def build_figure_4() -> None:
    data = _greenhouse()
    coverage = data["coverage_summary"]
    xs = sorted(int(key) for key in coverage.keys())
    y = [float(coverage[str(x)]["mean_lift"]) for x in xs]

    family_summary = data["family_group_summary"]
    top_groups = sorted(
        ((name, float(stats["mean_lift"])) for name, stats in family_summary.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    top_labels = [name.replace("coverage_2_", "").replace("coverage_3_", "").replace("_", "+") for name, _ in top_groups]
    top_values = [value for _, value in top_groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1, 1.8]})
    ax1.plot(xs, y, marker="o", color="#1d3557", linewidth=2)
    ax1.axhline(0.0, color="#222222", linewidth=0.8)
    for x, value in zip(xs, y):
        ax1.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(xs)
    ax1.set_xlabel("Proxy-family coverage")
    ax1.set_ylabel("Mean lift")
    ax1.set_title("Coverage effect\n(non-monotonic)")
    ax1.grid(axis="y", alpha=0.22)

    bars = ax2.barh(top_labels[::-1], top_values[::-1], color="#f4a261", edgecolor="#222222", linewidth=0.6)
    for bar in bars:
        width = float(bar.get_width())
        ax2.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.2f}", va="center", fontsize=8)
    ax2.set_xlabel("Mean lift")
    ax2.set_title("Proxy-family effect")
    ax2.grid(axis="x", alpha=0.22)
    _save(fig, "figure4_proxy_family_effects")


def build_figure_7() -> None:
    run_rows = []
    for label, path in SUMMARY_RUNS:
        rows = _parse_pass_table(path)
        baseline = next(row for row in rows if row[0].lower().startswith("baseline"))
        best = _best_tsce_row(rows)
        run_rows.append((label, baseline, best, 100.0 * (best[3] - baseline[3])))

    run_rows.sort(key=lambda item: item[3])
    labels = [item[0] for item in run_rows]
    baseline_rates = [100.0 * item[1][3] for item in run_rows]
    best_rates = [100.0 * item[2][3] for item in run_rows]
    lifts = [item[3] for item in run_rows]
    ypos = np.arange(len(run_rows))

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    for y, base, best, lift in zip(ypos, baseline_rates, best_rates, lifts):
        color = "#e76f51" if lift >= 0 else "#8d99ae"
        ax.plot([base, best], [y, y], color=color, linewidth=1.8, alpha=0.85)
    ax.scatter(baseline_rates, ypos, color="#8d99ae", edgecolor="#222222", s=38, label="Baseline", zorder=3)
    ax.scatter(best_rates, ypos, color="#e76f51", edgecolor="#222222", s=42, label="Best TSCE/HDA", zorder=4)
    ax.axvline(50, color="#222222", linewidth=0.6, alpha=0.35)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pass rate")
    ax.set_xlim(35, 101)
    ax.set_title("Cross-run black-box evaluations: baseline vs. best TSCE/HDA")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    positive = sum(1 for lift in lifts if lift > 0)
    median_lift = float(np.median(lifts))
    pooled_baseline_pass = sum(item[1][1] for item in run_rows)
    pooled_total = sum(item[1][2] for item in run_rows)
    pooled_best_pass = sum(item[2][1] for item in run_rows)
    ax.text(
        0.03,
        0.04,
        f"{positive}/{len(run_rows)} positive evals\nmedian lift {median_lift:+.1f} pp\npaired n={pooled_total:,}: "
        f"{100.0 * pooled_baseline_pass / pooled_total:.1f}% -> {100.0 * pooled_best_pass / pooled_total:.1f}%",
        fontsize=8.5,
        va="bottom",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.92),
    )
    _save(fig, "figure7_cross_run_lifts")


def build_figure_5() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    x = np.linspace(-3.2, 3.2, 220)
    y = np.linspace(-2.3, 2.3, 180)
    xx, yy = np.meshgrid(x, y)
    broad = np.exp(-((xx + 1.4) ** 2 / 2.8 + (yy + 0.1) ** 2 / 1.8))
    target = 1.35 * np.exp(-((xx - 1.25) ** 2 / 0.75 + (yy - 0.25) ** 2 / 0.5))
    distractor = 0.85 * np.exp(-((xx - 0.4) ** 2 / 0.9 + (yy + 1.05) ** 2 / 0.45))
    z = broad + target + distractor

    ax.contourf(xx, yy, z, levels=18, cmap="YlGnBu", alpha=0.82)
    ax.contour(xx, yy, z, levels=7, colors="#24435c", alpha=0.35, linewidths=0.6)
    ax.scatter([-1.55], [0.0], s=90, color="#8d99ae", edgecolor="#222222", zorder=4, label="No HDA")
    ax.scatter([1.25], [0.25], s=120, color="#e76f51", edgecolor="#222222", zorder=5, label="HDA-shifted")
    ax.scatter([-0.85], [-0.55], s=85, color="#6c757d", marker="x", linewidth=2.2, zorder=5, label="Matched random")
    ax.annotate("", xy=(1.05, 0.22), xytext=(-1.45, 0.02), arrowprops=dict(arrowstyle="->", lw=2.0, color="#e76f51"))
    ax.annotate("", xy=(-0.86, -0.52), xytext=(-1.45, 0.0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#6c757d", linestyle="--"))
    ax.text(1.37, 0.48, "target basin", fontsize=10, color="#1d3557", weight="bold")
    ax.text(0.52, -1.28, "distractor basin", fontsize=9, color="#1d3557")
    ax.text(-2.65, 1.42, "broad initial\nconditional state", fontsize=9, color="#1d3557")
    ax.set_title("Latent-basin view of HDA conditioning")
    ax.set_xlabel("activation direction 1")
    ax.set_ylabel("activation direction 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    _save(fig, "figure5_latent_basin_schematic")


def build_figure_6() -> None:
    data = json.loads(GEMMA_CLASS_ACTIVATION.read_text(encoding="utf-8"))
    layer_summary = data["layer_summary"]
    layers = sorted(int(key) for key in layer_summary.keys())
    good = np.array([float(layer_summary[str(layer)]["good_alignment"]) for layer in layers])
    control = np.array([float(layer_summary[str(layer)]["control_alignment"]) for layer in layers])
    shuffled = np.array([float(layer_summary[str(layer)]["shuffled_alignment"]) for layer in layers])
    separation = np.array([float(layer_summary[str(layer)]["separation"]) for layer in layers])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax1.plot(layers, good, color="#e76f51", linewidth=2, label="Good HDA")
    ax1.plot(layers, control, color="#6c757d", linewidth=1.7, label="Control")
    ax1.plot(layers, shuffled, color="#f4a261", linewidth=1.5, linestyle="--", label="Shuffled")
    ax1.axvspan(40, 41, color="#e76f51", alpha=0.12)
    ax1.set_title("Hidden-state class alignment")
    ax1.set_xlabel("Gemma layer")
    ax1.set_ylabel("Cosine alignment")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(fontsize=8, frameon=False)

    colors = np.where(separation >= 0.0, "#e76f51", "#8d99ae")
    ax2.bar(layers, separation, color=colors, width=0.82, edgecolor="#222222", linewidth=0.25)
    ax2.axhline(0, color="#222222", linewidth=0.8)
    ax2.axvspan(40, 41, color="#e76f51", alpha=0.12)
    ax2.set_title("Good HDA minus control")
    ax2.set_xlabel("Gemma layer")
    ax2.set_ylabel("Alignment separation")
    ax2.grid(axis="y", alpha=0.22)
    peak_layer = int(layers[int(np.argmax(separation))])
    peak_value = float(np.max(separation))
    ax2.text(peak_layer - 9, peak_value * 0.82, f"late-layer peak\nL{peak_layer}: +{peak_value:.3f}", fontsize=8, color="#333333")
    _save(fig, "figure6_activation_layer_separation")


def main() -> None:
    _ensure_out_dir()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    build_figure_1()
    build_figure_2()
    build_figure_3()
    build_figure_4()
    build_figure_5()
    build_figure_6()
    build_figure_7()
    print(f"[TSCE-FIGURES] wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

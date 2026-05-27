"""Variance-component decomposition figure.

Produces a stacked horizontal bar — one bar per benchmark — showing the
share of total variance contributed by each of the seven sources. This is
the visual answer to the project's core question: where does LLM-judge
score variance actually live?

Usage:
    python -m src.viz.variance_pie
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.gstudy import (
    BENCHMARKS,
    COMPONENTS,
    balanced_panel,
    fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

# Diagnostic interpretations from pre_analysis_plan.tex §3.1.
LABELS = {
    "m":     r"$\sigma^2_m$ (models)",
    "j":     r"$\sigma^2_j$ (judges)",
    "i":     r"$\sigma^2_i$ (items)",
    "mj":    r"$\sigma^2_{mj}$ (model$\times$judge)",
    "mi":    r"$\sigma^2_{mi}$ (model$\times$item)",
    "ji":    r"$\sigma^2_{ji}$ (judge$\times$item)",
    "mji_e": r"$\sigma^2_{mji,e}$ (residual)",
}
# Distinct hue per component so all seven are readable at a glance. Two
# pieces of structure are preserved deliberately:
#   - σ²_m (the signal) gets the saturated blue, which most readers
#     associate with the "good" variance.
#   - σ²_mj (the project's headline term) gets the only red, which
#     reads as "attention".
# The remaining five facets get distinct qualitative tab10 hues. Only the
# residual stays gray, which is the convention in variance-decomposition
# plots and lets a viewer immediately separate "structural" from "noise".
COLORS = {
    "m":     "#1f77b4",  # blue        — signal (highlighted)
    "j":     "#ff7f0e",  # orange      — judge main effect
    "i":     "#2ca02c",  # green       — item main effect
    "mj":    "#d62728",  # red         — model × judge (highlighted)
    "mi":    "#9467bd",  # purple      — model × item
    "ji":    "#17becf",  # teal        — judge × item
    "mji_e": "#7f7f7f",  # gray        — residual (noise convention)
}
# Which color backgrounds are dark enough that white in-bar text reads
# better than black? Determined by simple luminance on the palette above.
LIGHT_TEXT_ON = {"m", "j", "mj", "mi"}  # blue, orange (saturated), red, purple
DARK_TEXT_ON  = {"i", "ji", "mji_e"}     # green, teal, gray are lighter


def _fit_benchmark(parquet: Path, judges: tuple[str, ...]):
    df = pd.read_parquet(parquet)
    panel = balanced_panel(df, judges=judges)
    g = fit(panel)
    return g.share(), g


# Display labels for the three benchmark bars; the label includes the
# balanced (m, n_j, n_i) so the figure is self-documenting.
PRETTY = {
    "wildbench":    "WildBench",
    "arena_hard":   "Arena-Hard",
    "biggen_bench": "BiGGen-Bench",
}


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shares: dict[str, dict[str, float]] = {}
    for name, (parquet, judges) in BENCHMARKS.items():
        sh, g = _fit_benchmark(parquet, judges)
        label = (f"{PRETTY.get(name, name)}\n"
                 f"(m={g.n_m}, $n_j$={g.n_j}, $n_i$={g.n_i})")
        shares[label] = sh

    fig, ax = plt.subplots(figsize=(9, 0.9 + 1.2 * len(shares)))
    y_pos = list(range(len(shares)))
    left = [0.0] * len(shares)
    for c in COMPONENTS:
        vals = [shares[b][c] for b in shares]
        ax.barh(
            y_pos, vals, left=left, color=COLORS[c],
            edgecolor="white", linewidth=0.5, label=LABELS[c],
        )
        for i, v in enumerate(vals):
            if v >= 0.04:
                ax.text(left[i] + v / 2, i, f"{100*v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if c in LIGHT_TEXT_ON else "black")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(shares.keys()))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total observed-score variance")
    ax.set_title("Where does LLM-as-a-Judge score variance live?", loc="left")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"variance_decomposition.{ext}"
        plt.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"[viz] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    ARENA_HARD_JUDGES,
    COMPONENTS,
    WILDBENCH_JUDGES,
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
# Highlight σ²_m (signal) and σ²_mj (the project's headline term) with
# saturated colors; everything else is muted so the eye lands on those two.
COLORS = {
    "m":     "#1f77b4",
    "j":     "#d3d3d3",
    "i":     "#bdbdbd",
    "mj":    "#d62728",
    "mi":    "#a6a6a6",
    "ji":    "#909090",
    "mji_e": "#7a7a7a",
}


def _fit_benchmark(parquet: Path, judges: tuple[str, ...]) -> dict[str, float]:
    df = pd.read_parquet(parquet)
    panel = balanced_panel(df, judges=judges)
    g = fit(panel)
    return g.share()


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shares = {
        "WildBench\n(m=41, $n_j$=2, $n_i$=986)": _fit_benchmark(
            PROCESSED_DIR / "wildbench_long.parquet", WILDBENCH_JUDGES
        ),
        "Arena-Hard\n(m=8, $n_j$=5, $n_i$=462)": _fit_benchmark(
            PROCESSED_DIR / "arena_hard_long.parquet", ARENA_HARD_JUDGES
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 3.2))
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
                        color="white" if c in {"m", "mj"} else "black")
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

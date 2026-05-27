"""Rank-flip probability heatmap.

For each benchmark, plots a heatmap with adjacent-pair rank on the y-axis
(top = highest-ranked pair) and item count n_i on the x-axis, holding n_j
fixed at the modal-practice value of 1. Each cell shows the analytical RFP
for that adjacent pair at that design. Red = high flip rate, blue = stable.

This is the operationally meaningful answer to: "at the standard
single-judge n_j=1 setup, how many items do we need to make adjacent
leaderboard ranks reliable?"

Usage:
    python -m src.viz.rfp_heatmap
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"


def _short(name: str, n: int = 18) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _plot_one(ax, rfp_csv: Path, benchmark: str, fixed_n_j: int = 1) -> None:
    df = pd.read_csv(rfp_csv)
    df = df[df["n_j"] == fixed_n_j].copy()
    df["pair"] = df.apply(
        lambda r: f"{_short(r['model_high'])}  vs\n{_short(r['model_low'])}", axis=1
    )
    # Preserve adjacency order by row id of the first appearance.
    first_appearance = df.drop_duplicates("pair").reset_index(drop=True)
    pair_order = list(first_appearance["pair"])
    n_i_grid = sorted(df["n_i"].unique())

    mat = np.full((len(pair_order), len(n_i_grid)), np.nan)
    for r, pair in enumerate(pair_order):
        for c, n_i in enumerate(n_i_grid):
            sub = df[(df["pair"] == pair) & (df["n_i"] == n_i)]
            if not sub.empty:
                mat[r, c] = sub["rfp_analytical"].iloc[0]

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(n_i_grid)))
    ax.set_xticklabels([str(n) for n in n_i_grid])
    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels(pair_order, fontsize=6.5)
    ax.set_xlabel(f"$n_i$ (items per judge, $n_j={fixed_n_j}$)")
    ax.set_title(f"{benchmark}: adjacent-pair RFP", fontsize=11)

    # Cell annotations
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, f"{100*v:.0f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.25 else "black")
    return im


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # WildBench (40 pairs) dominates vertical space; allocate width
    # proportional to the number of adjacent pairs to keep cells readable.
    panels = [
        ("WildBench", TABLES_DIR / "rfp_wildbench.csv", 40),
        ("BiGGen-Bench", TABLES_DIR / "rfp_biggen_bench.csv", 3),  # m=4 → 3 pairs
        ("Arena-Hard", TABLES_DIR / "rfp_arena_hard.csv", 7),
    ]
    widths = [max(1, n // 5) for _, _, n in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(14, 9),
        gridspec_kw={"width_ratios": widths},
    )
    im = None
    for ax, (name, path, _) in zip(axes, panels):
        im = _plot_one(ax, path, name)

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Rank-flip probability (%)")
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.set_ticklabels(["0", "10", "20", "30", "40", "≥50"])

    fig.suptitle(
        "Adjacent-pair leaderboard ranks are unreliable at $n_j=1$ "
        "across achievable item counts",
        fontsize=12, y=0.995,
    )
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"rfp_heatmap.{ext}"
        plt.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"[viz] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

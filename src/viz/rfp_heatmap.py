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


def _short(name: str, n: int = 30) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _plot_one(ax, rfp_csv: Path, benchmark: str,
              fixed_n_j: int = 1, ytick_fontsize: float = 6.5) -> None:
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
    ax.set_yticklabels(pair_order, fontsize=ytick_fontsize)
    ax.set_xlabel(f"$n_i$ (items per judge, $n_j={fixed_n_j}$)")
    ax.set_title(f"{benchmark}  ($n_j={fixed_n_j}$)", fontsize=11)

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

    # Two-row layout:
    #   Row 1 (full width): WildBench — 40 adjacent pairs need both
    #     vertical room and horizontal room for two-line y-tick labels.
    #   Row 2: BiGGen-Bench (3 pairs) and Arena-Hard (7 pairs) side by
    #     side, with width roughly proportional to their pair counts so
    #     each panel has enough room for its title and x-ticks.
    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[40, 8],
        width_ratios=[3, 7],
        hspace=0.20,
        # Wide horizontal gap so Arena-Hard's long y-tick labels (e.g.
        # "claude-3-opus-20240229 vs claude-3-sonnet-20240229") don't
        # spill leftward into the BiGGen-Bench panel.
        wspace=0.65,
    )
    ax_wb = fig.add_subplot(gs[0, :])
    ax_bg = fig.add_subplot(gs[1, 0])
    ax_ah = fig.add_subplot(gs[1, 1])

    im_wb = _plot_one(
        ax_wb, TABLES_DIR / "rfp_wildbench.csv", "WildBench",
        ytick_fontsize=6.5,
    )
    _plot_one(
        ax_bg, TABLES_DIR / "rfp_biggen_bench.csv", "BiGGen-Bench",
        ytick_fontsize=8,
    )
    _plot_one(
        ax_ah, TABLES_DIR / "rfp_arena_hard.csv", "Arena-Hard",
        ytick_fontsize=8,
    )

    # Anchor the colorbar to the WildBench row so it has plenty of
    # vertical space without straddling both rows awkwardly.
    cbar = fig.colorbar(im_wb, ax=ax_wb, shrink=0.55, pad=0.02)
    cbar.set_label("Rank-flip probability (%)")
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.set_ticklabels(["0", "10", "20", "30", "40", "≥50"])

    fig.suptitle(
        "Adjacent-pair leaderboard ranks are unreliable at $n_j=1$ "
        "across achievable item counts",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"rfp_heatmap.{ext}"
        plt.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"[viz] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

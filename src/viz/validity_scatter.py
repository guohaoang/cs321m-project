"""Human-validity scatter.

Uses BiGGen-Bench's ``human_eval`` split, which gives 2{,}776 expert human
ratings on the same (model, item) cells that the five LLM judges score.
This is the project's check that σ²_m is recovering something
human-meaningful and not just inter-judge consensus on a bias.

Two complementary views:
  (a) Per-model: aggregate human and LLM-judge means by model, plot the
      4 points, fit a regression line, report Spearman ρ.
  (b) Per-judge: scatter human_score vs LLM-judge score at the cell
      level for each of the five judges, side-by-side.

Saves outputs/figures/validity_scatter.{png,pdf} and a per-judge
correlation table at outputs/tables/validity_correlations.csv.

Usage:
    python -m src.viz.validity_scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_parquet(PROCESSED_DIR / "biggen_bench_long.parquet")
    human_df = pd.read_parquet(PROCESSED_DIR / "biggen_human_scores.parquet")

    # Per-cell join: each (model, item) cell appears once per judge in
    # long_df; the human file has at most one rating per (model, item).
    merged = long_df.merge(
        human_df[["model", "item_id", "human_score"]],
        on=["model", "item_id"], how="inner",
    )

    judges = sorted(merged["judge"].unique())

    # ---------- per-judge correlations table ----------
    rows = []
    for j in judges:
        sub = merged[merged["judge"] == j]
        rho_s, p_s = spearmanr(sub["score"], sub["human_score"])
        rho_p, p_p = pearsonr(sub["score"], sub["human_score"])
        rows.append({
            "judge": j,
            "n": len(sub),
            "pearson_r": rho_p,
            "pearson_p": p_p,
            "spearman_rho": rho_s,
            "spearman_p": p_s,
        })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(TABLES_DIR / "validity_correlations.csv", index=False)
    print("[validity] per-judge cell-level correlations with human_score:")
    print(corr_df.round(3).to_string(index=False))

    # ---------- two-row layout: model summary on top, per-judge panel below ----
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(
        2, len(judges),
        height_ratios=[1.0, 1.2],
        hspace=0.55, wspace=0.35,
    )
    ax0 = fig.add_subplot(gs[0, :2])
    model_summary = (
        merged.groupby("model")
        .agg(human_mean=("human_score", "mean"),
             llm_mean=("score", "mean"))
        .reset_index()
    )
    ax0.scatter(model_summary["human_mean"], model_summary["llm_mean"],
                s=80, color="#1f77b4", zorder=3)
    for _, row in model_summary.iterrows():
        ax0.annotate(row["model"],
                     (row["human_mean"], row["llm_mean"]),
                     fontsize=7, xytext=(4, -3), textcoords="offset points")
    # Identity line
    lo = min(model_summary["human_mean"].min(), model_summary["llm_mean"].min()) - 0.3
    hi = max(model_summary["human_mean"].max(), model_summary["llm_mean"].max()) + 0.3
    ax0.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8)
    rho_m, p_m = spearmanr(model_summary["human_mean"], model_summary["llm_mean"])
    ax0.set_xlabel("Mean human score")
    ax0.set_ylabel("Mean LLM-judge score\n(averaged over 5 judges)")
    ax0.set_title(f"Per-model: ρ={rho_m:.2f}", fontsize=10)
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_aspect("equal")

    # ---------- per-judge scatter (bottom row) ----------
    rng = np.random.default_rng(42)
    for k, j in enumerate(judges):
        ax = fig.add_subplot(gs[1, k])
        sub = merged[merged["judge"] == j]
        # Tiny jitter so integer ratings don't pile up on a single pixel.
        jx = rng.uniform(-0.18, 0.18, len(sub))
        jy = rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(sub["human_score"] + jx, sub["score"] + jy,
                   s=3, alpha=0.18, color="#444444", edgecolors="none")
        # Per-judge regression line
        m, b = np.polyfit(sub["human_score"], sub["score"], 1)
        xs = np.linspace(1, 5, 50)
        ax.plot(xs, m * xs + b, color="#d62728", lw=1.5)
        ax.plot([1, 5], [1, 5], "--", color="gray", lw=0.6)
        r = corr_df[corr_df["judge"] == j].iloc[0]
        ax.set_title(f"{j}\n$r$={r['pearson_r']:.2f}, ρ={r['spearman_rho']:.2f}",
                     fontsize=9)
        ax.set_xlabel("human")
        if k == 0:
            ax.set_ylabel("judge")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])

    fig.suptitle(
        "Human-validity check (BiGGen-Bench): LLM-judge scores track human ratings",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"validity_scatter.{ext}"
        plt.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"[viz] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

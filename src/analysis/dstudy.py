"""Decision study (D-study) and pairwise rank-flip probability.

Given the seven variance components from ``gstudy.py``, computes:

* The standard G-coefficient ``E[ρ²] = σ²_m / (σ²_m + σ²_rel)`` and absolute
  dependability ``Φ`` at any operator's choice of (n_j, n_i).
* The analytical **pairwise rank-flip probability** under a Normal
  approximation. In our same-judges-same-items crossed design the judge
  main effect σ²_j and item main effect σ²_i cancel from the
  difference-score variance: every (m, j, i) cell of model A and model B
  shares the same j and i samples, so β_j and γ_i appear identically in
  both means and subtract out. The remaining difference-score variance is

      Var(X̄_A - X̄_B) = 2 · [ σ²_mj / n_j + σ²_mi / n_i + σ²_mji,e / (n_j n_i) ]

  and the rank-flip probability is

      RFP(A, B; n_j, n_i) = Pr[X̄_A < X̄_B | μ_A > μ_B]
                         = Φ_N(-(μ_A - μ_B) / SEM_diff)

* A parametric-bootstrap RFP that re-simulates judge/item/cell effects
  and serves as a validity check on the Normal approximation.
* (n_j, n_i) sweeps over the adjacent-pair leaderboard for the
  pre-registered H2 test.

Usage:
    python -m src.analysis.dstudy
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analysis.gstudy import (
    ARENA_HARD_JUDGES,
    BENCHMARKS,
    BIGGEN_BENCH_JUDGES,
    WILDBENCH_JUDGES,
    balanced_panel,
    fit,
)
from src.utils.seeds import DSTUDY_SEED

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"

# Sweep grids used by main() for the (n_j, n_i) tables. Includes the
# "modal practice" point (n_j=1, n_i=80) named in pre-registered H2 along
# with each benchmark's published item count and several intermediate
# values useful for the heatmap.
N_J_GRID = (1, 2, 3, 5, 10)
N_I_GRID = (40, 80, 160, 320, 500, 1000)


# -----------------------------------------------------------------------------
# Scalar reliability summaries
# -----------------------------------------------------------------------------

def relative_error_var(vc: dict[str, float], n_j: int, n_i: int) -> float:
    return (
        vc["mj"] / n_j
        + vc["mi"] / n_i
        + vc["mji_e"] / (n_j * n_i)
    )


def absolute_error_var(vc: dict[str, float], n_j: int, n_i: int) -> float:
    return relative_error_var(vc, n_j, n_i) + (
        vc["j"] / n_j
        + vc["i"] / n_i
        + vc["ji"] / (n_j * n_i)
    )


def g_coefficient(vc: dict[str, float], n_j: int, n_i: int) -> float:
    """Relative G-coefficient E[ρ²]."""
    rel = relative_error_var(vc, n_j, n_i)
    return vc["m"] / (vc["m"] + rel) if (vc["m"] + rel) > 0 else float("nan")


def phi_coefficient(vc: dict[str, float], n_j: int, n_i: int) -> float:
    """Absolute-dependability Φ."""
    abs_err = absolute_error_var(vc, n_j, n_i)
    return vc["m"] / (vc["m"] + abs_err) if (vc["m"] + abs_err) > 0 else float("nan")


# -----------------------------------------------------------------------------
# Pairwise rank-flip probability — analytical
# -----------------------------------------------------------------------------

def diff_score_sem(vc: dict[str, float], n_j: int, n_i: int) -> float:
    """Standard error of the difference X̄_A − X̄_B under the same-judges-
    same-items design (σ²_j and σ²_i cancel)."""
    var_diff = 2.0 * relative_error_var(vc, n_j, n_i)
    return float(np.sqrt(var_diff))


def rfp_analytical(
    mu_a: float,
    mu_b: float,
    vc: dict[str, float],
    n_j: int,
    n_i: int,
) -> float:
    """Pairwise rank-flip probability. Conditions on μ_A > μ_B (the higher-
    universe-score model 'should' beat the other)."""
    if mu_a <= mu_b:
        mu_a, mu_b = mu_b, mu_a
    sem = diff_score_sem(vc, n_j, n_i)
    if sem == 0:
        return 0.0
    z = -(mu_a - mu_b) / sem
    return float(norm.cdf(z))


# -----------------------------------------------------------------------------
# Pairwise RFP — parametric bootstrap (validity check on the Normal approx)
# -----------------------------------------------------------------------------

def rfp_bootstrap(
    mu_a: float,
    mu_b: float,
    vc: dict[str, float],
    n_j: int,
    n_i: int,
    n_sim: int = 2_000,
    seed: int = DSTUDY_SEED,
) -> float:
    """Simulate the same-judges-same-items design ``n_sim`` times and count
    the fraction of replicates where the wrong model wins."""
    if mu_a <= mu_b:
        mu_a, mu_b = mu_b, mu_a
    rng = np.random.default_rng(seed)

    sd_mj = np.sqrt(max(vc["mj"], 0.0))
    sd_mi = np.sqrt(max(vc["mi"], 0.0))
    sd_e = np.sqrt(max(vc["mji_e"], 0.0))

    # All judge/item-side terms cancel between A and B, so we only need to
    # simulate the parts that do not.
    flips = 0
    for _ in range(n_sim):
        # σ²_mj is a per-(model, judge) effect; A and B each draw their own
        # n_j-length vector. The arithmetic mean over n_j judges has variance
        # σ²_mj / n_j.
        e_mj_a = rng.normal(0.0, sd_mj, n_j).mean()
        e_mj_b = rng.normal(0.0, sd_mj, n_j).mean()

        # σ²_mi is per-(model, item). The mean over n_i items has variance
        # σ²_mi / n_i.
        e_mi_a = rng.normal(0.0, sd_mi, n_i).mean()
        e_mi_b = rng.normal(0.0, sd_mi, n_i).mean()

        # σ²_mji,e is per-cell, draw n_j × n_i and average.
        e_e_a = rng.normal(0.0, sd_e, (n_j, n_i)).mean()
        e_e_b = rng.normal(0.0, sd_e, (n_j, n_i)).mean()

        x_a = mu_a + e_mj_a + e_mi_a + e_e_a
        x_b = mu_b + e_mj_b + e_mi_b + e_e_b
        if x_a < x_b:
            flips += 1
    return flips / n_sim


# -----------------------------------------------------------------------------
# Leaderboard helpers
# -----------------------------------------------------------------------------

@dataclass
class Leaderboard:
    means: pd.Series   # index = model, value = empirical μ_m
    sorted_models: list[str]  # high → low by mean


def leaderboard_from_panel(panel: pd.DataFrame) -> Leaderboard:
    means = panel.groupby("model")["score"].mean().sort_values(ascending=False)
    return Leaderboard(means=means, sorted_models=list(means.index))


def adjacent_pairs(lb: Leaderboard) -> list[tuple[str, str]]:
    return list(zip(lb.sorted_models[:-1], lb.sorted_models[1:]))


# -----------------------------------------------------------------------------
# Aggregate the sweep + tables
# -----------------------------------------------------------------------------

def dstudy_grid(vc: dict[str, float], n_j_grid=N_J_GRID, n_i_grid=N_I_GRID) -> pd.DataFrame:
    rows = []
    for n_j in n_j_grid:
        for n_i in n_i_grid:
            rows.append({
                "n_j": n_j,
                "n_i": n_i,
                "g_coef": g_coefficient(vc, n_j, n_i),
                "phi": phi_coefficient(vc, n_j, n_i),
                "diff_sem": diff_score_sem(vc, n_j, n_i),
            })
    return pd.DataFrame(rows)


def rfp_table(
    lb: Leaderboard,
    vc: dict[str, float],
    n_j_grid=N_J_GRID,
    n_i_grid=N_I_GRID,
    bootstrap_check_at: tuple[int, int] | None = (1, 80),
    bootstrap_sims: int = 2_000,
) -> pd.DataFrame:
    """Full adjacent-pair × (n_j, n_i) RFP table.

    For one nominated grid point, also runs the parametric-bootstrap RFP as
    a sanity check against the Normal approximation.
    """
    rows = []
    pairs = adjacent_pairs(lb)
    for (a, b) in pairs:
        mu_a = lb.means[a]
        mu_b = lb.means[b]
        for n_j in n_j_grid:
            for n_i in n_i_grid:
                row = {
                    "model_high": a,
                    "model_low": b,
                    "mu_high": mu_a,
                    "mu_low": mu_b,
                    "delta": mu_a - mu_b,
                    "n_j": n_j,
                    "n_i": n_i,
                    "rfp_analytical": rfp_analytical(mu_a, mu_b, vc, n_j, n_i),
                }
                if bootstrap_check_at == (n_j, n_i):
                    row["rfp_bootstrap"] = rfp_bootstrap(
                        mu_a, mu_b, vc, n_j, n_i, n_sim=bootstrap_sims
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    name: str
    parquet_path: Path
    judges: tuple[str, ...]


def _run(b: BenchmarkRun) -> None:
    df = pd.read_parquet(b.parquet_path)
    panel = balanced_panel(df, judges=b.judges)
    g = fit(panel)
    vc = g.components_clipped
    lb = leaderboard_from_panel(panel)
    print(f"[dstudy] {b.name}: m={g.n_m}, j={g.n_j}, i={g.n_i}")

    # 1. D-study grid (Eρ², Φ vs (n_j, n_i))
    grid = dstudy_grid(vc)
    grid.insert(0, "benchmark", b.name)
    grid.to_csv(TABLES_DIR / f"dstudy_{b.name}_grid.csv", index=False)

    # 2. RFP across adjacent pairs and (n_j, n_i)
    rfp = rfp_table(lb, vc)
    rfp.insert(0, "benchmark", b.name)
    rfp.to_csv(TABLES_DIR / f"rfp_{b.name}.csv", index=False)

    # 3. Leaderboard snapshot
    lb_df = lb.means.reset_index().rename(columns={"score": "mu_m"})
    lb_df.insert(0, "benchmark", b.name)
    lb_df.to_csv(TABLES_DIR / f"leaderboard_{b.name}.csv", index=False)

    # 4. Pre-registered H2 snapshot at n_j=1, n_i=80
    snap = rfp[(rfp["n_j"] == 1) & (rfp["n_i"] == 80)].copy()
    print(f"[dstudy] {b.name}: H2 snapshot at n_j=1, n_i=80")
    print(f"    max adjacent-pair RFP = {snap['rfp_analytical'].max():.4f}")
    print(f"    median adjacent-pair RFP = {snap['rfp_analytical'].median():.4f}")
    if "rfp_bootstrap" in snap.columns:
        print(f"    bootstrap check (one pair): "
              f"analytical={snap['rfp_analytical'].iloc[0]:.4f}, "
              f"bootstrap={snap['rfp_bootstrap'].iloc[0]:.4f}")

    # 5. Smallest (n_j, n_i) holding RFP ≤ 5% for every adjacent pair
    per_grid = rfp.groupby(["n_j", "n_i"])["rfp_analytical"].max().reset_index()
    safe = per_grid[per_grid["rfp_analytical"] <= 0.05]
    if not safe.empty:
        # Pick min by n_j × n_i (judge calls), then by n_j (judges are pricier)
        safe = safe.assign(cost=safe["n_j"] * safe["n_i"]).sort_values(
            ["cost", "n_j", "n_i"]
        )
        winner = safe.iloc[0]
        print(f"[dstudy] {b.name}: cheapest design with max adjacent RFP ≤ 5%: "
              f"n_j={int(winner['n_j'])}, n_i={int(winner['n_i'])} "
              f"(judge-calls={int(winner['cost'])})")
    else:
        print(f"[dstudy] {b.name}: NO grid point holds max adjacent RFP ≤ 5%")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs = [BenchmarkRun(name, p, j) for name, (p, j) in BENCHMARKS.items()]
    for b in runs:
        _run(b)
    print(f"[dstudy] wrote tables to {TABLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

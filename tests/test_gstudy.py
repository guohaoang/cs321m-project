"""Smoke tests for the G-study variance-component estimator.

The recovery test generates synthetic data with known variance components,
fits, and checks that the estimator returns values close to the true ones.
The applied tests confirm the public surface on real ingest output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.gstudy import (
    COMPONENTS,
    balanced_panel,
    bootstrap,
    cross_validate_items,
    cross_validate_items_rasch_baseline,
    fit,
    summarize_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _simulate(n_m: int, n_j: int, n_i: int, sigmas: dict[str, float], seed: int) -> pd.DataFrame:
    """Generate a fully crossed three-way single-rating dataset with the given
    variance components."""
    rng = np.random.default_rng(seed)
    eff_m = rng.normal(0, np.sqrt(sigmas["m"]), n_m)
    eff_j = rng.normal(0, np.sqrt(sigmas["j"]), n_j)
    eff_i = rng.normal(0, np.sqrt(sigmas["i"]), n_i)
    eff_mj = rng.normal(0, np.sqrt(sigmas["mj"]), (n_m, n_j))
    eff_mi = rng.normal(0, np.sqrt(sigmas["mi"]), (n_m, n_i))
    eff_ji = rng.normal(0, np.sqrt(sigmas["ji"]), (n_j, n_i))
    eff_res = rng.normal(0, np.sqrt(sigmas["mji_e"]), (n_m, n_j, n_i))

    grand = 5.0
    tensor = (
        grand
        + eff_m[:, None, None]
        + eff_j[None, :, None]
        + eff_i[None, None, :]
        + eff_mj[:, :, None]
        + eff_mi[:, None, :]
        + eff_ji[None, :, :]
        + eff_res
    )
    rows = []
    for ix_m in range(n_m):
        for ix_j in range(n_j):
            for ix_i in range(n_i):
                rows.append({
                    "benchmark": "synth",
                    "model": f"m{ix_m}",
                    "judge": f"j{ix_j}",
                    "item_id": f"i{ix_i}",
                    "category": "synth",
                    "score": tensor[ix_m, ix_j, ix_i],
                })
    return pd.DataFrame(rows)


def test_estimator_is_unbiased_in_expectation() -> None:
    """Average across many simulations: each component's mean estimate should
    sit within ±20% of the true value. Single-shot precision is poor when
    a facet has few levels (especially σ²_j with n_j=4), but unbiasedness
    holds for the ANOVA MoM estimator under the random-effects model."""
    true = {
        "m": 1.0,
        "j": 0.6,
        "i": 0.8,
        "mj": 0.3,
        "mi": 0.4,
        "ji": 0.2,
        "mji_e": 0.5,
    }
    n_sims = 60
    estimates = {k: [] for k in COMPONENTS}
    for s in range(n_sims):
        df = _simulate(n_m=10, n_j=4, n_i=100, sigmas=true, seed=2026 + s)
        g = fit(df)
        for k in COMPONENTS:
            # Use unclipped estimates so symmetric noise around zero averages
            # to the true value rather than upwards-biased after clipping.
            estimates[k].append(g.components[k])
    for k in COMPONENTS:
        mean_est = float(np.mean(estimates[k]))
        tol = 0.2 * true[k] + 0.05
        assert abs(mean_est - true[k]) < tol, (
            f"mean σ²_{k}: {mean_est:.3f} vs true {true[k]:.3f} (tol={tol:.3f}) "
            f"over {n_sims} sims"
        )


def test_single_shot_main_components_within_tolerance() -> None:
    """With a big design (n_i=500), each component should be recoverable in
    a single simulation within wide tolerance. Cheap sanity check on the
    estimator wiring."""
    true = {"m": 1.0, "j": 0.6, "i": 0.8, "mj": 0.3, "mi": 0.4, "ji": 0.2, "mji_e": 0.5}
    df = _simulate(n_m=10, n_j=4, n_i=500, sigmas=true, seed=2026)
    g = fit(df)
    # σ²_j and σ²_ji still have only 3 df from n_j-1; loosen for them.
    looser = {"j", "ji"}
    for k in COMPONENTS:
        tol = 0.8 * true[k] + 0.1 if k in looser else 0.3 * true[k] + 0.05
        assert abs(g.components[k] - true[k]) < tol, (
            f"σ²_{k}: estimate={g.components[k]:.3f} true={true[k]:.3f} "
            f"(tol={tol:.3f})"
        )


def test_components_shape() -> None:
    df = _simulate(n_m=4, n_j=3, n_i=30,
                   sigmas={k: 0.5 for k in COMPONENTS}, seed=1)
    g = fit(df)
    assert set(g.components.keys()) == set(COMPONENTS)
    assert set(g.components_clipped.keys()) == set(COMPONENTS)
    assert g.n_m == 4
    assert g.n_j == 3
    assert g.n_i == 30


def test_bootstrap_returns_correct_shape() -> None:
    df = _simulate(n_m=3, n_j=2, n_i=20,
                   sigmas={k: 1.0 for k in COMPONENTS}, seed=7)
    boot = bootstrap(df, n_reps=50, seed=11)
    assert boot.shape == (50, len(COMPONENTS))
    summary = summarize_bootstrap(boot)
    assert set(summary["component"]) == set(COMPONENTS)
    assert (summary["ci_low"] <= summary["median"]).all()
    assert (summary["median"] <= summary["ci_high"]).all()


def test_balanced_panel_filters_to_intersection() -> None:
    # Build a synthetic panel where one item is missing from one cell.
    df = _simulate(n_m=3, n_j=2, n_i=10,
                   sigmas={k: 0.1 for k in COMPONENTS}, seed=2)
    # Drop one observation
    df = df.drop(df.index[5]).reset_index(drop=True)
    balanced = balanced_panel(df, drop_nan_scores=False)
    n_items = balanced["item_id"].nunique()
    assert n_items == 9, f"expected 9 items after dropping one cell, got {n_items}"
    # And all cells fully crossed
    counts = balanced.groupby(["judge", "model"]).size()
    assert (counts == 9).all()


def test_cross_validate_items_schema_and_bounds() -> None:
    """5-fold CV returns the expected schema and bounded metrics."""
    df = _simulate(n_m=4, n_j=3, n_i=40,
                   sigmas={k: 0.5 for k in COMPONENTS}, seed=3)
    cv = cross_validate_items(df, k=5, seed=42)
    # Schema: 5 fold rows + 1 ALL row.
    assert set(cv.columns) == {"fold", "n_test_cells", "r2", "rmse"}
    assert len(cv) == 6
    # The fold labels should be 0..4 then "ALL".
    assert list(cv["fold"])[:5] == [0, 1, 2, 3, 4]
    assert cv.iloc[5]["fold"] == "ALL"
    # Every R² should be ≤ 1 (can be negative if the predictor is worse than
    # the fold's own mean; should not happen on this synthetic).
    assert (cv["r2"] <= 1.0).all()
    assert (cv["rmse"] >= 0).all()
    # Aggregate test-cell count equals the panel size, since every cell is
    # held out exactly once.
    assert int(cv.iloc[5]["n_test_cells"]) == len(df)


def test_rasch_baseline_schema_and_bounds() -> None:
    """The Rasch baseline CV returns the same schema as the full predictor."""
    df = _simulate(n_m=4, n_j=3, n_i=40,
                   sigmas={k: 0.5 for k in COMPONENTS}, seed=5)
    cv_rasch = cross_validate_items_rasch_baseline(df, k=5, seed=42)
    assert set(cv_rasch.columns) == {"fold", "n_test_cells", "r2", "rmse"}
    assert len(cv_rasch) == 6
    assert list(cv_rasch["fold"])[:5] == [0, 1, 2, 3, 4]
    assert cv_rasch.iloc[5]["fold"] == "ALL"
    assert (cv_rasch["r2"] <= 1.0).all()
    assert (cv_rasch["rmse"] >= 0).all()
    assert int(cv_rasch.iloc[5]["n_test_cells"]) == len(df)


def test_rasch_baseline_never_beats_full_predictor() -> None:
    """The full predictor uses strictly more information than the Rasch
    baseline (adds judge main + (m, j) interaction), so its aggregate
    R² should not be lower. Allow a tiny tolerance for finite-sample
    noise."""
    sigmas = {"m": 1.0, "j": 0.6, "i": 0.5, "mj": 0.3,
              "mi": 0.4, "ji": 0.2, "mji_e": 0.5}
    df = _simulate(n_m=6, n_j=4, n_i=60, sigmas=sigmas, seed=6)
    cv_full = cross_validate_items(df, k=5, seed=42)
    cv_rasch = cross_validate_items_rasch_baseline(df, k=5, seed=42)
    full_r2 = cv_full[cv_full["fold"] == "ALL"].iloc[0]["r2"]
    rasch_r2 = cv_rasch[cv_rasch["fold"] == "ALL"].iloc[0]["r2"]
    assert full_r2 + 1e-6 >= rasch_r2, (
        f"full predictor R²={full_r2:.4f} < Rasch baseline R²={rasch_r2:.4f}"
    )


def test_cross_validate_items_no_item_effect_high_r2() -> None:
    """If σ²_i = σ²_mi = σ²_ji = 0 and within-cell noise is small, the
    (m, j) cell-mean predictor should achieve high CV R²."""
    sigmas = {"m": 1.0, "j": 0.5, "i": 0.0, "mj": 0.2,
              "mi": 0.0, "ji": 0.0, "mji_e": 0.01}
    df = _simulate(n_m=4, n_j=3, n_i=80, sigmas=sigmas, seed=4)
    cv = cross_validate_items(df, k=5, seed=42)
    agg_r2 = cv[cv["fold"] == "ALL"].iloc[0]["r2"]
    assert agg_r2 > 0.95, f"expected high R² with no item effect, got {agg_r2:.3f}"


@pytest.mark.parametrize("benchmark,judges,parquet_name", [
    ("wildbench",
     ("gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"),
     "wildbench_long.parquet"),
    ("arena_hard",
     ("claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
      "gemini-1.5-pro-api-0514", "gpt-4-1106-preview", "llama-3-70b-instruct"),
     "arena_hard_long.parquet"),
])
def test_fit_on_real_ingest(benchmark: str, judges: tuple[str, ...], parquet_name: str) -> None:
    """The headline G-study should run on real data without exceptions and
    produce finite variance components."""
    p = REPO_ROOT / "data" / "processed" / parquet_name
    if not p.exists():
        pytest.skip(f"{parquet_name} missing; run the ingest first.")
    df = pd.read_parquet(p)
    balanced = balanced_panel(df, judges=judges)
    if balanced.empty:
        pytest.skip(f"no balanced panel for {benchmark}")
    g = fit(balanced)
    for k in COMPONENTS:
        assert np.isfinite(g.components[k]), f"{benchmark} σ²_{k} is non-finite"
        assert g.components_clipped[k] >= 0

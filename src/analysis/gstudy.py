"""Generalizability Study (G-study).

Estimates the 7-component variance decomposition

    σ²(X_{mji}) = σ²_m + σ²_j + σ²_i + σ²_mj + σ²_mi + σ²_ji + σ²_{mji,e}

for a fully crossed model × judge × item single-rating design.

Method: Brennan-style ANOVA method-of-moments. Each of the seven expected
mean squares is a linear combination of the variance components, so the
seven sample MS values give a 7×7 triangular linear system that we solve
directly. With one rating per cell the three-way interaction and the
within-cell residual are confounded, so σ²_{mji,e} is the joint estimate
of both — the pre-analysis plan notes this and treats the residual as an
upper bound.

Negative point estimates can arise from sampling variability (especially
when a true component is near zero); we report them as zero per the
Brennan convention and surface a flag, rather than silently swallowing.

Bootstrap CIs resample items with replacement (the random facet over
which the operator wants to generalize). Per HANDOFF §5 step 7, default
is 1,000 reps.

Usage:
    python -m src.analysis.gstudy
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.seeds import BOOTSTRAP_SEED, CV_SEED

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"

# Variance-component keys, in canonical reporting order.
COMPONENTS = ("m", "j", "i", "mj", "mi", "ji", "mji_e")


@dataclass
class GStudyFit:
    components: dict[str, float]  # raw point estimates (can be slightly < 0)
    components_clipped: dict[str, float]  # max(0, .) per Brennan convention
    n_m: int
    n_j: int
    n_i: int
    negative_flags: tuple[str, ...]  # which components were < 0 before clipping

    def total(self) -> float:
        return sum(self.components_clipped.values())

    def share(self) -> dict[str, float]:
        tot = self.total()
        return {k: (v / tot if tot > 0 else float("nan"))
                for k, v in self.components_clipped.items()}

    def as_row(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in COMPONENTS:
            out[f"var_{k}"] = self.components_clipped[k]
        for k in COMPONENTS:
            out[f"share_{k}"] = self.share()[k]
        out["n_m"] = self.n_m
        out["n_j"] = self.n_j
        out["n_i"] = self.n_i
        out["negative_flags"] = ",".join(self.negative_flags)
        return out


# -----------------------------------------------------------------------------
# Balancing helper
# -----------------------------------------------------------------------------

def balanced_panel(
    df: pd.DataFrame,
    judges: tuple[str, ...] | None = None,
    models: tuple[str, ...] | None = None,
    drop_nan_scores: bool = True,
) -> pd.DataFrame:
    """Filter ``df`` to a fully crossed m × j × i panel.

    Steps:
      1. Optionally restrict to a chosen judge / model subset.
      2. Drop rows with NaN scores.
      3. Keep only items that appear in *every* (judge, model) cell of the
         remaining data. This produces a balanced panel where each item
         contributes exactly one observation per (judge, model) cell, which
         is what the ANOVA MoM derivation assumes.
    """
    out = df
    if judges is not None:
        out = out[out["judge"].isin(judges)]
    if models is not None:
        out = out[out["model"].isin(models)]
    if drop_nan_scores:
        out = out[out["score"].notna()]

    # Auto-restrict to the model intersection across the selected judges.
    # Without this, a model judged by only some of the requested judges
    # would inflate the expected-cell count and cause every item to drop.
    if models is None:
        models_per_judge = out.groupby("judge")["model"].unique()
        if len(models_per_judge) == 0:
            return out.head(0)
        common_models = set.intersection(*(set(ms) for ms in models_per_judge))
        out = out[out["model"].isin(common_models)]

    expected_cells = out["judge"].nunique() * out["model"].nunique()
    cell_count_per_item = out.groupby("item_id").size()
    keep_items = cell_count_per_item[cell_count_per_item == expected_cells].index
    out = out[out["item_id"].isin(keep_items)]

    # Final sanity: the balanced panel must have exactly one row per cell.
    cell_sizes = out.groupby(["judge", "model", "item_id"]).size()
    if not (cell_sizes == 1).all():
        bad = cell_sizes[cell_sizes != 1]
        raise ValueError(
            f"balanced_panel produced duplicate (judge, model, item) cells: "
            f"{bad.head(3).to_dict()}"
        )
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# ANOVA MoM core
# -----------------------------------------------------------------------------

def _build_score_tensor(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Pivot the long-form panel into a dense (n_m, n_j, n_i) tensor."""
    models = sorted(df["model"].unique())
    judges = sorted(df["judge"].unique())
    items = sorted(df["item_id"].unique())
    m_ix = {m: k for k, m in enumerate(models)}
    j_ix = {j: k for k, j in enumerate(judges)}
    i_ix = {i: k for k, i in enumerate(items)}

    tensor = np.full((len(models), len(judges), len(items)), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        tensor[m_ix[row.model], j_ix[row.judge], i_ix[row.item_id]] = row.score

    if np.isnan(tensor).any():
        # balanced_panel should have eliminated this case.
        missing = int(np.isnan(tensor).sum())
        raise ValueError(f"score tensor still has {missing} NaN cells")
    return tensor, models, judges, items


def _mean_squares(tensor: np.ndarray) -> dict[str, float]:
    """Classical balanced three-way ANOVA mean squares.

    With one observation per (m, j, i) cell:
      df_m   = n_m - 1
      df_j   = n_j - 1
      df_i   = n_i - 1
      df_mj  = (n_m-1)(n_j-1)
      df_mi  = (n_m-1)(n_i-1)
      df_ji  = (n_j-1)(n_i-1)
      df_res = (n_m-1)(n_j-1)(n_i-1)    (three-way interaction = residual)
    """
    n_m, n_j, n_i = tensor.shape
    grand = tensor.mean()
    m_mean = tensor.mean(axis=(1, 2))            # (n_m,)
    j_mean = tensor.mean(axis=(0, 2))            # (n_j,)
    i_mean = tensor.mean(axis=(0, 1))            # (n_i,)
    mj_mean = tensor.mean(axis=2)                # (n_m, n_j)
    mi_mean = tensor.mean(axis=1)                # (n_m, n_i)
    ji_mean = tensor.mean(axis=0)                # (n_j, n_i)

    ss_m = n_j * n_i * np.sum((m_mean - grand) ** 2)
    ss_j = n_m * n_i * np.sum((j_mean - grand) ** 2)
    ss_i = n_m * n_j * np.sum((i_mean - grand) ** 2)
    ss_mj = n_i * np.sum(
        (mj_mean - m_mean[:, None] - j_mean[None, :] + grand) ** 2
    )
    ss_mi = n_j * np.sum(
        (mi_mean - m_mean[:, None] - i_mean[None, :] + grand) ** 2
    )
    ss_ji = n_m * np.sum(
        (ji_mean - j_mean[:, None] - i_mean[None, :] + grand) ** 2
    )
    ss_total = np.sum((tensor - grand) ** 2)
    ss_res = ss_total - ss_m - ss_j - ss_i - ss_mj - ss_mi - ss_ji

    return {
        "m": ss_m / (n_m - 1),
        "j": ss_j / (n_j - 1),
        "i": ss_i / (n_i - 1),
        "mj": ss_mj / ((n_m - 1) * (n_j - 1)),
        "mi": ss_mi / ((n_m - 1) * (n_i - 1)),
        "ji": ss_ji / ((n_j - 1) * (n_i - 1)),
        "mji_e": ss_res / ((n_m - 1) * (n_j - 1) * (n_i - 1)),
    }


def _components_from_ms(ms: dict[str, float], n_m: int, n_j: int, n_i: int) -> dict[str, float]:
    """Solve the Brennan EMS triangular system for the seven components.

    Expected mean squares (random model, single rating per cell):
        E[MS_res] = σ²_{mji,e}
        E[MS_mj]  = σ²_{mji,e} + n_i σ²_mj
        E[MS_mi]  = σ²_{mji,e} + n_j σ²_mi
        E[MS_ji]  = σ²_{mji,e} + n_m σ²_ji
        E[MS_m]   = σ²_{mji,e} + n_i σ²_mj + n_j σ²_mi + n_j n_i σ²_m
        E[MS_j]   = σ²_{mji,e} + n_i σ²_mj + n_m σ²_ji + n_m n_i σ²_j
        E[MS_i]   = σ²_{mji,e} + n_j σ²_mi + n_m σ²_ji + n_m n_j σ²_i
    """
    res = ms["mji_e"]
    var_mj = (ms["mj"] - res) / n_i
    var_mi = (ms["mi"] - res) / n_j
    var_ji = (ms["ji"] - res) / n_m
    var_m = (ms["m"] - ms["mj"] - ms["mi"] + res) / (n_j * n_i)
    var_j = (ms["j"] - ms["mj"] - ms["ji"] + res) / (n_m * n_i)
    var_i = (ms["i"] - ms["mi"] - ms["ji"] + res) / (n_m * n_j)
    return {
        "m": var_m,
        "j": var_j,
        "i": var_i,
        "mj": var_mj,
        "mi": var_mi,
        "ji": var_ji,
        "mji_e": res,
    }


def fit(df: pd.DataFrame) -> GStudyFit:
    """Run the G-study on an already-balanced long-form panel."""
    tensor, _, _, _ = _build_score_tensor(df)
    n_m, n_j, n_i = tensor.shape
    ms = _mean_squares(tensor)
    components = _components_from_ms(ms, n_m, n_j, n_i)
    flags = tuple(k for k, v in components.items() if v < 0)
    clipped = {k: max(0.0, v) for k, v in components.items()}
    return GStudyFit(
        components=components,
        components_clipped=clipped,
        n_m=n_m,
        n_j=n_j,
        n_i=n_i,
        negative_flags=flags,
    )


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def bootstrap(
    df: pd.DataFrame,
    n_reps: int = 1_000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Item-bootstrap CIs.

    Resamples item_ids with replacement, then rebuilds the balanced panel
    from the resampled item set and refits. Returns a (n_reps, 7) DataFrame
    of variance-component estimates, indexed by replicate.
    """
    items = df["item_id"].unique()
    rng = np.random.default_rng(seed)
    # Pre-pivot the balanced panel to a tensor for fast resample-and-recompute.
    tensor, models, judges, ref_items = _build_score_tensor(df)
    item_idx = {it: k for k, it in enumerate(ref_items)}
    item_array = np.array([item_idx[it] for it in items])

    rows: list[dict[str, float]] = []
    n_m, n_j, _ = tensor.shape
    for _ in range(n_reps):
        sample_ix = rng.choice(item_array, size=len(item_array), replace=True)
        resampled = tensor[:, :, sample_ix]
        ms = _mean_squares(resampled)
        comp = _components_from_ms(ms, n_m, n_j, resampled.shape[2])
        rows.append({k: max(0.0, v) for k, v in comp.items()})
    return pd.DataFrame.from_records(rows)


def summarize_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    """Per-component CI summary: median + 2.5/97.5 quantiles."""
    rows = []
    for c in COMPONENTS:
        rows.append({
            "component": c,
            "median": boot[c].median(),
            "ci_low": boot[c].quantile(0.025),
            "ci_high": boot[c].quantile(0.975),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Goodness-of-fit: leave-items-out k-fold cross-validation
# -----------------------------------------------------------------------------

def cross_validate_items(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = CV_SEED,
) -> pd.DataFrame:
    """k-fold leave-items-out CV with the (m, j)-cell-mean predictor.

    For each held-out cell (m, j, i*), the prediction is the train-set
    (m, j) cell mean: \\hat y = (1/|I_train|) Σ X_{m,j,i} for i in I_train.

    This is the optimal item-agnostic predictor under the random-effects
    model — it's the BLUP for a fresh item when item-specific terms are
    integrated out. It matches the D-study's generalization target
    (fresh item draws), so leave-items-out is the appropriate CV setup
    for a G-Theory analysis.

    The closed-form expected aggregate R² is
        (σ²_m + σ²_j + σ²_mj) / σ²_total,
    i.e. the share of variance that survives item-averaging. The
    measured aggregate R² should land within a few percentage points
    of this ceiling; a large gap indicates either a model
    mis-specification or finite-sample noise in the cell-mean
    estimator.

    Args:
        df: balanced long-form panel (columns: model, judge, item_id,
            score). Use balanced_panel() to produce.
        k: number of folds (default 5).
        seed: RNG seed for item shuffling (default CV_SEED).

    Returns:
        DataFrame with k + 1 rows. Per-fold rows have
        ``fold=0..k-1``; the aggregate row has ``fold='ALL'`` and
        reports R²/RMSE pooled across every test cell. Columns:
        ``fold, n_test_cells, r2, rmse``.
    """
    items = sorted(df["item_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(items)
    folds = np.array_split(shuffled, k)

    rows: list[dict] = []
    all_y: list[np.ndarray] = []
    all_yhat: list[np.ndarray] = []

    for fold_idx, test_items in enumerate(folds):
        test_set = set(test_items.tolist())
        train_mask = ~df["item_id"].isin(test_set)
        train_df = df[train_mask]
        test_df = df[~train_mask]

        # Train-set (m, j) cell mean — the BLUP for a fresh item.
        train_mj = train_df.groupby(["model", "judge"])["score"].mean()
        # Predict: per-row lookup into the train-set (m, j) table.
        test_keys = list(zip(test_df["model"], test_df["judge"]))
        train_global_mean = train_df["score"].mean()
        yhat = np.array([
            train_mj.get(key, train_global_mean) for key in test_keys
        ])
        y = test_df["score"].values

        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        rmse = float(np.sqrt(sse / len(y)))

        rows.append({
            "fold": fold_idx,
            "n_test_cells": int(len(y)),
            "r2": r2,
            "rmse": rmse,
        })
        all_y.append(y)
        all_yhat.append(yhat)

    pooled_y = np.concatenate(all_y)
    pooled_yhat = np.concatenate(all_yhat)
    agg_sse = float(np.sum((pooled_y - pooled_yhat) ** 2))
    agg_sst = float(np.sum((pooled_y - pooled_y.mean()) ** 2))
    rows.append({
        "fold": "ALL",
        "n_test_cells": int(len(pooled_y)),
        "r2": 1.0 - agg_sse / agg_sst if agg_sst > 0 else float("nan"),
        "rmse": float(np.sqrt(agg_sse / len(pooled_y))),
    })
    return pd.DataFrame(rows)


def cross_validate_items_rasch_baseline(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = CV_SEED,
) -> pd.DataFrame:
    """k-fold leave-items-out CV with a textbook 2-facet Rasch baseline.

    Predicts each held-out cell (m, j, i*) using only the train-set
    model main effect:

        \\hat y = mu_train + alpha_m_train = mean over train of X_{m, ., .}

    This is the textbook Rasch model (persons × items, no judge facet)
    evaluated under held-out items, where the unknown item difficulty
    delta_{i*} goes to its prior mean of zero. The judge facet is
    discarded entirely — judges' main effects and the model×judge
    interaction are *not* used in the prediction.

    Used as a baseline to test whether the added complexity of the
    full random-effects model (judges as a random facet, model×judge
    interaction) has out-of-sample generalization value beyond a
    simple Rasch decomposition. The expected aggregate R² ceiling is
    σ²_m / σ²_total — the share of variance attributable to the
    model main effect alone.

    Shares the fold split with ``cross_validate_items`` when called
    with the same seed, so per-fold and aggregate differences are
    paired and directly comparable.

    Args:
        df: balanced long-form panel; same shape as cross_validate_items.
        k: number of folds (default 5).
        seed: RNG seed for item shuffling (default CV_SEED).

    Returns:
        DataFrame with k + 1 rows, columns
        ``fold, n_test_cells, r2, rmse``. The aggregate row has
        ``fold='ALL'``.
    """
    items = sorted(df["item_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(items)
    folds = np.array_split(shuffled, k)

    rows: list[dict] = []
    all_y: list[np.ndarray] = []
    all_yhat: list[np.ndarray] = []

    for fold_idx, test_items in enumerate(folds):
        test_set = set(test_items.tolist())
        train_mask = ~df["item_id"].isin(test_set)
        train_df = df[train_mask]
        test_df = df[~train_mask]

        # Train-set model main effect: per-model mean across all (j, i)
        # cells in the training items. The judge facet is discarded.
        train_m = train_df.groupby("model")["score"].mean()
        train_global_mean = train_df["score"].mean()
        yhat = np.array([
            train_m.get(m_, train_global_mean) for m_ in test_df["model"]
        ])
        y = test_df["score"].values

        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        rmse = float(np.sqrt(sse / len(y)))

        rows.append({
            "fold": fold_idx,
            "n_test_cells": int(len(y)),
            "r2": r2,
            "rmse": rmse,
        })
        all_y.append(y)
        all_yhat.append(yhat)

    pooled_y = np.concatenate(all_y)
    pooled_yhat = np.concatenate(all_yhat)
    agg_sse = float(np.sum((pooled_y - pooled_yhat) ** 2))
    agg_sst = float(np.sum((pooled_y - pooled_y.mean()) ** 2))
    rows.append({
        "fold": "ALL",
        "n_test_cells": int(len(pooled_y)),
        "r2": 1.0 - agg_sse / agg_sst if agg_sst > 0 else float("nan"),
        "rmse": float(np.sqrt(agg_sse / len(pooled_y))),
    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI: fit both benchmarks at their primary designs and write summary tables
# -----------------------------------------------------------------------------

WILDBENCH_JUDGES = ("gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13")
ARENA_HARD_JUDGES = (
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "gemini-1.5-pro-api-0514",
    "gpt-4-1106-preview",
    "llama-3-70b-instruct",
)
BIGGEN_BENCH_JUDGES = (
    "gpt-4", "gpt-4-04-turbo", "claude",
    "prometheus-8x7b", "prometheus-8x7b-bgb",
)


# Canonical benchmark registry used by gstudy + dstudy + viz CLIs.
BENCHMARKS: dict[str, tuple[Path, tuple[str, ...]]] = {
    "wildbench":    (PROCESSED_DIR / "wildbench_long.parquet",   WILDBENCH_JUDGES),
    "arena_hard":   (PROCESSED_DIR / "arena_hard_long.parquet",  ARENA_HARD_JUDGES),
    "biggen_bench": (PROCESSED_DIR / "biggen_bench_long.parquet", BIGGEN_BENCH_JUDGES),
}


def _run_one(
    name: str,
    parquet: Path,
    judges: tuple[str, ...],
    n_boot: int,
) -> tuple[GStudyFit, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(parquet)
    balanced = balanced_panel(df, judges=judges)
    n_m = balanced["model"].nunique()
    n_j = balanced["judge"].nunique()
    n_i = balanced["item_id"].nunique()
    print(f"[gstudy] {name}: balanced panel m={n_m}, j={n_j}, i={n_i} "
          f"({len(balanced):,} rows)")
    point = fit(balanced)
    boot = bootstrap(balanced, n_reps=n_boot)
    summary = summarize_bootstrap(boot)
    cv = cross_validate_items(balanced, k=5)
    cv_rasch = cross_validate_items_rasch_baseline(balanced, k=5)
    return point, boot, summary, cv, cv_rasch


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    n_boot = 1_000

    runs = [(name, *cfg) for name, cfg in BENCHMARKS.items()]
    all_rows = []
    all_cv_rows: list[dict] = []
    for name, parquet, judges in runs:
        point, boot, summary, cv, cv_rasch = _run_one(
            name, parquet, judges, n_boot=n_boot
        )
        row = {"benchmark": name, **point.as_row()}
        all_rows.append(row)
        # Per-benchmark CI table.
        summary.insert(0, "benchmark", name)
        summary.to_csv(TABLES_DIR / f"gstudy_{name}_bootstrap.csv", index=False)
        # Pretty-print to stdout.
        print(f"[gstudy] {name} variance components (point | clipped to ≥0):")
        for k in COMPONENTS:
            raw = point.components[k]
            clip = point.components_clipped[k]
            print(f"    σ²_{k:<6s} = {raw:>12.6f}   (clipped: {clip:.6f}, "
                  f"share: {point.share()[k]:6.2%})")
        if point.negative_flags:
            print(f"    [warning] negative raw estimates: {point.negative_flags}")
        # Goodness-of-fit: in-sample structural R² + leave-items-out CV
        # for both the full predictor and the Rasch baseline.
        in_sample_r2 = 1.0 - point.components_clipped["mji_e"] / point.total()
        agg_full = cv[cv["fold"] == "ALL"].iloc[0]
        agg_rasch = cv_rasch[cv_rasch["fold"] == "ALL"].iloc[0]
        gap = agg_full["r2"] - agg_rasch["r2"]
        print(f"[gstudy] {name} fit: in-sample R²_struct = {in_sample_r2:.4f}; "
              f"CV R² full = {agg_full['r2']:.4f}, Rasch = {agg_rasch['r2']:.4f} "
              f"(Δ = {gap:+.4f}; RMSE full = {agg_full['rmse']:.4f}, "
              f"Rasch = {agg_rasch['rmse']:.4f}; "
              f"n_test_cells = {int(agg_full['n_test_cells']):,})")
        cv = cv.copy()
        cv.insert(0, "benchmark", name)
        cv.insert(1, "predictor", "full_random_effects")
        all_cv_rows.extend(cv.to_dict("records"))
        cv_rasch = cv_rasch.copy()
        cv_rasch.insert(0, "benchmark", name)
        cv_rasch.insert(1, "predictor", "rasch_baseline")
        all_cv_rows.extend(cv_rasch.to_dict("records"))

    pd.DataFrame(all_rows).to_csv(TABLES_DIR / "gstudy_point_estimates.csv", index=False)
    pd.DataFrame(all_cv_rows).to_csv(TABLES_DIR / "gof_cv.csv", index=False)
    print(f"[gstudy] wrote summary tables to {TABLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

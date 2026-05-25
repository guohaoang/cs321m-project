"""Smoke tests on the ingested long-form parquets.

These guard against schema drift in upstream releases — if a future WildBench
or Arena-Hard re-release changes row counts or column types, the next ingest
run will fail loudly here instead of silently propagating bad numbers into
the variance components.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WB_PARQUET = REPO_ROOT / "data" / "processed" / "wildbench_long.parquet"

# WildBench v2.0522 release facts, locked in 2026-05-24 from the GitHub tree.
# (See PIVOT.md for why WildBench replaces MT-Bench as the primary benchmark.)
WB_EXPECTED_ITEMS = 1024
WB_EXPECTED_JUDGES = {
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "claude-3-5-sonnet-20240620",
}
WB_PRIMARY_JUDGES = {"gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"}
# Lower bound on rows: 100 (judge, model) cells × ~1023 items each. Exact totals
# depend on whether any cells contain 1023 vs 1024 entries in the release;
# we assert a safe floor and let the print summary surface the true count.
WB_MIN_ROWS = 100_000
WB_MIN_COMMON_MODELS = 40  # 41 in the 2026-05-24 snapshot; allow 1 slip
# Several (judge, model) cells are short by up to ~6 items in the release
# (e.g. gpt-4-turbo × Llama-3-8B-Tulu-330K has 1018 rows in the upstream
# JSON, not 1024). Treat anything ≥1000 as full crossing.
WB_PER_CELL_MIN_ROWS = 1000


@pytest.fixture(scope="module")
def wb_df() -> pd.DataFrame:
    if not WB_PARQUET.exists():
        pytest.skip(
            "data/processed/wildbench_long.parquet missing; "
            "run `python -m src.ingest.wildbench` first."
        )
    return pd.read_parquet(WB_PARQUET)


def test_wb_schema(wb_df: pd.DataFrame) -> None:
    expected = ["benchmark", "model", "judge", "item_id", "category", "score"]
    assert list(wb_df.columns) == expected
    assert (wb_df["benchmark"] == "wildbench").all()


def test_wb_row_count(wb_df: pd.DataFrame) -> None:
    assert len(wb_df) >= WB_MIN_ROWS, f"got only {len(wb_df):,} rows"


def test_wb_judges_present(wb_df: pd.DataFrame) -> None:
    found = set(wb_df["judge"].unique())
    missing = WB_EXPECTED_JUDGES - found
    assert not missing, f"missing judges in release: {missing}"


def test_wb_item_count(wb_df: pd.DataFrame) -> None:
    # The release contains 1024 distinct session_ids; one cell may be short
    # by 1 due to a parser miss, which we treat as acceptable.
    n_items = wb_df["item_id"].nunique()
    assert n_items >= WB_EXPECTED_ITEMS - 1, f"only {n_items} unique items"
    assert n_items <= WB_EXPECTED_ITEMS, f"unexpected items {n_items} > {WB_EXPECTED_ITEMS}"


def test_wb_score_range(wb_df: pd.DataFrame) -> None:
    valid = wb_df["score"].dropna()
    assert valid.between(1, 10).all(), "scores must lie in [1, 10] after parsing"
    # NaN tolerance is checked on the two primary judges only. The released
    # claude-3-5-sonnet cell has most rows with parsed_result=null in the
    # upstream pipeline, which is an upstream data-quality artifact rather
    # than an ingest bug; we do not include claude in the primary analysis.
    primary = wb_df[wb_df["judge"].isin(WB_PRIMARY_JUDGES)]
    nan_frac = primary["score"].isna().mean()
    assert nan_frac < 0.01, f"too many NaN scores in primary judges: {nan_frac:.2%}"


def test_wb_common_models_fully_crossed(wb_df: pd.DataFrame) -> None:
    """The two primary judges should jointly score at least 40 common models,
    and each common cell must have ≈1024 rows. This is the design assumption
    the G-study relies on."""
    primary = wb_df[wb_df["judge"].isin(WB_PRIMARY_JUDGES)]
    by_judge = primary.groupby("judge")["model"].unique().to_dict()
    common = set.intersection(*(set(m) for m in by_judge.values()))
    assert len(common) >= WB_MIN_COMMON_MODELS, (
        f"only {len(common)} models scored by both primary judges"
    )
    sub = primary[primary["model"].isin(common)]
    counts = sub.groupby(["judge", "model"]).size()
    assert (counts >= WB_PER_CELL_MIN_ROWS).all(), (
        f"some common cells are short: min={counts.min()}"
    )


def test_wb_categories_populated(wb_df: pd.DataFrame) -> None:
    cat_missing = wb_df["category"].isna().mean()
    assert cat_missing < 0.001, (
        f"category should be populated for ~all items via item-metadata merge; "
        f"missing {cat_missing:.2%}"
    )

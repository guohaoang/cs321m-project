"""BiGGen-Bench ingest.

Downloads ``prometheus-eval/BiGGen-Bench-Results`` and reshapes the wide
schema (one column per judge) into the project's canonical long form:

    columns = [benchmark, model, judge, item_id, category, score]

BiGGen-Bench releases five LLM judges (``gpt4``, ``gpt4_04_turbo``,
``claude``, ``prometheus_8x7b``, ``prometheus_8x7b_bgb``) scoring four
models (``Llama-2-13b-hf``, ``Mistral-7B-Instruct-v0.2``,
``Mixtral-8x7B-Instruct-v0.1``, ``gpt-3.5-turbo-0125``) on ~690 items
spanning eight capability families (used as ``category``). Scores are
1--5 Likert with rubric.

For the Prometheus judges the file stores a 5-element raw-replicate
array; we report the mean as the canonical per-cell score and the
within-cell standard deviation in a separate file so the manuscript can
estimate σ²_e independently of σ²_mji (which the single-rating
WildBench and Arena-Hard designs cannot do).

Also unloads the ``human_eval`` split, which contains the same four
models on a near-identical item set with an additional ``human_score``
column. The result lands at ``data/processed/biggen_human_scores.parquet``
and is the validity anchor for the σ²_m component.

Usage:
    python -m src.ingest.biggen_bench
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

BENCHMARK = "biggen_bench"

# Judge column → canonical judge name in our long-form table.
JUDGE_COLUMNS = {
    "gpt4_score":               "gpt-4",
    "gpt4_04_turbo_score":      "gpt-4-04-turbo",
    "claude_score":             "claude",
    "prometheus_8x7b_score":    "prometheus-8x7b",       # 5-rep array → mean
    "prometheus_8x7b_bgb_score":"prometheus-8x7b-bgb",   # 5-rep array → mean
}

HF_TEST_URL = (
    "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
HF_HUMAN_URL = (
    "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results/"
    "resolve/main/data/human_eval-00000-of-00001.parquet"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "biggen_bench"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_TEST_PATH = RAW_DIR / "test.parquet"
RAW_HUMAN_PATH = RAW_DIR / "human_eval.parquet"
PROCESSED_PATH = PROCESSED_DIR / "biggen_bench_long.parquet"
WITHIN_JUDGE_PATH = PROCESSED_DIR / "biggen_within_judge_std.parquet"
HUMAN_PATH = PROCESSED_DIR / "biggen_human_scores.parquet"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def _agg_score(raw) -> float:
    """Convert one stored cell value (scalar or 5-element array) to a scalar."""
    if raw is None:
        return float("nan")
    if hasattr(raw, "__len__") and not isinstance(raw, (str, bytes)):
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return float("nan")
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _within_judge_std(raw) -> float:
    """Standard deviation of the within-cell replicate scores (Prometheus only)."""
    if raw is None or not hasattr(raw, "__len__"):
        return float("nan")
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def build_long_form(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in test_df.itertuples(index=False):
        for col, judge_name in JUDGE_COLUMNS.items():
            score = _agg_score(getattr(r, col))
            rows.append({
                "benchmark": BENCHMARK,
                "model": r.model_name,
                "judge": judge_name,
                "item_id": r.id,
                "category": r.capability,
                "score": score,
            })
    return pd.DataFrame.from_records(rows)


def build_within_judge_std(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in test_df.itertuples(index=False):
        for col in ("prometheus_8x7b_score", "prometheus_8x7b_bgb_score"):
            std = _within_judge_std(getattr(r, col))
            rows.append({
                "benchmark": BENCHMARK,
                "model": r.model_name,
                "judge": JUDGE_COLUMNS[col],
                "item_id": r.id,
                "category": r.capability,
                "within_judge_std": std,
                "within_judge_var": std * std if not np.isnan(std) else float("nan"),
            })
    return pd.DataFrame.from_records(rows)


def build_human_scores(human_df: pd.DataFrame) -> pd.DataFrame:
    # `human_score` of -1 means the rater did not provide a rating; drop those.
    h = human_df[human_df["human_score"] >= 1][
        ["model_name", "id", "capability", "human_score"]
    ].rename(
        columns={
            "model_name": "model",
            "id": "item_id",
            "capability": "category",
        }
    )
    h["benchmark"] = BENCHMARK
    return h[["benchmark", "model", "item_id", "category", "human_score"]]


def main() -> int:
    print(f"[biggen_bench] downloading parquets to {RAW_DIR}")
    if not _download(HF_TEST_URL, RAW_TEST_PATH):
        raise RuntimeError("Failed to download BiGGen-Bench test split.")
    if not _download(HF_HUMAN_URL, RAW_HUMAN_PATH):
        raise RuntimeError("Failed to download BiGGen-Bench human_eval split.")

    test_df = pd.read_parquet(RAW_TEST_PATH)
    human_df = pd.read_parquet(RAW_HUMAN_PATH)

    long_df = build_long_form(test_df)
    within_df = build_within_judge_std(test_df)
    human_score_df = build_human_scores(human_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(PROCESSED_PATH, index=False)
    within_df.to_parquet(WITHIN_JUDGE_PATH, index=False)
    human_score_df.to_parquet(HUMAN_PATH, index=False)

    print(f"[biggen_bench] wrote {len(long_df):,} rows to {PROCESSED_PATH}")
    print("[biggen_bench] per-judge model coverage:")
    print(long_df.groupby("judge")["model"].nunique().to_string())
    print(f"[biggen_bench] item count: {long_df['item_id'].nunique()}")
    print(f"[biggen_bench] NaN scores: {long_df['score'].isna().sum()}")
    print(f"[biggen_bench] categories: {sorted(long_df['category'].unique())}")

    # Within-judge variance summary
    valid = within_df.dropna(subset=["within_judge_var"])
    if len(valid) > 0:
        print(f"[biggen_bench] within-judge variance (Prometheus, 5-rep): "
              f"mean σ²_e ≈ {valid['within_judge_var'].mean():.4f}")
    print(f"[biggen_bench] human-anchor rows (after -1 drop): {len(human_score_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

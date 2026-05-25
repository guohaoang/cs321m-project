"""WildBench v2.0522 ingest.

Downloads the public per-(judge, model) Likert score files from
``allenai/WildBench`` and the v2 item metadata from the matching HF dataset,
then unifies them into one long-form parquet:

    columns = [benchmark, model, judge, item_id, category, score]

This is the primary-benchmark ingest. The original pre-analysis plan named
MT-Bench, but only ``gpt-4_single.jsonl`` is publicly released for MT-Bench;
see ``PIVOT.md`` for the substitution rationale.

Usage:
    python -m src.ingest.wildbench
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BENCHMARK = "wildbench"
RELEASE = "v2.0522"

# All three judges that appear under ``score.v2/`` in the release. The third
# (claude-3-5-sonnet) judged only one model; we still download it so the
# downstream analysis can choose whether to use it.
JUDGES = (
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "claude-3-5-sonnet-20240620",
)

# Models scored by at least one of the listed judges in
# eval_results/v2.0522/score.v2/. Sourced from the GitHub tree on 2026-05-24.
# Kept as a static list so an offline re-run does not need to hit the GitHub
# tree API.
MODELS: tuple[str, ...] = (
    "Athene-70B",
    "Hermes-2-Theta-Llama-3-8B",
    "Llama-2-70b-chat-hf",
    "Llama-2-7b-chat-hf",
    "Llama-3-8B-Magpie-Align-v0.1",
    "Llama-3-8B-OpenHermes-243K",
    "Llama-3-8B-ShareGPT-112K",
    "Llama-3-8B-Tulu-330K",
    "Llama-3-8B-Ultrachat-200K",
    "Llama-3-8B-WildChat",
    "Llama-3-8B-WizardLM-196K",
    "Llama-3-Instruct-8B-SimPO",
    "Llama-3-Instruct-8B-SimPO-ExPO",
    "Llama-3-Instruct-8B-SimPO-v0.2",
    "Magpie-Pro-SFT-v0.1",
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "Mistral-Large-2",
    "Mistral-Nemo-Instruct-2407",
    "mistral-large-2402",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Phi-3-medium-128k-instruct",
    "Phi-3-mini-128k-instruct",
    "Qwen1.5-72B-Chat",
    "Qwen1.5-72B-Chat-greedy",
    "Qwen1.5-7B-Chat@together",
    "Qwen2-72B-Instruct",
    "SELM-Llama-3-8B-Instruct-iter-3",
    "SELM-Zephyr-7B-iter-3",
    "Starling-LM-7B-beta",
    "Starling-LM-7B-beta-ExPO",
    "Yi-1.5-34B-Chat",
    "Yi-1.5-6B-Chat",
    "Yi-1.5-9B-Chat",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "command-r",
    "command-r-plus",
    "dbrx-instruct@together",
    "deepseek-coder-v2",
    "deepseekv2-chat",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemma-2b-it",
    "gemma-7b-it",
    "glm-4-9b-chat",
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "nemotron-4-340b-instruct",
    "neo_7b_instruct_v0.1",
    "neo_7b_instruct_v0.1-ExPO",
    "reka-core-20240501",
    "reka-edge",
    "reka-flash-20240226",
    "tulu-2-dpo-70b",
    "yi-1.5-34b-chat-original",
    "yi-large",
    "yi-large-preview",
)

GH_RAW = "https://raw.githubusercontent.com/allenai/WildBench/main"
HF_ITEMS_URL = (
    "https://huggingface.co/datasets/allenai/WildBench/resolve/main/v2/"
    "test-00000-of-00001.parquet"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wildbench"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "wildbench_long.parquet"
ITEMS_RAW_PATH = RAW_DIR / "items_v2.parquet"


def _download(url: str, dest: Path) -> bool:
    """Cache-aware download. Returns True if the file is present at the end."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return False  # this (judge, model) cell was not released
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def _score_url(judge: str, model: str) -> str:
    return f"{GH_RAW}/eval_results/{RELEASE}/score.v2/eval={judge}/{model}.json"


def _local_path(judge: str, model: str) -> Path:
    return RAW_DIR / "score.v2" / f"eval={judge}" / f"{model}.json"


def download_all() -> dict[tuple[str, str], Path]:
    """Download every (judge, model) cell that exists. Returns the located paths."""
    located: dict[tuple[str, str], Path] = {}
    for judge in JUDGES:
        for model in MODELS:
            dest = _local_path(judge, model)
            if _download(_score_url(judge, model), dest):
                located[(judge, model)] = dest
    if not _download(HF_ITEMS_URL, ITEMS_RAW_PATH):
        raise RuntimeError("Failed to download WildBench v2 item metadata parquet.")
    return located


def _parse_score(raw) -> float:
    """WildBench stores score as a string like ``\"7\"``; some judges occasionally
    return non-integer text. Anything unparseable becomes NaN so the analysis
    layer decides how to handle missing cells."""
    if raw is None:
        return float("nan")
    try:
        v = int(str(raw).strip())
    except (ValueError, TypeError):
        return float("nan")
    return float(v) if 1 <= v <= 10 else float("nan")


def _extract_score(entry: dict) -> float:
    """Pull a score from a WildBench score.v2 row.

    GPT-judge files populate the top-level ``score`` field; the Claude file
    leaves it null and stores the score inside ``parsed_result.score``. Try
    both before declaring the cell unparseable.
    """
    top = entry.get("score")
    if top is not None and str(top).strip() != "":
        return _parse_score(top)
    parsed = entry.get("parsed_result")
    if isinstance(parsed, dict):
        return _parse_score(parsed.get("score"))
    return float("nan")


def _rows_from_cell(judge: str, model: str, path: Path) -> Iterable[dict]:
    with path.open() as f:
        cell = json.load(f)
    for entry in cell:
        yield {
            "benchmark": BENCHMARK,
            "model": model,
            "judge": judge,
            "item_id": entry["session_id"],
            "score": _extract_score(entry),
        }


def build_long_form(located: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows = []
    for (judge, model), path in located.items():
        rows.extend(_rows_from_cell(judge, model, path))
    df = pd.DataFrame.from_records(rows)

    items = pd.read_parquet(ITEMS_RAW_PATH, columns=["session_id", "primary_tag"])
    items = items.rename(columns={"session_id": "item_id", "primary_tag": "category"})
    df = df.merge(items, on="item_id", how="left")

    return df[["benchmark", "model", "judge", "item_id", "category", "score"]]


def main() -> int:
    print(f"[wildbench] downloading score.v2/ files to {RAW_DIR}")
    located = download_all()
    print(f"[wildbench]   {len(located)} (judge, model) cells released")
    df = build_long_form(located)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"[wildbench] wrote {len(df):,} rows to {PROCESSED_PATH}")
    print("[wildbench] per-judge model coverage:")
    print(df.groupby("judge")["model"].nunique().to_string())
    print(f"[wildbench] item count: {df['item_id'].nunique()}")
    print(f"[wildbench] NaN scores: {df['score'].isna().sum()}")
    print(f"[wildbench] missing categories: {df['category'].isna().sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

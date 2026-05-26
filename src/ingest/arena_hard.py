"""Arena-Hard v0.1 ingest.

Downloads every public per-(judge, model) pairwise-judgment file from the
``lmarena-ai/arena-hard-auto`` HuggingFace dataset and unifies them into one
long-form parquet:

    columns = [benchmark, model, judge, item_id, category, score]

Arena-Hard judges produce a 5-level ordinal preference (``A>>B``, ``A>B``,
``A=B``, ``B>A``, ``B>>A``) between the candidate model and a fixed baseline
(``gpt-4-0314``). Each (judge, model, item) cell contains *two* games with
the positions swapped to control for position bias; we map both to the
candidate model's signed advantage on [-1, +1] and average them, yielding
one score per cell on a 9-level Likert-like scale {-1, -.75, -.5, -.25, 0,
+.25, +.5, +.75, +1}.

The 2026-05-24 release matrix is:

    claude-3-5-sonnet-20240620 :  14 models
    claude-3-opus-20240229     :  31 models
    gemini-1.5-pro-api-0514    :  25 models
    gpt-4-1106-preview         :  72 models
    llama-3-70b-instruct       :  20 models

with an 8-model intersection across all five judges, supporting a fully
crossed ``n_j=5 × m=8 × n_i=500`` G-study design — strictly stronger on the
judge facet than the pre-analysis plan's ``n_j=2`` Arena-Hard design.

Usage:
    python -m src.ingest.arena_hard
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BENCHMARK = "arena_hard"
RELEASE = "v0.1"
BASELINE = "gpt-4-0314"  # all pairwise judgments compare candidate vs this baseline

# Manifest of every released (judge, model) cell on 2026-05-24. Inferred from
# the HF tree at lmarena-ai/arena-hard-auto/data/arena-hard-v0.1/model_judgment/.
# Frozen as a constant so reproduction does not depend on the HF tree API.
JUDGE_TO_MODELS: dict[str, tuple[str, ...]] = {
    "claude-3-5-sonnet-20240620": (
        "claude-2.1",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "gemini-1.5-pro-api-0514",
        "gpt-3.5-turbo-0613",
        "gpt-4-0613",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "llama-2-70b-chat",
        "mistral-large-2402",
        "mistral-medium",
        "mixtral-8x7b-instruct-v0.1",
        "qwen1.5-72b-chat",
        "yi-34b-chat",
    ),
    "claude-3-opus-20240229": (
        "claude-2.0", "claude-2.1", "claude-3-haiku-20240307",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "command-r",
        "dbrx-instruct-preview", "gemini-1.5-pro-api-0409-preview",
        "gemini-pro", "gemma-2b-it", "gemma-7b-it", "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106",
        "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-turbo-2024-04-09",
        "llama-2-70b-chat", "mistral-7b-instruct", "mistral-large-2402",
        "mistral-medium", "mistral-next", "mixtral-8x7b-instruct-v0.1",
        "qwen1.5-72b-chat", "snorkel-mistral-pairrm-dpo",
        "starling-lm-7b-alpha", "starling-lm-7b-beta", "tulu-2-dpo-70b",
        "vicuna-33b", "yi-34b-chat",
    ),
    "gemini-1.5-pro-api-0514": (
        "claude-2.0", "claude-2.1", "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229",
        "dbrx-instruct-preview", "deepseek-coder-v2",
        "gemini-1.5-pro-api-0514", "gemini-pro", "gpt-3.5-turbo-0314",
        "gpt-3.5-turbo-0613", "gpt-4-0314", "gpt-4-0613",
        "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "llama-2-70b-chat",
        "mistral-large-2402", "mistral-medium", "mixtral-8x7b-instruct-v0.1",
        "phi-3-medium-4k-instruct", "qwen1.5-72b-chat",
        "starling-lm-7b-alpha", "tulu-2-dpo-70b", "vicuna-33b", "yi-34b-chat",
    ),
    "gpt-4-1106-preview": (
        "athene-70b-0725", "athene-v2-chat", "claude-2.0", "claude-2.1",
        "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307", "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", "command-r", "command-r-plus",
        "dbrx-instruct-preview", "deepseek-coder-v2",
        "gemini-1.5-flash-api-0514", "gemini-1.5-pro-api-0409-preview",
        "gemini-1.5-pro-api-0514", "gemini-pro", "gemma-1.1-2b-it",
        "gemma-1.1-7b-it", "gemma-2-27b-it", "gemma-2-9b-it", "gemma-2b-it",
        "gemma-7b-it", "glm-4-0116", "glm-4-0520", "glm-4-air",
        "gpt-3.5-turbo-0125", "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106", "gpt-4-0125-preview", "gpt-4-0613",
        "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18", "internlm2-20b-5-chat",
        "internlm2-20b-chat", "llama-2-70b-chat", "llama-3-70b-instruct",
        "llama-3-8b-instruct", "llama-3.1-405b-instruct-fp8",
        "llama-3.1-405b-instruct-fp8-no-sys-prompt",
        "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
        "llama-3.1-nemotron-51b-instruct", "llama-3.1-nemotron-70b-instruct",
        "mistral-7b-instruct", "mistral-large-2402", "mistral-large-2407",
        "mistral-medium", "mistral-next", "mixtral-8x22b-instruct-v0.1",
        "mixtral-8x7b-instruct-v0.1", "o1-mini-2024-09-12",
        "o1-preview-2024-09-12", "phi-3-medium-4k-instruct",
        "phi-3-mini-128k-instruct", "phi-3-small-8k-instruct",
        "qwen1.5-72b-chat", "qwen2-72b-instruct", "qwen2.5-72b-instruct",
        "snorkel-mistral-pairrm-dpo", "snowflake-arctic-instruct",
        "starling-lm-7b-alpha", "starling-lm-7b-beta", "tulu-2-dpo-70b",
        "vicuna-33b", "yi-34b-chat", "yi-large", "yi-large-preview",
        "yi-lightning",
    ),
    "llama-3-70b-instruct": (
        "Llama-2-70b-chat-hf", "Mixtral-8x7B-Instruct-v0.1",
        "Qwen1.5-72B-Chat", "Starling-LM-7B-alpha", "Yi-34B-Chat",
        "claude-2.0", "claude-2.1", "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", "dbrx-instruct", "gemini-1.0-pro",
        "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4-0314",
        "gpt-4-0613", "gpt-4-turbo-2024-04-09", "mistral-large-2402",
        "mistral-medium", "tulu-2-dpo-70b", "vicuna-33b-v1.3",
    ),
}

# Map the 5-level ordinal label to the candidate model's signed advantage
# (positive = model beats baseline) under "Assistant A = model, Assistant B = baseline".
# Position-swapped games (model = Assistant B) flip the sign at parse time.
SCORE_MAP = {"A>>B": 1.0, "A>B": 0.5, "A=B": 0.0, "B>A": -0.5, "B>>A": -1.0}

HF_BASE = "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main"
HF_JUDGMENT_URL = HF_BASE + "/data/arena-hard-{release}/model_judgment/{judge}/{model}.jsonl"
HF_QUESTION_URL = HF_BASE + "/data/arena-hard-{release}/question.jsonl"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "arena_hard"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "arena_hard_long.parquet"
QUESTION_RAW_PATH = RAW_DIR / "question.jsonl"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def _local_path(judge: str, model: str) -> Path:
    return RAW_DIR / "model_judgment" / judge / f"{model}.jsonl"


def download_all() -> dict[tuple[str, str], Path]:
    located: dict[tuple[str, str], Path] = {}
    for judge, models in JUDGE_TO_MODELS.items():
        for model in models:
            dest = _local_path(judge, model)
            url = HF_JUDGMENT_URL.format(release=RELEASE, judge=judge, model=model)
            if _download(url, dest):
                located[(judge, model)] = dest
    if not _download(HF_QUESTION_URL.format(release=RELEASE), QUESTION_RAW_PATH):
        raise RuntimeError("Failed to download Arena-Hard question.jsonl.")
    return located


def _per_cell_score(games: list[dict]) -> float:
    """Average the two position-swapped games into one signed score in [-1, 1].

    Per arena-hard-auto's gen_judgment.py, the saved games list has:

        games[0]: Assistant A = baseline,         Assistant B = candidate
        games[1]: Assistant A = candidate,        Assistant B = baseline

    So a label like ``A>>B`` means *baseline >> candidate* in game[0] (bad
    for the candidate) and *candidate >> baseline* in game[1] (good for
    the candidate). The candidate's signed advantage is therefore
    ``-SCORE_MAP`` in game[0] and ``+SCORE_MAP`` in game[1].
    """
    if not games:
        return float("nan")
    pieces: list[float] = []
    for i, game in enumerate(games):
        raw = game.get("score")
        if raw not in SCORE_MAP:
            continue
        s = SCORE_MAP[raw]
        if i == 0:
            s = -s  # game[0]: candidate is Assistant B
        pieces.append(s)
    if not pieces:
        return float("nan")
    return sum(pieces) / len(pieces)


def _rows_from_cell(judge: str, model: str, path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            yield {
                "benchmark": BENCHMARK,
                "model": model,
                "judge": judge,
                "item_id": row["uid"],
                "score": _per_cell_score(row.get("games", [])),
            }


def build_long_form(located: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for (judge, model), path in located.items():
        rows.extend(_rows_from_cell(judge, model, path))
    df = pd.DataFrame.from_records(rows)

    questions = pd.read_json(QUESTION_RAW_PATH, lines=True)
    questions = questions[["uid", "cluster"]].rename(
        columns={"uid": "item_id", "cluster": "category"}
    )
    # Inner join: drop the 3 orphan items that appear only in the
    # claude-3-opus judgment files but were retired from the official
    # 500-item set (uids 6af6c9e3..., d92d1632..., dfbfaf85...).
    df = df.merge(questions, on="item_id", how="inner")

    return df[["benchmark", "model", "judge", "item_id", "category", "score"]]


def main() -> int:
    print(f"[arena_hard] downloading model_judgment files to {RAW_DIR}")
    located = download_all()
    print(f"[arena_hard]   {len(located)} (judge, model) cells released")
    df = build_long_form(located)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"[arena_hard] wrote {len(df):,} rows to {PROCESSED_PATH}")
    print("[arena_hard] per-judge model coverage:")
    print(df.groupby("judge")["model"].nunique().to_string())
    print(f"[arena_hard] item count: {df['item_id'].nunique()}")
    print(f"[arena_hard] NaN scores: {df['score'].isna().sum()}")
    print(f"[arena_hard] missing categories: {df['category'].isna().sum()}")
    # Surface the 5-judge intersection so the design fact is in the run log.
    by_judge = df.groupby("judge")["model"].unique().to_dict()
    common = set.intersection(*(set(m) for m in by_judge.values()))
    print(f"[arena_hard] {len(common)}-model intersection across all 5 judges: "
          f"{sorted(common)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

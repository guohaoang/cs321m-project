# Pivot: from MT-Bench to WildBench as primary benchmark

**Date:** 2026-05-24
**Decision by:** Henry Ang (project author)
**Affects:** §4 of `pre_analysis_plan.tex`, all downstream MT-Bench inputs

## Finding

The pre-analysis plan committed to MT-Bench as the primary benchmark with **four LLM judges** — `gpt-4`, `gpt-3.5-turbo`, `claude-v1` (from FastChat `model_judgment/{judge}_single.jsonl`) and `prometheus-2-7b`. Implementation-time scouting confirmed that **only `gpt-4_single.jsonl` is publicly released**:

| Source | Status |
|---|---|
| `lmsys/mt-bench` HF Space `data/mt_bench/model_judgment/` | only `gpt-4_single.jsonl` + `gpt-4_pair.jsonl` |
| `lm-sys/FastChat` GitHub (all branches: `main`, `archive`, `major_cleanup`, `leaderboard`) | judgment files absent — `data/mt_bench/` contains only `question.jsonl`, `reference_answer/`, `misc/radar.png` |
| `lmsys/mt_bench_human_judgments` HF dataset | only `human` and `gpt4_pair` splits |
| `prometheus-eval/prometheus-eval` GitHub | `eval/benchmark/data/mt_bench_eval.json` contains 4 models × 80 prompts of *input* + GPT-4 reference scores; Prometheus-2's own scores are computed at runtime and not released as a static file |

`gpt-3.5-turbo_single.jsonl` and `claude-v1_single.jsonl` returned "Entry not found" on the HF Space and 404 on every FastChat branch. A fully crossed `m × j × i` design requires the same models scored by every judge; with only one LLM judge available, $\sigma^2_j, \sigma^2_{mj}, \sigma^2_{ji}$ are all unidentifiable on MT-Bench.

The HANDOFF §2 hard-constraint **"no new model inference, no API calls"** rules out filling the gap by running the missing judges ourselves.

## Substitution

| Role | Original | Replacement |
|---|---|---|
| Primary benchmark | MT-Bench: m=6, n_j=4, n_i=80×2 turns=160 | **WildBench v2.0522**: m=41, n_j=2, n_i=1024 |
| Secondary benchmark | Arena-Hard v0.1: m=8, n_j=2, n_i=500 | Arena-Hard v0.1 with expanded judge facet (see addendum below) |
| Human anchor (primary) | `lmsys/mt_bench_human_judgments` Bradley–Terry | WildBench's length-penalized GPT-4 Elo leaderboard (see WildBench README) |
| Human anchor (secondary) | Chatbot Arena Elo | Chatbot Arena Elo (unchanged) |

### Why WildBench

- **Data is released as static per-(judge, model) JSON files** in `allenai/WildBench` GitHub: `eval_results/v2.0522/score.v2/eval={judge}/{model}.json`. Each row is one (item, score) pair on a Likert 1–10 scale.
- **Two LLM judges:** `gpt-4-turbo-2024-04-09` and `gpt-4o-2024-05-13`. (Claude-3-5-sonnet judged only one model in the release.)
- **41 models scored by both judges** — far more degrees of freedom for $\sigma^2_m$ than MT-Bench's 6.
- **1024 items** vs MT-Bench's 80 (×2 turns) — far more degrees of freedom for $\sigma^2_i$.
- Spot-check showed judges disagree on **~57%** of items for a representative model — $\sigma^2_{mj}$ and $\sigma^2_{ji}$ will be non-trivial.
- WildBench prompts are real Chatbot Arena conversations (closer to Arena-Hard's distribution than MT-Bench's hand-written prompts), which slightly weakens the "different item style" axis of cross-benchmark differentiation; this trade is acknowledged below.

### Impact on pre-registered hypotheses

| Hypothesis | Surviveas-is? | Reason |
|---|---|---|
| H1: $\sigma^2_{mj} \geq 0.1 \cdot \sigma^2_m$ on both benchmarks | ✓ | $\sigma^2_{mj}$ is identifiable with $n_j=2$ and large $m$. WildBench's m=41 makes this strictly stronger than the MT-Bench m=6 design would have been. |
| H2: adjacent-pair RFPs exceed 20% at modal practice | ✓ | "Modal practice" item count is benchmark-specific (80 for MT-Bench, the original framing; will be 500 or 1024 for the new benchmarks). The hypothesis text in the final manuscript will read "at $n_j=1$ and each benchmark's published item count." |
| H3: variance-component ordering is preserved across benchmarks | ✓ | Now both benchmarks have $n_j=2$, which makes the cross-benchmark comparison more *symmetric* than the original asymmetric (4-judge vs 2-judge) design. The qualitative ordering test is unchanged. |

### Costs of the pivot

- **$\sigma^2_j$ is descriptively reported, never inferentially.** Both benchmarks now have $n_j=2$, the regime the pre-analysis plan §5 already flagged as "$\sigma^2_j$ effectively unidentified" for Arena-Hard. The new manuscript will report $\sigma^2_j$ as a point estimate without a bootstrap CI on both benchmarks, and the discussion will frame it as descriptive.
- **The 4-judge × 1024-item joint design is no longer the headline claim.** Replaced by "two independent 2-judge designs that replicate the same qualitative pattern."
- **Prometheus-2 facet dropped.** No 2024+ frontier open-evaluator joins the analysis; the conclusions are conditional on the **gpt-4-turbo / gpt-4o / claude-3-opus / gpt-4-1106-preview** judge pool.
- **GPT-4 single-vs-pairwise probe dropped.** The "scoring-format facet" was already a partial substitute for the rubric-prompt facet; both are out of scope now.
- **Cross-benchmark item-style differentiation is weaker.** WildBench and Arena-Hard both source from Chatbot Arena conversations, whereas MT-Bench is hand-written. The H3 replication test is now between two "in-the-wild" benchmarks rather than "in-the-wild vs hand-curated." This narrows the population of items the conclusions generalize to.

## Arena-Hard ingest addendum (2026-05-24)

Implementation-time scouting of the `lmarena-ai/arena-hard-auto` HF dataset surfaced a much richer Arena-Hard v0.1 release than the pre-analysis plan assumed:

| Judge | Models judged |
|---|---|
| `gpt-4-1106-preview` | 72 |
| `claude-3-opus-20240229` | 31 |
| `gemini-1.5-pro-api-0514` | 25 |
| `llama-3-70b-instruct` | 20 |
| `claude-3-5-sonnet-20240620` | 14 |

The fully crossed 5-judge intersection contains **8 models** (`claude-2.1`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `gpt-3.5-turbo-0613`, `gpt-4-0613`, `gpt-4-turbo-2024-04-09`, `mistral-large-2402`, `mistral-medium`) — exactly matching the m=8 the plan named for Arena-Hard, but at **n_j=5 instead of n_j=2**. This makes σ²_j inferentially identifiable on Arena-Hard with bootstrap CIs, *upgrading* Arena-Hard's role in the project: it becomes the n_j-rich design that the original MT-Bench was supposed to provide. WildBench remains the m-rich and n_i-rich design.

The analysis therefore fits each variance-component model twice on each benchmark:
- **WildBench (m=41, n_j=2, n_i=1024)** — σ²_m and σ²_i well-identified, σ²_j point-only.
- **Arena-Hard 5-judge (m=8, n_j=5, n_i=500)** — every component including σ²_j has a bootstrap CI.
- **Arena-Hard 2-judge subset** matching {gpt-4-1106-preview, claude-3-opus} on m=31 — apples-to-apples comparison to WildBench's 2-judge design.

Score scales differ between the two benchmarks (WildBench: 1–10 Likert; Arena-Hard: signed pairwise advantage in [−1, +1]). The G-study uses raw scores within each benchmark; cross-benchmark hypotheses (H3) compare *relative* variance-component ratios, not absolute magnitudes. A rank-transformed sensitivity check will be reported alongside the headline raw-scale numbers.

## Manuscript implications

A 1-paragraph addendum will appear in §4 of the final manuscript explicitly stating the data-availability finding and the substitution, and the discussion will note the narrower item-style generalization. This is consistent with pre-registration norms: pre-registration commits to a *transparent, falsifiable* analysis, not to running an analysis on data that turns out to be unavailable.

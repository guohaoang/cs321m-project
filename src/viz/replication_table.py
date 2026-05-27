"""Cross-benchmark replication table.

Pre-registered H3 asks whether the relative ordering of (σ²_m, σ²_j,
σ²_i, σ²_mj) is preserved between the two benchmarks. We emit:

  * A side-by-side LaTeX-compatible CSV of variance components, shares,
    and bootstrap 95% CIs for both benchmarks.
  * Spearman rank correlation across all seven components, and again
    restricted to the four pre-registered components, with both signed
    and absolute interpretations.

Usage:
    python -m src.viz.replication_table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.analysis.gstudy import (
    BENCHMARKS,
    COMPONENTS,
    balanced_panel,
    bootstrap,
    fit,
    summarize_bootstrap,
)
from itertools import combinations

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"

H3_COMPONENTS = ("m", "j", "i", "mj")


def _run_benchmark(parquet: Path, judges: tuple[str, ...], n_boot: int):
    df = pd.read_parquet(parquet)
    panel = balanced_panel(df, judges=judges)
    g = fit(panel)
    boot = bootstrap(panel, n_reps=n_boot)
    summary = summarize_bootstrap(boot)
    return g, summary


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    benches = BENCHMARKS
    n_boot = 1_000

    rows = []
    shares = {}
    for name, (parquet, judges) in benches.items():
        g, summary = _run_benchmark(parquet, judges, n_boot=n_boot)
        s = g.share()
        shares[name] = s
        for c in COMPONENTS:
            ci = summary[summary["component"] == c].iloc[0]
            rows.append({
                "component": c,
                "benchmark": name,
                "n_m": g.n_m,
                "n_j": g.n_j,
                "n_i": g.n_i,
                "var_point": g.components_clipped[c],
                "share": s[c],
                "var_ci_low": ci["ci_low"],
                "var_ci_high": ci["ci_high"],
            })
    out = pd.DataFrame(rows)

    # Pivot wide for the side-by-side LaTeX/CSV deliverable.
    wide = out.pivot(index="component", columns="benchmark",
                     values=["var_point", "share", "var_ci_low", "var_ci_high"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reindex(list(COMPONENTS))
    wide.to_csv(TABLES_DIR / "replication_variance_components.csv")

    # Pairwise Spearman over every benchmark pair, both for the full
    # 7-component decomposition and the H3 pre-registered subset (m, j, i, mj).
    rows_sp = []
    for a, b in combinations(shares.keys(), 2):
        for scope_name, comps in [
            ("all 7 components", COMPONENTS),
            ("H3 pre-registered (m, j, i, mj)", H3_COMPONENTS),
        ]:
            va = [shares[a][c] for c in comps]
            vb = [shares[b][c] for c in comps]
            rho, p = spearmanr(va, vb)
            rows_sp.append({
                "benchmark_A": a,
                "benchmark_B": b,
                "scope": scope_name,
                "spearman_rho": rho,
                "p_value": p,
                "A_ranks_by_share": ",".join(
                    [c for c, _ in sorted(zip(comps, va), key=lambda x: -x[1])]),
                "B_ranks_by_share": ",".join(
                    [c for c, _ in sorted(zip(comps, vb), key=lambda x: -x[1])]),
            })
    spearman_df = pd.DataFrame(rows_sp)
    spearman_df.to_csv(TABLES_DIR / "replication_spearman.csv", index=False)

    print("[replication] variance components (share %):")
    print(wide.round(4))
    print()
    print("[replication] pairwise Spearman ρ (share-based ordering):")
    print(spearman_df.to_string(index=False))
    print()
    # H3 verdict per pair and overall (mean ρ on the pre-registered subset)
    h3_only = spearman_df[spearman_df["scope"] == "H3 pre-registered (m, j, i, mj)"]
    mean_rho = h3_only["spearman_rho"].mean()
    overall_verdict = "REPLICATES" if mean_rho >= 0.7 else "DOES NOT REPLICATE"
    print(f"[replication] H3 pairs:")
    for r in h3_only.itertuples():
        v = "REPLICATES" if r.spearman_rho >= 0.7 else "fails"
        print(f"   {r.benchmark_A:>13} vs {r.benchmark_B:<13}  ρ={r.spearman_rho:+.3f}  {v}")
    print(f"[replication] H3 mean ρ across {len(h3_only)} pairs = {mean_rho:.3f} "
          f"-> {overall_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    ARENA_HARD_JUDGES,
    COMPONENTS,
    WILDBENCH_JUDGES,
    balanced_panel,
    bootstrap,
    fit,
    summarize_bootstrap,
)

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
    benches = {
        "wildbench": (PROCESSED_DIR / "wildbench_long.parquet", WILDBENCH_JUDGES),
        "arena_hard": (PROCESSED_DIR / "arena_hard_long.parquet", ARENA_HARD_JUDGES),
    }
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

    # Spearman over all 7 components, and over the 4 pre-registered ones.
    wb_vec = [shares["wildbench"][c] for c in COMPONENTS]
    ah_vec = [shares["arena_hard"][c] for c in COMPONENTS]
    rho_all, p_all = spearmanr(wb_vec, ah_vec)

    wb_h3 = [shares["wildbench"][c] for c in H3_COMPONENTS]
    ah_h3 = [shares["arena_hard"][c] for c in H3_COMPONENTS]
    rho_h3, p_h3 = spearmanr(wb_h3, ah_h3)

    spearman_df = pd.DataFrame([
        {"scope": "all 7 components",
         "spearman_rho": rho_all, "p_value": p_all,
         "components": ",".join(COMPONENTS),
         "wb_ranks_by_share": ",".join(
            [c for c, _ in sorted(zip(COMPONENTS, wb_vec), key=lambda x: -x[1])]),
         "ah_ranks_by_share": ",".join(
            [c for c, _ in sorted(zip(COMPONENTS, ah_vec), key=lambda x: -x[1])])},
        {"scope": "H3 pre-registered (m, j, i, mj)",
         "spearman_rho": rho_h3, "p_value": p_h3,
         "components": ",".join(H3_COMPONENTS),
         "wb_ranks_by_share": ",".join(
            [c for c, _ in sorted(zip(H3_COMPONENTS, wb_h3), key=lambda x: -x[1])]),
         "ah_ranks_by_share": ",".join(
            [c for c, _ in sorted(zip(H3_COMPONENTS, ah_h3), key=lambda x: -x[1])])},
    ])
    spearman_df.to_csv(TABLES_DIR / "replication_spearman.csv", index=False)

    print("[replication] variance components (share %):")
    print(wide.round(4))
    print()
    print("[replication] Spearman ρ (share-based ordering):")
    print(spearman_df.to_string(index=False))
    print()
    verdict_all = "REPLICATES" if rho_all >= 0.7 else "DOES NOT REPLICATE"
    verdict_h3 = "REPLICATES" if rho_h3 >= 0.7 else "DOES NOT REPLICATE"
    print(f"[replication] H3 verdict (all 7): ρ={rho_all:.3f} -> {verdict_all}")
    print(f"[replication] H3 verdict (m,j,i,mj): ρ={rho_h3:.3f} -> {verdict_h3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

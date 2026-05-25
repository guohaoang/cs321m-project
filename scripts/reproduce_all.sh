#!/usr/bin/env bash
# Reproduce every figure and table for the manuscript.
# Run from the repo root: `bash scripts/reproduce_all.sh`
# Idempotent: re-running picks up where it left off (raw downloads are cached).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/7] ingest WildBench"
$PYTHON -m src.ingest.wildbench

echo "==> [2/7] ingest Arena-Hard"
$PYTHON -m src.ingest.arena_hard

echo "==> [3/7] human anchors (WildBench Elo + Arena Elo)"
echo "    (stub — not yet implemented)"
# $PYTHON -m src.ingest.human_anchors

echo "==> [4/7] G-study (variance components)"
echo "    (stub — not yet implemented)"
# $PYTHON -m src.analysis.gstudy

echo "==> [5/7] D-study + RFP sweep"
echo "    (stub — not yet implemented)"
# $PYTHON -m src.analysis.dstudy

echo "==> [6/7] figures"
echo "    (stub — not yet implemented)"
# $PYTHON -m src.viz.variance_pie
# $PYTHON -m src.viz.rfp_heatmap
# $PYTHON -m src.viz.replication_table

echo "==> [7/7] tests"
$PYTHON -m pytest tests/ -q

echo "done"

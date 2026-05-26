"""Tests for the D-study + RFP module.

The headline test is that the analytical RFP under the Normal approximation
agrees with the parametric-bootstrap RFP across a range of (Δ, n_j, n_i)
points. This validates the math the pre-analysis plan binds the project to.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.dstudy import (
    adjacent_pairs,
    diff_score_sem,
    g_coefficient,
    Leaderboard,
    phi_coefficient,
    relative_error_var,
    rfp_analytical,
    rfp_bootstrap,
)
import pandas as pd


VC_TYPICAL = {
    "m": 1.0,
    "j": 0.4,
    "i": 0.6,
    "mj": 0.15,
    "mi": 0.30,
    "ji": 0.10,
    "mji_e": 0.50,
}


# -----------------------------------------------------------------------------
# Sanity on scalar reliability
# -----------------------------------------------------------------------------

def test_rel_var_decreases_with_n_j() -> None:
    rel1 = relative_error_var(VC_TYPICAL, n_j=1, n_i=100)
    rel5 = relative_error_var(VC_TYPICAL, n_j=5, n_i=100)
    assert rel1 > rel5


def test_g_coefficient_in_unit_interval_and_monotone() -> None:
    g1 = g_coefficient(VC_TYPICAL, n_j=1, n_i=80)
    g5 = g_coefficient(VC_TYPICAL, n_j=5, n_i=500)
    assert 0.0 <= g1 <= 1.0
    assert 0.0 <= g5 <= 1.0
    assert g5 > g1


def test_phi_le_g_coefficient() -> None:
    """Φ uses a wider error variance so Φ ≤ E[ρ²] always."""
    for n_j, n_i in [(1, 80), (2, 500), (5, 1000)]:
        g = g_coefficient(VC_TYPICAL, n_j, n_i)
        phi = phi_coefficient(VC_TYPICAL, n_j, n_i)
        assert phi <= g + 1e-12


# -----------------------------------------------------------------------------
# Pairwise RFP — analytical bounds and bootstrap agreement
# -----------------------------------------------------------------------------

def test_rfp_zero_for_identical_models() -> None:
    rfp = rfp_analytical(0.5, 0.5, VC_TYPICAL, n_j=2, n_i=80)
    # μ_A == μ_B → z = 0 → RFP = 0.5
    assert abs(rfp - 0.5) < 1e-9


def test_rfp_drops_as_delta_grows() -> None:
    a = rfp_analytical(1.0, 0.5, VC_TYPICAL, n_j=2, n_i=80)
    b = rfp_analytical(1.0, 0.9, VC_TYPICAL, n_j=2, n_i=80)
    assert b > a  # closer means → more flips


def test_rfp_drops_as_design_grows() -> None:
    small = rfp_analytical(1.0, 0.8, VC_TYPICAL, n_j=1, n_i=40)
    big = rfp_analytical(1.0, 0.8, VC_TYPICAL, n_j=5, n_i=500)
    assert big < small


@pytest.mark.parametrize("delta,n_j,n_i", [
    (0.3, 2, 80),
    (0.5, 1, 160),
    (0.2, 3, 500),
])
def test_rfp_analytical_matches_bootstrap(delta: float, n_j: int, n_i: int) -> None:
    """The Normal approximation should agree with the parametric bootstrap
    to within Monte-Carlo error at typical operating points."""
    rfp_a = rfp_analytical(0.0, -delta, VC_TYPICAL, n_j=n_j, n_i=n_i)
    rfp_b = rfp_bootstrap(0.0, -delta, VC_TYPICAL, n_j=n_j, n_i=n_i,
                          n_sim=4_000, seed=42)
    # The MC SE of a probability p with N=4000 is sqrt(p(1-p)/N) ≤ 0.008.
    # Allow 2× that plus a small slack.
    assert abs(rfp_a - rfp_b) < 0.025, (
        f"analytical={rfp_a:.4f}, bootstrap={rfp_b:.4f}, "
        f"delta={delta}, n_j={n_j}, n_i={n_i}"
    )


def test_rfp_symmetric_in_argument_order() -> None:
    """RFP conditions on the higher-μ model, so swapping A and B must
    return the same number."""
    a = rfp_analytical(1.0, 0.6, VC_TYPICAL, n_j=2, n_i=80)
    b = rfp_analytical(0.6, 1.0, VC_TYPICAL, n_j=2, n_i=80)
    assert abs(a - b) < 1e-12


# -----------------------------------------------------------------------------
# Leaderboard / adjacent pairs
# -----------------------------------------------------------------------------

def test_adjacent_pairs_match_ordering() -> None:
    means = pd.Series({"X": 1.0, "Y": 0.7, "Z": 0.4})
    lb = Leaderboard(means=means.sort_values(ascending=False),
                     sorted_models=["X", "Y", "Z"])
    assert adjacent_pairs(lb) == [("X", "Y"), ("Y", "Z")]


# -----------------------------------------------------------------------------
# Numeric edge cases
# -----------------------------------------------------------------------------

def test_zero_variance_design_is_perfectly_reliable() -> None:
    zero = {k: 0.0 for k in VC_TYPICAL}
    zero["m"] = 1.0
    assert g_coefficient(zero, 1, 1) == 1.0
    assert phi_coefficient(zero, 1, 1) == 1.0
    assert rfp_analytical(1.0, 0.5, zero, n_j=1, n_i=1) == 0.0

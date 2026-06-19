"""Single source of truth for RNG seeds — referenced from every analysis script."""
from __future__ import annotations

import random

import numpy as np

GLOBAL_SEED = 20260524  # date of pivot decision; arbitrary but fixed
BOOTSTRAP_SEED = 31415
DSTUDY_SEED = 27182
CV_SEED = 20260616  # leave-items-out cross-validation (see gstudy.cross_validate_items)


def fix_all(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

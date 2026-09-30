"""Day-level train/val/test splits (all grid cells share the same split of days).

``blocked_within_season``  (primary)
    Days are cut into contiguous ``BLOCK_DAYS`` blocks inside each season; blocks are
    randomly assigned to train/val/test *within* each season, so every season is present
    in every split (a pure time split leaves post-monsoon with no training data at all,
    which is exactly the seasonal-boundary distortion the Q4 prototype hit: climatology
    RMSE 7.2 C for t2m). Train days within ``EMBARGO_DAYS`` of any val/test day are
    dropped to stop day-to-day weather autocorrelation leaking across the boundary.

``time_forward``  (stress test)  train Jan-Aug, val Sep, test Oct-Dec, as in the prototype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

TRAIN, VAL, TEST, EMBARGO = "train", "val", "test", "embargo"


def blocked_within_season(dates: pd.DatetimeIndex, seed: int = C.SEED) -> pd.Series:
    dates = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(seed)
    season = pd.Series(dates.month).map(C.SEASON_OF_MONTH).values
    split = np.empty(len(dates), dtype=object)
    for s in C.SEASONS:
        idx = np.flatnonzero(season == s)
        if len(idx) == 0:
            continue
        # contiguous runs of this season (winter = Jan-Feb only in a single calendar year)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        blocks = [r[i:i + C.BLOCK_DAYS] for r in runs for i in range(0, len(r), C.BLOCK_DAYS)]
        nb = len(blocks)
        n_test = max(C.MIN_TEST_BLOCKS_PER_SEASON, int(round(C.TEST_FRAC * nb)))
        n_val = max(C.MIN_VAL_BLOCKS_PER_SEASON, int(round(C.VAL_FRAC * nb)))
        # never put the first block of the year in test/val (no history for rolling features)
        order = rng.permutation(nb)
        labels = np.array([TRAIN] * nb, dtype=object)
        eligible = [b for b in order if blocks[b][0] > C.WARMUP_DAYS + max(C.LEADS)]
        labels[eligible[:n_test]] = TEST
        labels[eligible[n_test:n_test + n_val]] = VAL
        for b, lab in zip(blocks, labels):
            split[b] = lab
    split = pd.Series(split, index=dates, name="split")
    return apply_embargo(split)


def apply_embargo(split: pd.Series, embargo: int = C.EMBARGO_DAYS) -> pd.Series:
    held = np.flatnonzero(split.isin([VAL, TEST]).values)
    out = split.copy()
    vals = out.values
    for k in range(1, embargo + 1):
        for j in np.concatenate([held - k, held + k]):
            if 0 <= j < len(vals) and vals[j] == TRAIN:
                vals[j] = EMBARGO
    out[:] = vals
    return out


def time_forward(dates: pd.DatetimeIndex) -> pd.Series:
    dates = pd.DatetimeIndex(dates)
    lab = np.where(dates <= pd.Timestamp(C.TF_TRAIN_END), TRAIN,
                   np.where(dates <= pd.Timestamp(C.TF_VAL_END), VAL, TEST))
    return pd.Series(lab, index=dates, name="split")


SPLITTERS = {"blocked": blocked_within_season, "time_forward": time_forward}


def summarize(split: pd.Series) -> pd.DataFrame:
    season = pd.Series(pd.DatetimeIndex(split.index).month).map(C.SEASON_OF_MONTH).values
    return pd.crosstab(season, split.values).reindex(C.SEASONS)

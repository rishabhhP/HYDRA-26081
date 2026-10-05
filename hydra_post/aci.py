"""Adaptive conformal intervals (Gibbs & Candès 2021, asymmetric form) for HYDRA's 80% range.

Split conformal calibrates once, on the most recent part of the training window. When the regime drifts (monsoon
into post-monsoon) that calibration is stale and coverage drifts too: 86.6% instead of 80% on IMD truth.

Adaptive conformal keeps each tail's working miss level a_t and, after each verified day, moves it by
    a_{t+1} = a_t + gamma * (target - observed_miss_rate_t)
so a tail that misses too often widens and one that never misses narrows. Long-run miss frequency converges to the
target whatever the drift. Bounds are conformal quantiles of a rolling buffer of recent verified scores.

Settings are declared here, before any test-period evaluation, and are not tuned on test data:
    ALPHA = 0.20 (each tail 0.10), GAMMA = 0.03 per day, BUFFER_DAYS = 45.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .models import GBM, _cats, split_time

ALPHA, GAMMA, BUFFER_DAYS = 0.20, 0.03, 45
A_MIN, A_MAX = 0.005, 0.45


def _q(scores: np.ndarray, level: float) -> float:
    s = np.sort(scores[np.isfinite(scores)])
    if not len(s):
        return 0.0
    k = int(min(len(s), max(1, np.ceil((len(s) + 1) * level))))
    return float(s[k - 1])


class AdaptiveInterval:
    def __init__(self, cols, alpha=ALPHA, gamma=GAMMA, buffer_days=BUFFER_DAYS, calib_frac=0.25):
        self.cols, self.alpha, self.gamma, self.buffer_days, self.calib_frac = cols, alpha, gamma, buffer_days, calib_frac
        self.a_lo = self.a_hi = alpha / 2

    def fit(self, tr: pd.DataFrame, state: dict | None = None):
        fit_part, cal = split_time(tr, self.calib_frac)
        a2 = self.alpha / 2
        self.lo_m = HistGradientBoostingRegressor(**GBM, loss="quantile", quantile=a2, categorical_features=_cats(self.cols)).fit(fit_part[self.cols], fit_part["y"])
        self.hi_m = HistGradientBoostingRegressor(**GBM, loss="quantile", quantile=1 - a2, categorical_features=_cats(self.cols)).fit(fit_part[self.cols], fit_part["y"])
        self.buffer = deque()
        for day, g in cal.groupby("date"):
            lo, hi = self.lo_m.predict(g[self.cols]), self.hi_m.predict(g[self.cols])
            self.buffer.append((day, lo - g["y"].to_numpy(), g["y"].to_numpy() - hi))
        self._trim()
        if state:  # carry the adapted miss levels across monthly refits
            self.a_lo, self.a_hi = state["a_lo"], state["a_hi"]
        return self

    def _trim(self):
        while len(self.buffer) > self.buffer_days:
            self.buffer.popleft()

    def state(self) -> dict:
        return {"a_lo": self.a_lo, "a_hi": self.a_hi}

    def run(self, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Sequential by date. Bounds for day d use only scores verified before d; then day d's outcomes update the levels."""
        lo_out = np.full(len(te), np.nan)
        hi_out = np.full(len(te), np.nan)
        trace = []
        pos = {idx: i for i, idx in enumerate(te.index)}
        for day, g in te.sort_values("date").groupby("date", sort=True):
            s_lo = np.concatenate([b[1] for b in self.buffer]) if self.buffer else np.zeros(1)
            s_hi = np.concatenate([b[2] for b in self.buffer]) if self.buffer else np.zeros(1)
            raw_lo, raw_hi = self.lo_m.predict(g[self.cols]), self.hi_m.predict(g[self.cols])
            lo = np.clip(raw_lo - _q(s_lo, 1 - self.a_lo), 0, None)
            hi = raw_hi + _q(s_hi, 1 - self.a_hi)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            idx = [pos[i] for i in g.index]
            lo_out[idx], hi_out[idx] = lo, hi
            y = g["y"].to_numpy(float)
            ok = np.isfinite(y)
            trace.append({"date": day, "a_lo": self.a_lo, "a_hi": self.a_hi, "below": float(np.mean(y[ok] < lo[ok])) if ok.any() else np.nan,
                          "above": float(np.mean(y[ok] > hi[ok])) if ok.any() else np.nan})
            if ok.any():  # update only with verified outcomes
                target = self.alpha / 2
                self.a_lo = float(np.clip(self.a_lo + self.gamma * (target - np.mean(y[ok] < lo[ok])), A_MIN, A_MAX))
                self.a_hi = float(np.clip(self.a_hi + self.gamma * (target - np.mean(y[ok] > hi[ok])), A_MIN, A_MAX))
                self.buffer.append((day, raw_lo[ok] - y[ok], y[ok] - raw_hi[ok]))
                self._trim()
        return lo_out, hi_out, pd.DataFrame(trace)

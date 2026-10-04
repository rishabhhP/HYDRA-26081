"""Correctly labelled conformal intervals.

v2 used the 90th percentile of absolute calibration residuals and called it an 80% interval;
on the replay it covered ~90.8%. v3 uses split-conformal quantile regression (CQR, Romano et
al. 2019) at alpha = 0.20, i.e. a central 80% interval:

    score_i = max(lower_i - y_i, y_i - upper_i)
    q       = the ceil((n + 1) * (1 - alpha)) / n empirical quantile of the scores
    interval = [lower - q, upper + q]

Calibration is stratified (state x season x lead x regime) with a hierarchical fallback to
coarser strata when a stratum has too few calibration samples, and every applied row records
which stratum level supplied its correction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def conformal_rank(n: int, alpha: float) -> int:
    """1-based order statistic k = ceil((n + 1)(1 - alpha)), capped at n."""
    return int(min(n, np.ceil((n + 1) * (1 - alpha))))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    return float(scores[conformal_rank(len(scores), alpha) - 1])


def symmetric_margin(residuals: np.ndarray, alpha: float = C.ALPHA) -> float:
    """Half-width of a central (1 - alpha) interval from absolute residuals (point-forecast case)."""
    return conformal_quantile(np.abs(residuals), alpha)


class StratifiedConformal:
    def __init__(self, levels=C.STATE_STRATA, min_n: int = C.MIN_STATE_STRATUM, alpha: float = C.ALPHA,
                 nonneg: bool = True):
        self.levels = [tuple(level) for level in levels]
        self.min_n, self.alpha, self.nonneg = min_n, alpha, nonneg
        self.tables: dict[str, dict[str, dict]] = {}

    def fit(self, lower, upper, y, strata: pd.DataFrame) -> "StratifiedConformal":
        scores = np.maximum(np.asarray(lower) - np.asarray(y), np.asarray(y) - np.asarray(upper))
        frame = strata.reset_index(drop=True).assign(_score=scores)
        frame = frame[np.isfinite(frame["_score"])]
        for level in self.levels:
            name = "+".join(level) or "global"
            table = {}
            groups = frame.groupby(list(level)) if level else [((), frame)]
            for key, group in groups:
                if len(group) < (self.min_n if level else 1):
                    continue
                key = key if isinstance(key, tuple) else (key,)
                table["|".join(map(str, key)) if level else "*"] = {
                    "q": round(conformal_quantile(group["_score"].to_numpy(), self.alpha), 5), "n": int(len(group))}
            self.tables[name] = table
        return self

    def apply(self, lower, upper, strata: pd.DataFrame):
        lower, upper = np.asarray(lower, float).copy(), np.asarray(upper, float).copy()
        adj = np.full(len(lower), np.nan)
        used = np.full(len(lower), "", dtype=object)
        frame = strata.reset_index(drop=True)
        for level in self.levels:
            name = "+".join(level) or "global"
            table = self.tables.get(name, {})
            todo = np.isnan(adj)
            if not todo.any():
                break
            keys = (frame.loc[todo, list(level)].astype(str).agg("|".join, axis=1) if level
                    else pd.Series("*", index=frame.index[todo]))
            q = keys.map(lambda k: table.get(k, {}).get("q", np.nan)).to_numpy(float)
            hit = np.isfinite(q)
            positions = np.flatnonzero(todo)[hit]
            adj[positions] = q[hit]
            used[positions] = name
        adj = np.nan_to_num(adj, nan=0.0)
        lower, upper = lower - adj, upper + adj
        crossed = lower > upper
        mid = (lower + upper) / 2
        lower[crossed], upper[crossed] = mid[crossed], mid[crossed]
        return (np.clip(lower, 0, None) if self.nonneg else lower), upper, used

    def summary(self) -> dict:
        return {"alpha": self.alpha, "nominal_coverage": 1 - self.alpha, "method": "stratified split-conformal CQR",
                "levels": {name: len(table) for name, table in self.tables.items()}, "min_stratum": self.min_n}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"levels": self.levels, "min_n": self.min_n, "alpha": self.alpha,
                                    "nonneg": self.nonneg, "tables": self.tables}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "StratifiedConformal":
        state = json.loads(path.read_text())
        obj = cls([tuple(level) for level in state["levels"]], state["min_n"], state["alpha"], state.get("nonneg", True))
        obj.tables = state["tables"]
        return obj

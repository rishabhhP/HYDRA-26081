"""(c) Quantile regression on top of the blended forecast + (e) disagreement signal.

LightGBM quantile objective, one booster per level in ``config.QUANTILES``. Inputs are the
model features plus the gating blend (``blend_pred``) and the expert spread (``expert_spread``,
the model-disagreement signal, already in MODEL_FEATURES). Crossing quantiles are repaired
by sorting. Intervals are then conformalised per season on the validation split (CQR,
Romano et al. 2019) so nominal coverage holds out of sample; raw and conformalised
coverage are both reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config as C
from ..features import CAT_FEATURES, MODEL_FEATURES

QFEATURES = MODEL_FEATURES + ["blend_pred"]
INTERVALS = {"80": (0.1, 0.9), "90": (0.05, 0.95)}


class QuantileModel:
    def __init__(self, levels=C.QUANTILES):
        self.levels = list(levels)
        self.models: dict[float, lgb.Booster] = {}
        self.cqr: dict[str, dict[int, float]] = {}  # interval -> season_id -> additive widening
        self.nonneg = False

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, nonneg: bool = False) -> "QuantileModel":
        self.nonneg = nonneg
        if len(train) > C.QUANTILE_MAX_TRAIN_ROWS:
            train = train.sample(n=C.QUANTILE_MAX_TRAIN_ROWS, random_state=C.SEED)
        cats = [c for c in CAT_FEATURES if c in QFEATURES]
        for q in self.levels:
            params = dict(C.LGBM_PARAMS, objective="quantile", alpha=q, learning_rate=0.08)
            dtr = lgb.Dataset(train[QFEATURES], train["target"], categorical_feature=cats)
            dva = lgb.Dataset(val[QFEATURES], val["target"], categorical_feature=cats, reference=dtr)
            self.models[q] = lgb.train(params, dtr, C.QUANTILE_ROUNDS, valid_sets=[dva],
                                       callbacks=[lgb.early_stopping(50, verbose=False)])
        self._fit_cqr(val)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        Q = np.column_stack([self.models[q].predict(df[QFEATURES]) for q in self.levels])
        Q = np.sort(Q, 1)
        if self.nonneg:
            Q = np.clip(Q, 0, None)
        return Q

    def _fit_cqr(self, val: pd.DataFrame) -> None:
        Q = self.predict_raw(val)
        y = val["target"].to_numpy()
        for name, (lo_q, hi_q) in INTERVALS.items():
            lo, hi = Q[:, self.levels.index(lo_q)], Q[:, self.levels.index(hi_q)]
            score = np.maximum(lo - y, y - hi)
            alpha = 1 - (hi_q - lo_q)
            self.cqr[name] = {}
            for s in range(len(C.SEASONS)):
                m = (val.season_id == s).to_numpy()
                if m.sum() < 100:
                    continue
                n = m.sum()
                k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
                self.cqr[name][s] = float(np.quantile(score[m], k))

    def interval(self, df: pd.DataFrame, name: str = "80", conformal: bool = True, Q: np.ndarray | None = None):
        Q = self.predict_raw(df) if Q is None else Q
        lo_q, hi_q = INTERVALS[name]
        lo, hi = Q[:, self.levels.index(lo_q)].copy(), Q[:, self.levels.index(hi_q)].copy()
        if conformal:
            adj = df["season_id"].map(self.cqr[name]).fillna(0).to_numpy()
            lo, hi = lo - adj, hi + adj
            if self.nonneg:
                lo = np.clip(lo, 0, None)
        return lo, hi

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for q, m in self.models.items():
            m.save_model(str(d / f"quantile_q{int(round(q * 100)):02d}.txt"))
        (d / "quantile_meta.json").write_text(json.dumps(
            {"levels": self.levels, "features": QFEATURES, "cqr_by_season_id": self.cqr,
             "intervals": INTERVALS, "nonneg": self.nonneg}, indent=1))

    @classmethod
    def load(cls, d: Path) -> "QuantileModel":
        meta = json.loads((d / "quantile_meta.json").read_text())
        obj = cls(meta["levels"])
        obj.nonneg = meta["nonneg"]
        obj.cqr = {k: {int(s): v for s, v in dd.items()} for k, dd in meta["cqr_by_season_id"].items()}
        obj.models = {q: lgb.Booster(model_file=str(d / f"quantile_q{int(round(q * 100)):02d}.txt")) for q in obj.levels}
        return obj

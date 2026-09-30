"""(a) Per-season LightGBM regressor - the reference model.

Same idea as the earlier prototype (one gradient-boosted regressor per season, experts +
context as inputs), extended with lead time, historical-skill and regime features.
If a season has no training rows (time-forward split: post-monsoon), a global all-season
model is used for it and this is recorded in ``fallback_seasons``.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config as C
from ..features import CAT_FEATURES, MODEL_FEATURES


def _subsample(df: pd.DataFrame, n: int, seed: int = C.SEED) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if len(df) > n else df


class SeasonalLGBM:
    def __init__(self, features: list[str] = MODEL_FEATURES, params: dict | None = None):
        self.features = list(features)
        self.params = dict(C.LGBM_PARAMS, **(params or {}))
        self.models: dict[str, lgb.Booster] = {}
        self.fallback_seasons: list[str] = []

    def _fit_one(self, tr: pd.DataFrame, va: pd.DataFrame) -> lgb.Booster:
        cats = [c for c in CAT_FEATURES if c in self.features]
        dtr = lgb.Dataset(tr[self.features], tr["target"], categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(va[self.features], va["target"], categorical_feature=cats, reference=dtr)
        return lgb.train(self.params, dtr, C.LGBM_ROUNDS, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOP, verbose=False)])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "SeasonalLGBM":
        glob = None
        for s_id, s in enumerate(C.SEASONS):
            tr, va = train[train.season_id == s_id], val[val.season_id == s_id]
            if len(tr) < 1000 or len(va) < 100:
                self.fallback_seasons.append(s)
                continue
            self.models[s] = self._fit_one(_subsample(tr, C.LGBM_MAX_TRAIN_ROWS // 2), va)
        if self.fallback_seasons:
            glob = self._fit_one(_subsample(train, C.LGBM_MAX_TRAIN_ROWS), val)
            for s in self.fallback_seasons:
                self.models[s] = glob
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for s_id, s in enumerate(C.SEASONS):
            m = (df.season_id == s_id).to_numpy()
            if m.any():
                out[m] = self.models[s].predict(df.loc[m, self.features])
        return out

    def importance(self) -> pd.DataFrame:
        rows = []
        for s, m in self.models.items():
            gain = m.feature_importance("gain")
            tot = gain.sum() or 1.0
            rows += [(s, f, g / tot) for f, g in zip(m.feature_name(), gain)]
        return pd.DataFrame(rows, columns=["season", "feature", "gain_share"])

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for s, m in self.models.items():
            m.save_model(str(d / f"lgbm_{s}.txt"))
        (d / "lgbm_meta.json").write_text(json.dumps(
            {"features": self.features, "fallback_seasons": self.fallback_seasons, "seasons": C.SEASONS}, indent=1))

    @classmethod
    def load(cls, d: Path) -> "SeasonalLGBM":
        meta = json.loads((d / "lgbm_meta.json").read_text())
        obj = cls(meta["features"])
        obj.fallback_seasons = meta["fallback_seasons"]
        obj.models = {s: lgb.Booster(model_file=str(d / f"lgbm_{s}.txt")) for s in meta["seasons"]}
        return obj

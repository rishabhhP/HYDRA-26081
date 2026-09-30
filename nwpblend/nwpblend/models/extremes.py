"""(d) Extreme-event classifiers with calibrated probabilities.

One LightGBM binary classifier per flag (all seasons, season as a feature), followed by
isotonic calibration fitted on the validation split. Calibration maps are exported as
plain (x, y) knot arrays in JSON, so the backend can apply them with ``np.interp``.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .. import config as C
from ..features import CAT_FEATURES, MODEL_FEATURES

CLF_FEATURES = MODEL_FEATURES + ["blend_pred"]


class ExtremeClassifier:
    def __init__(self, flag: str):
        self.flag = flag
        self.model: lgb.Booster | None = None
        self.iso_x: np.ndarray | None = None
        self.iso_y: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "ExtremeClassifier":
        if len(train) > C.CLF_MAX_TRAIN_ROWS:
            # keep every positive, subsample negatives (then calibration on untouched val fixes the prior)
            pos = train[train[self.flag] == 1]
            neg = train[train[self.flag] == 0].sample(n=C.CLF_MAX_TRAIN_ROWS - len(pos), random_state=C.SEED)
            train = pd.concat([pos, neg])
        params = dict(C.LGBM_PARAMS, objective="binary", learning_rate=0.05, min_data_in_leaf=100)
        cats = [c for c in CAT_FEATURES if c in CLF_FEATURES]
        dtr = lgb.Dataset(train[CLF_FEATURES], train[self.flag], categorical_feature=cats)
        dva = lgb.Dataset(val[CLF_FEATURES], val[self.flag], categorical_feature=cats, reference=dtr)
        self.model = lgb.train(params, dtr, C.CLF_ROUNDS, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
        raw = self.model.predict(val[CLF_FEATURES])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, val[self.flag])
        self.iso_x, self.iso_y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[CLF_FEATURES])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.interp(self.predict_raw(df), self.iso_x, self.iso_y)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(d / f"clf_{self.flag}.txt"))
        (d / f"clf_{self.flag}_calibration.json").write_text(json.dumps(
            {"features": CLF_FEATURES, "isotonic_x": self.iso_x.tolist(), "isotonic_y": self.iso_y.tolist()}))

    @classmethod
    def load(cls, d: Path, flag: str) -> "ExtremeClassifier":
        obj = cls(flag)
        obj.model = lgb.Booster(model_file=str(d / f"clf_{flag}.txt"))
        cal = json.loads((d / f"clf_{flag}_calibration.json").read_text())
        obj.iso_x, obj.iso_y = np.array(cal["isotonic_x"]), np.array(cal["isotonic_y"])
        return obj

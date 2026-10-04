"""HYDRA rainfall v3 experts.

Baselines (closed form, kept from v2):  climatology, persistence, recent3, anom_persistence.
Trained (scikit-learn HistGradientBoosting, NaN-aware, CPU):
    gbm_era5        Poisson-loss boosted trees on every ERA5 atmospheric + rainfall feature
    hurdle          P(rain >= 1 mm) classifier  x  gamma-loss wet-day amount regressor
    spatial         neighbourhood-only model (surrounding rain, CAPE, moisture within +-0.5/+-1 deg)
    monsoon         specialist fitted on monsoon-season rows only
    upper_q         85th-percentile quantile regressor, the extreme-aware expert

For the gate, trained experts are supplied as out-of-fold predictions (contiguous day blocks
with an embargo), so the gate learns how much to trust each expert on data that expert
never saw.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as C
from .features import CAT_COLS

BASELINES = ("climatology", "persistence", "recent3", "anom_persistence")
TRAINED = ("gbm_era5", "hurdle", "spatial", "monsoon", "upper_q")
EXPERTS = BASELINES + TRAINED
AUX = ("hurdle_p_rain", "hurdle_wet_amount")      # extra expert outputs passed to the gate
SPATIAL_PREFIXES = ("nb", "tp", "wet7", "clim_", "latitude", "longitude")
MIN_MONSOON_ROWS = 5000


def baseline_values(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["climatology"] = rows["clim_target"]
    out["persistence"] = rows["tp0"]
    out["recent3"] = rows["tp3"]
    out["anom_persistence"] = np.clip(rows["clim_target"] + rows["tp0"] - rows["clim_issue"], 0, None)
    return out.astype(np.float32)


def _matrix(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = rows[cols].astype(np.float32).copy()
    if "state_id" in X:
        X.loc[X["state_id"] < 0, "state_id"] = np.nan
    return X


class ExpertBank:
    def __init__(self, numeric: list[str]):
        self.numeric = list(numeric)
        self.cols = self.numeric + list(CAT_COLS)
        self.spatial_cols = [c for c in self.numeric if c.startswith(SPATIAL_PREFIXES)] + ["season_id", "lead_days"]
        self.models: dict[str, object] = {}
        self.meta: dict = {}

    # -------------------------------------------------- fitting
    def _gbm(self, cols, **kw):
        cats = [c in ("season_id", "regime_id", "state_id") for c in cols]
        params = dict(C.GBM_PARAMS, categorical_features=cats)
        params.update(kw)
        return params

    def _fit_one(self, name: str, rows: pd.DataFrame):
        y = rows["target"].to_numpy(np.float64)
        w = rows["_w"].to_numpy(np.float64)
        if name == "gbm_era5":
            return HistGradientBoostingRegressor(**self._gbm(self.cols, loss="poisson")).fit(
                _matrix(rows, self.cols), y, sample_weight=w)
        if name == "spatial":
            return HistGradientBoostingRegressor(**self._gbm(self.spatial_cols, loss="poisson")).fit(
                _matrix(rows, self.spatial_cols), y, sample_weight=w)
        if name == "upper_q":
            return HistGradientBoostingRegressor(**self._gbm(self.cols, loss="quantile", quantile=C.UPPER_QUANTILE)).fit(
                _matrix(rows, self.cols), y, sample_weight=w)
        if name == "monsoon":
            subset = rows[rows["season_id"] == C.SEASONS.index("monsoon")]
            if len(subset) < MIN_MONSOON_ROWS:
                subset = rows[rows["regime_id"] >= C.REGIMES.index("wet_spell")]
            self.meta["monsoon_rows"] = int(len(subset))
            if len(subset) < MIN_MONSOON_ROWS:
                subset = rows
                self.meta["monsoon_fallback"] = "insufficient monsoon rows; fitted on all rows"
            return HistGradientBoostingRegressor(**self._gbm(self.cols, loss="poisson")).fit(
                _matrix(subset, self.cols), subset["target"].to_numpy(np.float64),
                sample_weight=subset["_w"].to_numpy(np.float64))
        if name == "hurdle":
            wet = y >= C.WET_MM
            clf = HistGradientBoostingClassifier(**self._gbm(self.cols, loss="log_loss")).fit(
                _matrix(rows, self.cols), wet.astype(int), sample_weight=w)
            amount = HistGradientBoostingRegressor(**self._gbm(self.cols, loss="gamma")).fit(
                _matrix(rows[wet], self.cols), y[wet], sample_weight=w[wet])
            return clf, amount
        raise KeyError(name)

    def fit(self, rows: pd.DataFrame) -> "ExpertBank":
        rows = _cap(rows, C.MAX_EXPERT_ROWS)
        for name in TRAINED:
            print(f"    expert {name}: fitting on {len(rows):,} rows", flush=True)
            self.models[name] = self._fit_one(name, rows)
        self.meta["train_rows"] = int(len(rows))
        return self

    # -------------------------------------------------- prediction
    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = baseline_values(rows)
        X = _matrix(rows, self.cols)
        out["gbm_era5"] = self.models["gbm_era5"].predict(X)
        out["spatial"] = self.models["spatial"].predict(_matrix(rows, self.spatial_cols))
        out["monsoon"] = self.models["monsoon"].predict(X)
        out["upper_q"] = np.clip(self.models["upper_q"].predict(X), 0, None)
        clf, amount = self.models["hurdle"]
        p = clf.predict_proba(X)[:, 1]
        wet_amount = amount.predict(X)
        out["hurdle"] = p * wet_amount
        out["hurdle_p_rain"] = p
        out["hurdle_wet_amount"] = wet_amount
        return out.astype(np.float32)

    def oof_predict(self, rows: pd.DataFrame, folds: int | None = None) -> pd.DataFrame:
        """Out-of-fold expert predictions over contiguous issue-day blocks with an embargo."""
        folds = C.EXPERT_OOF_FOLDS if folds is None else folds
        days = np.sort(rows["issue_idx"].unique())
        blocks = np.array_split(days, folds)
        out = pd.DataFrame(index=rows.index, columns=list(EXPERTS) + list(AUX), dtype=np.float32)
        for k, block in enumerate(blocks):
            lo, hi = block.min(), block.max()
            held = rows["issue_idx"].between(lo, hi)
            far = (rows["issue_idx"] < lo - C.EMBARGO_DAYS - max(C.LEADS)) | (rows["issue_idx"] > hi + C.EMBARGO_DAYS)
            print(f"    out-of-fold block {k + 1}/{folds}: {int(held.sum()):,} rows held out", flush=True)
            bank = ExpertBank(self.numeric).fit(rows[far])
            out.loc[held] = bank.predict(rows[held]).to_numpy()
        return out.astype(np.float32)

    # -------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "experts.pkl").open("wb") as stream:
            pickle.dump({"numeric": self.numeric, "models": self.models, "meta": self.meta}, stream)
        (directory / "experts.json").write_text(json.dumps({"experts": EXPERTS, "aux": AUX, "numeric": self.numeric,
                                                            "spatial_cols": self.spatial_cols, "meta": self.meta}, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "ExpertBank":
        with (directory / "experts.pkl").open("rb") as stream:
            state = pickle.load(stream)
        bank = cls(state["numeric"])
        bank.models, bank.meta = state["models"], state["meta"]
        return bank


def _cap(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Row cap that keeps every heavy-rain row and reweights the subsampled rest (unbiased totals)."""
    rows = rows.assign(_w=np.float32(1.0))
    if len(rows) <= limit:
        return rows
    heavy = rows["target"] >= C.HEAVY_THRESHOLDS_MM[0]
    keep_heavy = rows[heavy]
    budget = max(1, limit - len(keep_heavy))
    rest = rows[~heavy]
    sampled = rest.sample(n=min(budget, len(rest)), random_state=C.SEED)
    sampled = sampled.assign(_w=np.float32(len(rest) / max(len(sampled), 1)))
    return pd.concat([keep_heavy, sampled]).sort_index()

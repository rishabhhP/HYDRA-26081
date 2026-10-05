"""Post-processing models for HYDRA v3 state-day forecasts.

AmountHead     priority 1  extreme-aware amount: log1p head with extreme weights + smearing, Poisson head, or a
                           heavy/non-heavy mixture (separate log-transformed heavy-amount head x calibrated P(>=20 mm))
ProbabilityHeads priority 1/6 calibrated P(state mean >= 10, 20, 64.5 mm) and P(any grid cell >= 64.5 mm) with alert tiers
AsymmetricInterval priority 2 separate lower/upper quantile models, each tail conformalised at alpha/2 by stratum
SkillGate      priority 3  online expert re-weighting by recent verified skill (Hedge), concentration cap, persistence-relative
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SEED = 7
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


def _cats(cols):
    return [c in ("state_id", "season") for c in cols]


def extreme_weight(y, alpha=3.0, ref=20.0, cap=4.0):
    return 1 + alpha * np.minimum(np.asarray(y) / ref, cap)


def split_time(frame: pd.DataFrame, frac: float = 0.25):
    days = np.sort(frame["date"].unique())
    cut = days[int(len(days) * (1 - frac))]
    return frame[frame["date"] < cut], frame[frame["date"] >= cut]


# ------------------------------------------------------------------ amount
class AmountHead:
    VARIANTS = ("log_extreme", "poisson", "mixture")

    def __init__(self, cols, variant="log_extreme", extreme_alpha: float = 3.0):
        self.cols, self.variant, self.extreme_alpha = cols, variant, extreme_alpha

    def fit(self, tr: pd.DataFrame):
        X, y = tr[self.cols], tr["y"].to_numpy()
        w = extreme_weight(y, alpha=self.extreme_alpha)
        if self.variant == "log_extreme":
            self.m = HistGradientBoostingRegressor(**GBM, categorical_features=_cats(self.cols)).fit(X, np.log1p(y), sample_weight=w)
            res = np.log1p(y) - self.m.predict(X)
            self.smear = float(np.average(np.exp(res), weights=w))
        elif self.variant == "poisson":
            self.m = HistGradientBoostingRegressor(**GBM, loss="poisson", categorical_features=_cats(self.cols)).fit(X, y, sample_weight=w)
        else:
            heavy = y >= 20
            self.p = ProbabilityHeads(self.cols, targets={"state_20": ("y", 20.0)}).fit(tr)
            self.lo = HistGradientBoostingRegressor(**GBM, loss="poisson", categorical_features=_cats(self.cols)).fit(X[~heavy], y[~heavy])
            hv = tr[heavy]
            if heavy.sum() >= 30:
                self.hi = HistGradientBoostingRegressor(**{**GBM, "min_samples_leaf": 10}, categorical_features=_cats(self.cols)).fit(hv[self.cols], np.log1p(hv["y"]))
                self.hi_smear = float(np.mean(np.exp(np.log1p(hv["y"]) - self.hi.predict(hv[self.cols]))))
            else:
                self.hi, self.hi_const = None, float(hv["y"].mean()) if len(hv) else 30.0
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        X = te[self.cols]
        if self.variant == "log_extreme":
            return np.clip(np.exp(self.m.predict(X)) * self.smear - 1, 0, None)
        if self.variant == "poisson":
            return np.clip(self.m.predict(X), 0, None)
        p20 = self.p.predict(te)["state_20"]
        low = np.clip(self.lo.predict(X), 0, 20)
        high = np.maximum(np.exp(self.hi.predict(X)) * self.hi_smear - 1, 20) if self.hi is not None else np.full(len(te), self.hi_const)
        self.last_heavy_amount = high
        return (1 - p20) * low + p20 * high


# ------------------------------------------------------------------ probabilities
class ProbabilityHeads:
    DEFAULT = {"state_10": ("y", 10.0), "state_20": ("y", 20.0), "state_64.5": ("y", 64.5), "local_64.5": ("y_local_max", 64.5)}
    SMALL = ["hydra", "hi80", "local_peak", "obs_max7", "localmax_max7", "pm_64.5", "pa_20"]

    def __init__(self, cols, targets=None, calib_frac=0.25):
        self.cols, self.targets, self.calib_frac = cols, targets or self.DEFAULT, calib_frac
        self.models, self.kind = {}, {}

    def fit(self, tr: pd.DataFrame):
        fit_part, cal_part = split_time(tr, self.calib_frac)
        for name, (col, thr) in self.targets.items():
            yb = (tr[col] >= thr).to_numpy()
            pos = int(yb.sum())
            if pos < 30:  # rare event: small regularised logistic model, no separate calibration split
                cols = [c for c in self.SMALL if c in tr]
                X = np.log1p(tr[cols].fillna(0).clip(lower=0))
                if pos == 0:
                    self.models[name], self.kind[name] = float(0.0), "none"
                    continue
                m = LogisticRegression(C=0.3, max_iter=2000).fit(X, yb)
                self.models[name], self.kind[name] = (m, cols), f"logistic (only {pos} training events)"
                continue
            yf = (fit_part[col] >= thr).to_numpy()
            yc = (cal_part[col] >= thr).to_numpy()
            clf = HistGradientBoostingClassifier(**GBM, categorical_features=_cats(self.cols)).fit(fit_part[self.cols], yf)
            raw = clf.predict_proba(cal_part[self.cols])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, yc) if 0 < yc.sum() < len(yc) else None
            # refit the classifier on the whole window; keep the calibration map learnt out-of-sample
            clf_full = HistGradientBoostingClassifier(**GBM, categorical_features=_cats(self.cols)).fit(tr[self.cols], yb)
            self.models[name], self.kind[name] = (clf_full, iso), f"gradient boosting + isotonic ({pos} training events)"
        return self

    def predict(self, te: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for name, model in self.models.items():
            if self.kind[name] == "none":
                out[name] = np.zeros(len(te))
            elif self.kind[name].startswith("logistic"):
                m, cols = model
                out[name] = m.predict_proba(np.log1p(te[cols].fillna(0).clip(lower=0)))[:, 1]
            else:
                clf, iso = model
                raw = clf.predict_proba(te[self.cols])[:, 1]
                out[name] = iso.predict(raw) if iso is not None else raw
        return out


# ------------------------------------------------------------------ asymmetric interval
class AsymmetricInterval:
    """Lower and upper bounds modelled and conformalised separately, so each tail gets alpha/2 = 10%."""

    def __init__(self, cols, alpha=0.2, calib_frac=0.25, min_n=40):
        self.cols, self.alpha, self.calib_frac, self.min_n = cols, alpha, calib_frac, min_n

    @staticmethod
    def stratum(frame):
        wet = np.where(frame["obs_lag1"].fillna(0) >= 5, "wet", "dry")
        return frame["season"].astype(str).to_numpy() + "|" + wet

    @staticmethod
    def _q(scores, level):
        s = np.sort(scores[np.isfinite(scores)])
        k = int(min(len(s), np.ceil((len(s) + 1) * level)))
        return float(s[k - 1])

    def fit(self, tr: pd.DataFrame):
        fit_part, cal = split_time(tr, self.calib_frac)
        a2 = self.alpha / 2
        self.lo_m = HistGradientBoostingRegressor(**GBM, loss="quantile", quantile=a2, categorical_features=_cats(self.cols)).fit(fit_part[self.cols], fit_part["y"])
        self.hi_m = HistGradientBoostingRegressor(**GBM, loss="quantile", quantile=1 - a2, categorical_features=_cats(self.cols)).fit(fit_part[self.cols], fit_part["y"])
        lo, hi, y = self.lo_m.predict(cal[self.cols]), self.hi_m.predict(cal[self.cols]), cal["y"].to_numpy()
        s_lo, s_hi = lo - y, y - hi
        strata = self.stratum(cal)
        level = 1 - a2
        self.adj = {"*": (self._q(s_lo, level), self._q(s_hi, level))}
        for key in np.unique(strata):
            m = strata == key
            if m.sum() >= self.min_n:
                self.adj[key] = (self._q(s_lo[m], level), self._q(s_hi[m], level))
        return self

    def predict(self, te: pd.DataFrame):
        lo, hi = self.lo_m.predict(te[self.cols]), self.hi_m.predict(te[self.cols])
        keys = self.stratum(te)
        a_lo = np.array([self.adj.get(k, self.adj["*"])[0] for k in keys])
        a_hi = np.array([self.adj.get(k, self.adj["*"])[1] for k in keys])
        lo, hi = np.clip(lo - a_lo, 0, None), hi + a_hi
        return np.minimum(lo, hi), np.maximum(lo, hi)


# ------------------------------------------------------------------ skill-aware gate (online)
def capped_weights(raw: np.ndarray, cap: float, floor: float) -> np.ndarray:
    """Exact projection onto {w >= floor, w <= cap, sum w = 1}, keeping the ordering of `raw`."""
    k = len(raw)
    if k * floor >= 1 or cap * k < 1:
        return np.full(k, 1.0 / k)
    inner_cap = (cap - floor) / (1 - k * floor)
    w = np.maximum(raw, 0).astype(float)
    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    fixed = np.zeros(k, bool)
    for _ in range(k):
        free = ~fixed
        remaining = 1 - inner_cap * fixed.sum()
        w[free] = w[free] / w[free].sum() * remaining if w[free].sum() > 0 else remaining / free.sum()
        over = free & (w > inner_cap)
        if not over.any():
            break
        fixed |= over
        w[fixed] = inner_cap
    return floor + (1 - k * floor) * w


def skill_gate(df: pd.DataFrame, experts: list[str], eta: float = 0.15, decay: float = 0.97, cap: float = 0.5,
               floor: float = 0.01, include_hydra: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per state, weight experts by exp(-eta * decayed absolute error over verified past days).

    The error of valid day t-1 is known on issue day t-1 (its observation is the persistence input), so the weight
    for day t uses errors up to t-1 only. Experts that keep losing to persistence lose weight; no expert can exceed
    `cap`, which stops a single expert taking over after a lucky streak.
    """
    names = experts + (["hydra"] if include_hydra else [])
    pred = np.full(len(df), np.nan)
    W = np.full((len(df), len(names)), np.nan)
    for _, idx in df.groupby("state", sort=False).indices.items():
        sub = df.iloc[idx]
        E = np.column_stack([sub[f"e_{n}"] if n != "hydra" else sub["hydra"] for n in names]).astype(float)
        y = sub["y"].to_numpy(float)
        loss = np.zeros(len(names))
        for t in range(len(sub)):
            w = capped_weights(np.exp(-eta * (loss - loss.min())), cap, floor)
            W[idx[t]] = w
            pred[idx[t]] = float(np.nansum(w * np.nan_to_num(E[t])))
            if np.isfinite(y[t]):
                loss = decay * loss + np.abs(np.nan_to_num(E[t]) - y[t])
    return np.clip(pred, 0, None), W


# ------------------------------------------------------------------ calibrated native HYDRA heads
class NativeCalibration:
    """Isotonic map from HYDRA's own heavy-rain heads to calibrated event probabilities, plus a small logistic stack.

    HYDRA's p_heavy_area is an expected wet-area fraction and p_heavy_max_cell a max over cells: both rank events well
    but are not probabilities of the state-day event. These maps turn them into calibrated probabilities.
    """
    MAP = {"state_10": ("pa_20", "y", 10.0), "state_20": ("pa_20", "y", 20.0), "state_64.5": ("pa_64.5", "y", 64.5),
           "local_64.5": ("pm_64.5", "y_local_max", 64.5)}
    STACK = ["hydra", "hi80", "obs_lag1", "obs_max7", "local_peak", "localmax_lag1"]

    def fit(self, tr: pd.DataFrame):
        self.iso, self.stack = {}, {}
        for name, (src, col, thr) in self.MAP.items():
            yb = (tr[col] >= thr).to_numpy()
            if yb.sum() < 5 or yb.all():
                self.iso[name] = self.stack[name] = None
                continue
            self.iso[name] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(tr[src].fillna(0), yb)
            X = self._x(tr, src)
            self.stack[name] = LogisticRegression(C=0.5 if yb.sum() >= 30 else 0.1, max_iter=3000).fit(X, yb)
        return self

    def _x(self, frame, src):
        p = frame[src].fillna(0).clip(1e-4, 1 - 1e-4)
        cols = [np.log(p / (1 - p))] + [np.log1p(frame[c].fillna(0).clip(lower=0)) for c in self.STACK]
        return np.column_stack(cols)

    def predict(self, te: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        for name, (src, _, _) in self.MAP.items():
            if self.iso[name] is None:
                out[name] = {"iso": np.zeros(len(te)), "stack": np.zeros(len(te))}
                continue
            out[name] = {"iso": self.iso[name].predict(te[src].fillna(0)), "stack": self.stack[name].predict_proba(self._x(te, src))[:, 1]}
        return out

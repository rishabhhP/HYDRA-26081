"""Scoring functions: deterministic, probabilistic (quantile) and binary-event metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config as C


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def improvement_pct(model_err: float, base_err: float) -> float:
    """Positive = model better than baseline."""
    return float(100.0 * (base_err - model_err) / base_err) if base_err > 0 else np.nan


def day_block_bootstrap_rmse_gain(y, p_model, p_base, day, n_boot: int = 500, seed: int = C.SEED):
    """95% CI of % RMSE improvement, resampling whole test DAYS (grid cells on one day are
    strongly correlated, so resampling rows would give absurdly narrow intervals)."""
    df = pd.DataFrame({"d": day, "m": (np.asarray(y) - p_model) ** 2, "b": (np.asarray(y) - p_base) ** 2})
    g = df.groupby("d")[["m", "b"]].agg(["sum", "count"])
    sm, sb, n = g[("m", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("m", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(sm)
    idx = rng.integers(0, k, size=(n_boot, k))
    rm = np.sqrt(sm[idx].sum(1) / n[idx].sum(1))
    rb = np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    gains = 100 * (rb - rm) / rb
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


# ------------------------------------------------------------ quantiles
def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def crps_from_quantiles(y, qmat: np.ndarray, levels) -> float:
    """CRPS approximated as 2 x mean pinball loss over the quantile levels (exact in the limit
    of a dense, evenly spaced grid; with 7 levels it is a consistent approximation, used
    identically for model and baselines so skill scores are comparable)."""
    return float(2.0 * np.mean([pinball(y, qmat[:, j], q) for j, q in enumerate(levels)]))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


# ------------------------------------------------------------ binary events
def binary_metrics(y, p, day=None, thresholds=(0.2, 0.5)) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    npos = int(y.sum())
    pos_days = int(pd.Series(day)[y == 1].nunique()) if day is not None else None
    out = {"n": len(y), "n_pos": npos, "pos_days": pos_days, "base_rate": float(y.mean()) if len(y) else np.nan}
    trusted = npos >= C.MIN_POSITIVES_TRUST and (pos_days is None or pos_days >= C.MIN_POSITIVE_DAYS_TRUST)
    out["trusted"] = bool(trusted)
    if 0 < npos < len(y):
        out["auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = out["avg_precision"] = np.nan
    out["brier"] = float(np.mean((p - y) ** 2))
    ref = out["base_rate"] * (1 - out["base_rate"])
    out["brier_skill_vs_climo"] = float(1 - out["brier"] / ref) if ref > 0 else np.nan
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        out[f"precision@{t}"] = tp / pred.sum() if pred.sum() else np.nan
        out[f"recall@{t}"] = tp / npos if npos else np.nan
    return out


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), obs_freq=("y", "mean"), n=("y", "size")).reset_index()
    g["bin_range"] = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in g["bin"]]
    return g

"""Rainfall verification metrics used by training, rolling validation and the UI payloads.

Pure numpy so the backend and report scripts can use it without torch or LightGBM.
Every function ignores rows where the observation or the prediction is not finite.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

WET_MM = 1.0
INTENSITY_BINS = ((0.0, 1.0, "dry_lt1"), (1.0, 10.0, "light_1_10"), (10.0, 20.0, "moderate_10_20"),
                  (20.0, 64.5, "rather_heavy_20_64_5"), (64.5, np.inf, "heavy_ge64_5"))
MIN_EVENTS_TRUST = 10


def _finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), bool)
    for array in arrays:
        if array is not None:
            mask &= np.isfinite(np.asarray(array, float))
    return mask


def _r(value: float, digits: int = 4) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def continuous(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = _finite(obs, pred)
    if not m.any():
        return {"mae": None, "rmse": None, "bias": None, "samples": 0}
    err = pred[m] - obs[m]
    return {"mae": _r(np.mean(np.abs(err))), "rmse": _r(np.sqrt(np.mean(err ** 2))),
            "bias": _r(np.mean(err)), "samples": int(m.sum())}


def contingency(obs_event: np.ndarray, pred_event: np.ndarray) -> dict:
    """Binary event scores. ``recall`` is POD; ``precision`` is 1 - FAR."""
    o, p = np.asarray(obs_event, bool), np.asarray(pred_event, bool)
    hits, misses = int(np.sum(o & p)), int(np.sum(o & ~p))
    false_alarms, correct_neg = int(np.sum(~o & p)), int(np.sum(~o & ~p))
    n = hits + misses + false_alarms + correct_neg

    def ratio(a, b):
        return _r(a / b) if b else None

    return {
        "events": hits + misses, "forecast_events": hits + false_alarms, "samples": n,
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "accuracy": ratio(hits + correct_neg, n), "recall": ratio(hits, hits + misses),
        "precision": ratio(hits, hits + false_alarms), "csi": ratio(hits, hits + misses + false_alarms),
        "trustworthy": (hits + misses) >= MIN_EVENTS_TRUST,
    }


def brier(obs_event: np.ndarray, prob: np.ndarray) -> float | None:
    o, p = np.asarray(obs_event, float), np.asarray(prob, float)
    m = _finite(o, p)
    return _r(np.mean((p[m] - o[m]) ** 2)) if m.any() else None


def occurrence(obs: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None,
               threshold: float = WET_MM) -> dict:
    """Rain/no-rain skill. Uses the probability head (p >= 0.5) when supplied."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = _finite(obs, pred, prob)
    forecast = (np.asarray(prob, float)[m] >= 0.5) if prob is not None else (pred[m] >= threshold)
    out = contingency(obs[m] >= threshold, forecast)
    out["threshold_mm"] = threshold
    out["decision"] = "p_rain >= 0.5" if prob is not None else f"forecast >= {threshold} mm"
    if prob is not None:
        out["brier"] = brier(obs[m] >= threshold, np.asarray(prob, float)[m])
    return out


def heavy(obs: np.ndarray, pred: np.ndarray, thresholds: Iterable[float],
          probs: Mapping[str, np.ndarray] | None = None, prob_cut: float = 0.5) -> dict:
    """Heavy-rain detection per threshold, from probabilities when available, else the point forecast."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    out = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        prob = None if probs is None else probs.get(key)
        m = _finite(obs, pred, prob)
        if prob is not None:
            forecast = np.asarray(prob, float)[m] >= prob_cut
        else:
            forecast = pred[m] >= threshold
        scores = contingency(obs[m] >= threshold, forecast)
        scores["threshold_mm"] = float(threshold)
        scores["decision"] = f"p >= {prob_cut}" if prob is not None else f"forecast >= {threshold} mm"
        if prob is not None:
            scores["brier"] = brier(obs[m] >= threshold, np.asarray(prob, float)[m])
        out[key] = scores
    return out


def threshold_key(threshold: float | str) -> str:
    if isinstance(threshold, str):
        return threshold
    return f"{threshold:g}"


def peaks(obs: np.ndarray, pred: np.ndarray, dates: np.ndarray | None = None, top: int = 5) -> dict:
    """Error on the largest observed rainfall values (where underprediction matters most)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = _finite(obs, pred)
    if not m.any():
        return {"max_observed": None}
    idx = np.flatnonzero(m)
    order = idx[np.argsort(obs[idx])[::-1]]
    k = min(top, len(order))
    largest = order[0]
    top_idx = order[:k]
    out = {
        "max_observed": _r(obs[largest], 3), "predicted_at_max": _r(pred[largest], 3),
        "error_at_max": _r(pred[largest] - obs[largest], 3),
        "ratio_at_max": _r(pred[largest] / obs[largest], 3) if obs[largest] > 0 else None,
        f"top{k}_mean_error": _r(np.mean(pred[top_idx] - obs[top_idx]), 3),
        f"top{k}_mean_ratio": _r(np.sum(pred[top_idx]) / np.sum(obs[top_idx]), 3) if np.sum(obs[top_idx]) > 0 else None,
    }
    if dates is not None:
        out["date_of_max"] = str(np.asarray(dates)[largest])
    return out


def interval(obs: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> dict:
    obs, lower, upper = (np.asarray(a, float) for a in (obs, lower, upper))
    m = _finite(obs, lower, upper)
    if not m.any():
        return {"nominal": nominal, "coverage": None, "mean_width": None, "samples": 0}
    inside = (obs[m] >= lower[m]) & (obs[m] <= upper[m])
    return {"nominal": nominal, "coverage": _r(np.mean(inside)), "mean_width": _r(np.mean(upper[m] - lower[m]), 3),
            "below": _r(np.mean(obs[m] < lower[m])), "above": _r(np.mean(obs[m] > upper[m])), "samples": int(m.sum())}


def skill_vs(obs: np.ndarray, pred: np.ndarray, reference: np.ndarray) -> dict:
    """Skill score 1 - score_model / score_reference (positive = HYDRA better)."""
    model, ref = continuous(obs, pred), continuous(obs, reference)
    out = {"reference_mae": ref["mae"], "reference_rmse": ref["rmse"]}
    for name in ("mae", "rmse"):
        out[f"{name}_skill"] = (_r(1 - model[name] / ref[name]) if model[name] is not None and ref[name] else None)
    return out


def by_intensity(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    out = {}
    for low, high, name in INTENSITY_BINS:
        m = (obs >= low) & (obs < high)
        out[name] = continuous(obs[m], pred[m])
    return out


def full_report(obs, pred, *, lower=None, upper=None, nominal=0.8, p_rain=None, heavy_probs=None,
                heavy_thresholds=(20.0, 64.5), references: Mapping[str, np.ndarray] | None = None,
                dates=None) -> dict:
    """Every metric the HYDRA rainfall evaluation contract requires, in one dictionary."""
    report = {
        "continuous": continuous(obs, pred),
        "occurrence": occurrence(obs, pred, p_rain),
        "heavy": heavy(obs, pred, heavy_thresholds, heavy_probs),
        "peaks": peaks(obs, pred, dates),
        "by_intensity": by_intensity(obs, pred),
    }
    if lower is not None and upper is not None:
        report["interval"] = interval(obs, lower, upper, nominal)
    if references:
        report["skill"] = {name: skill_vs(obs, pred, ref) for name, ref in references.items()}
    return report


def weight_dynamics(weights: np.ndarray, experts: list[str], group: np.ndarray | None = None) -> dict:
    """How much the gate actually moves. ``group`` (e.g. dates) averages rows before measuring change."""
    w = np.asarray(weights, float)
    if group is not None:
        keys, inverse = np.unique(np.asarray(group), return_inverse=True)
        sums = np.zeros((len(keys), w.shape[1]))
        np.add.at(sums, inverse, w)
        w = sums / np.bincount(inverse)[:, None]
    std = w.std(0)
    day_change = np.abs(np.diff(w, axis=0)).sum(1) / 2 if len(w) > 1 else np.zeros(1)
    entropy = -(w * np.log(np.clip(w, 1e-9, 1))).sum(1) / np.log(w.shape[1])
    argmax = w.argmax(1)
    modal = np.bincount(argmax, minlength=w.shape[1]).argmax()
    return {
        "experts": experts,
        "mean_weight": [_r(v) for v in w.mean(0)],
        "std_weight": [_r(v) for v in std],
        "mean_std": _r(std.mean()),
        "mean_total_variation_step": _r(day_change.mean()),
        "mean_normalised_entropy": _r(entropy.mean()),
        "share_rows_not_modal_expert": _r(np.mean(argmax != modal)),
        "static_warning": bool(std.mean() < 0.02),
    }

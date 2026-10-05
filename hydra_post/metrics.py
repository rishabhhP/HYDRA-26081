"""The diagnostic's full metric contract, plus paired day-block bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

BINS = ((0, 1, "dry <1"), (1, 10, "1-10"), (10, 20, "10-20"), (20, 64.5, "20-64.5"), (64.5, np.inf, ">=64.5"))


def contingency(obs: np.ndarray, fc: np.ndarray) -> dict:
    hits, miss, fa = int((obs & fc).sum()), int((obs & ~fc).sum()), int((~obs & fc).sum())
    n_obs = hits + miss
    return {"events": n_obs, "forecasts": hits + fa, "recall": hits / n_obs if n_obs else np.nan,
            "precision": hits / (hits + fa) if hits + fa else np.nan, "csi": hits / (hits + miss + fa) if hits + miss + fa else np.nan}


def point(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    out = {"n": len(y), "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()), "bias": e.mean()}
    for t in (1, 10, 20, 64.5):
        c = contingency(y >= t, p >= t)
        out[f"recall_{t:g}"], out[f"precision_{t:g}"], out[f"csi_{t:g}"], out[f"events_{t:g}"] = c["recall"], c["precision"], c["csi"], c["events"]
    for lo, hi, name in BINS:
        m = (y >= lo) & (y < hi)
        out[f"bias[{name}]"] = e[m].mean() if m.any() else np.nan
        out[f"n[{name}]"] = int(m.sum())
    i = int(np.argmax(y))
    top = np.argsort(y)[::-1][:5]
    out["peak_obs"], out["peak_pred"] = y[i], p[i]
    out["top5_ratio"] = p[top].sum() / y[top].sum()
    return out


def interval(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return {"coverage": ((y >= lo) & (y <= hi)).mean(), "above": (y > hi).mean(), "below": (y < lo).mean(), "width": (hi - lo).mean()}


def skill(y, p, ref) -> dict:
    y, p, ref = (np.asarray(a, float) for a in (y, p, ref))
    return {"mae_skill": 1 - np.abs(p - y).mean() / np.abs(ref - y).mean(),
            "rmse_skill": 1 - np.sqrt(((p - y) ** 2).mean()) / np.sqrt(((ref - y) ** 2).mean())}


def prob(obs, pr) -> dict:
    obs, pr = np.asarray(obs, bool), np.asarray(pr, float)
    base = obs.mean()
    bs = ((pr - obs) ** 2).mean()
    bs_ref = ((base - obs) ** 2).mean()
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(obs, pr) if 0 < obs.sum() < len(obs) else np.nan
    except Exception:
        auc = np.nan
    # reliability: 5 bins
    bins = np.clip((pr * 5).astype(int), 0, 4)
    rel = [{"bin": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "n": int((bins == b).sum()), "forecast": float(pr[bins == b].mean()) if (bins == b).any() else None,
            "observed": float(obs[bins == b].mean()) if (bins == b).any() else None} for b in range(5)]
    return {"events": int(obs.sum()), "base_rate": base, "brier": bs, "bss": 1 - bs / bs_ref if bs_ref else np.nan, "auc": auc, "reliability": rel}


def tiers(obs, pr, cuts=(("watch", 0.2), ("alert", 0.4), ("warning", 0.6))) -> list[dict]:
    obs, pr = np.asarray(obs, bool), np.asarray(pr, float)
    out = []
    for name, c in cuts:
        f = pr >= c
        out.append({"tier": name, "min_prob": c, "issued": int(f.sum()), "hit_rate": float(obs[f].mean()) if f.any() else None,
                    "events_captured": float(f[obs].mean()) if obs.any() else None})
    return out


def bootstrap(df: pd.DataFrame, fn, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI resampling whole days (keeps cross-state correlation on a day)."""
    rng = np.random.default_rng(seed)
    days = df["date"].unique()
    groups = {d: idx for d, idx in df.groupby("date").indices.items()}
    vals = []
    for _ in range(reps):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        vals.append(fn(df.iloc[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

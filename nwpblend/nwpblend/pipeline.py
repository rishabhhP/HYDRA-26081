"""Training + test-prediction pipeline.

For each target: fit static blend, adaptive gating network, per-season LightGBM,
quantile models and the extreme classifiers attached to that target's table; save
artifacts; write test-set predictions (subgrid + spatial-holdout cells) that the
evaluation module scores.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from . import config as C
from .features import load_table
from .models.extremes import ExtremeClassifier
from .models.gating import GatingBlender
from .models.lgbm_season import SeasonalLGBM
from .models.quantile import QuantileModel

KEEP_COLS = ["date", "cell", "lead_days", "season_id", "regime_id", "region_id", "is_ocean", "coastal",
             "latitude", "longitude", "target", "expert_spread", "expert_range"] + \
            [f"exp_{e}" for e in C.EXPERTS] + C.FLAGS


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_target(split_name: str, target: str, full: bool = True) -> dict:
    art = C.ARTIFACTS_DIR / split_name / target
    res_dir = C.RESULTS_DIR / split_name
    res_dir.mkdir(parents=True, exist_ok=True)
    train = load_table(split_name, target, splits_keep=["train"])
    val = load_table(split_name, target, splits_keep=["val"])
    _log(f"{split_name}/{target}: train {len(train):,} val {len(val):,}")

    static = GatingBlender(static=True).fit(train, val)
    static.save(art, "gating_static")
    _log("  static blend done")
    gate = GatingBlender().fit(train, val, anchor_init=static.static_logits())
    gate.save(art, "gating")
    _log(f"  gating done ({len(gate.history)} epochs, best val {min(h['val_mse_norm'] for h in gate.history):.4f})")
    lgbm = SeasonalLGBM().fit(train, val)
    lgbm.save(art)
    lgbm.importance().to_csv(res_dir / f"lgbm_importance_{target}.csv", index=False)
    _log(f"  lgbm done (fallback seasons: {lgbm.fallback_seasons})")

    qm, clfs = None, {}
    if full:
        for df in (train, val):
            df["blend_pred"] = gate.predict(df)
        qm = QuantileModel().fit(train, val, nonneg=(target == "tp_mm"))
        qm.save(art)
        _log("  quantiles done")
        for flag, tab in C.FLAG_TABLE.items():
            if tab == target:
                clfs[flag] = ExtremeClassifier(flag).fit(train, val)
                clfs[flag].save(art)
                _log(f"  classifier {flag} done")
    del train, val

    cellsets = ["subgrid", "holdout"] if full else ["subgrid"]
    for cs in cellsets:
        parts = []
        for h in C.LEADS:
            te = load_table(split_name, target, leads=[h], cellset=cs, splits_keep=["test"])
            out = te[KEEP_COLS].copy()
            out["static_blend"] = static.predict(te)
            out["gating"], w = gate.predict(te, return_weights=True)
            for k, e in enumerate(C.EXPERTS):
                out[f"w_{e}"] = w[:, k].astype(np.float32)
            out["lgbm_season"] = lgbm.predict(te)
            if full:
                te["blend_pred"] = out["gating"].to_numpy()
                Q = qm.predict_raw(te)
                for j, q in enumerate(qm.levels):
                    out[f"q{int(round(q * 100)):02d}"] = Q[:, j].astype(np.float32)
                for name in ("80", "90"):
                    lo, hi = qm.interval(te, name, conformal=True, Q=Q)
                    out[f"cqr{name}_lo"], out[f"cqr{name}_hi"] = lo.astype(np.float32), hi.astype(np.float32)
                for flag, clf in clfs.items():
                    out[f"prob_{flag}"] = clf.predict_proba(te).astype(np.float32)
                    out[f"rawprob_{flag}"] = clf.predict_raw(te).astype(np.float32)
            parts.append(out)
            del te
        pd.concat(parts, ignore_index=True).to_parquet(res_dir / f"preds_{target}_{cs}.parquet", index=False)
        _log(f"  wrote test predictions ({cs})")
    return {"gating_epochs": len(gate.history), "lgbm_fallback": lgbm.fallback_seasons}


def save_shared_artifacts(split_name: str) -> None:
    """Fitted train-only statistics needed at inference (climatology, skill tables, statics)."""
    with open(C.FEATURES_DIR / split_name / "fitted_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    art = C.ARTIFACTS_DIR / split_name
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art / "climatology_by_doy.npz", **stats.clim_by_doy, msl_train_mean=stats.msl_train_mean)
    stats.region_skill.to_csv(art / "region_season_skill.csv", index=False)
    stats.static.reset_index().to_parquet(art / "grid_static.parquet", index=False)
    (art / "manifest.json").write_text(json.dumps({
        "split": split_name, "targets": C.TARGETS, "leads_days": C.LEADS, "experts": C.EXPERTS,
        "flags": C.FLAGS, "seasons": C.SEASONS, "regimes": C.REGIMES, "quantiles": C.QUANTILES,
        "n_train_days": stats.meta.get("n_train_days"),
        "thresholds": {"heavy_rain_mm": C.HEAVY_RAIN_MM, "heatwave_tmax_c": C.HEATWAVE_TMAX_C,
                       "high_wind_max_ms": C.HIGH_WIND_MAX_MS},
    }, indent=1))


def run(split_name: str = "blocked", full: bool = True, targets=C.TARGETS) -> None:
    save_shared_artifacts(split_name)
    for t in targets:
        train_target(split_name, t, full=full)

"""Inference entry point for the backend.

    from nwpblend.inference import BlendingForecaster
    fc = BlendingForecaster("artifacts/blocked")
    out = fc.forecast(history_df, issue_date="2025-12-28", leads=(1, 2, 3))

``history_df`` = daily truth rows for the FULL 0.25-deg grid (schema of
``data/processed/era5_india_2025_daily_processed.parquet``) covering at least the 35 days up to
and including ``issue_date``. Nothing after ``issue_date`` is read.

Optional untrained NWP source: pass ``nwp={"t2m_C_mean": df_with_lat_lon_value, ...}`` and
``nwp_weight={"t2m_C_mean": 0.3}`` to mix a raw NWP field into the final blend with a
*user-chosen* weight. No weight is learned for it (no verifiable overlap exists - see report s.9c),
so the default is off and the output carries ``nwp_weight_<target>`` for auditability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import CUBE_VARS, FeatureBuilder, FittedStats
from .models.extremes import ExtremeClassifier
from .models.gating import GatingBlender
from .models.lgbm_season import SeasonalLGBM
from .models.quantile import QuantileModel

MIN_HISTORY_DAYS = 35


class BlendingForecaster:
    def __init__(self, artifacts_dir: str | Path = C.ARTIFACTS_DIR / "blocked"):
        d = Path(artifacts_dir)
        self.dir = d
        self.manifest = json.loads((d / "manifest.json").read_text())
        z = np.load(d / "climatology_by_doy.npz")
        static = pd.read_parquet(d / "grid_static.parquet").set_index("cell")
        self.stats = FittedStats(clim_by_doy={t: z[t] for t in C.TARGETS}, msl_train_mean=z["msl_train_mean"],
                                 region_skill=pd.read_csv(d / "region_season_skill.csv"), static=static)
        self.models = {}
        for t in C.TARGETS:
            td = d / t
            m = {"gating": GatingBlender.load(td, "gating"), "static": GatingBlender.load(td, "gating_static"),
                 "lgbm": SeasonalLGBM.load(td)}
            if (td / "quantile_meta.json").exists():
                m["quantile"] = QuantileModel.load(td)
            m["clf"] = {f: ExtremeClassifier.load(td, f) for f, tab in C.FLAG_TABLE.items()
                        if tab == t and (td / f"clf_{f}.txt").exists()}
            self.models[t] = m

    # ------------------------------------------------------------ features
    def build_features(self, history: pd.DataFrame, issue_date, leads=C.LEADS, cells=None) -> dict[str, pd.DataFrame]:
        issue = pd.Timestamp(issue_date)
        h = history[(history["date"] <= issue) & (history["date"] > issue - pd.Timedelta(days=MIN_HISTORY_DAYS + 5))]
        h = h.sort_values(["date", "latitude", "longitude"], ascending=[True, False, True])
        dates = pd.DatetimeIndex(h["date"].unique())
        N = len(self.stats.static)
        if len(dates) < MIN_HISTORY_DAYS or dates[-1] != issue:
            raise ValueError(f"need >= {MIN_HISTORY_DAYS} consecutive days of history ending on {issue.date()}")
        if len(h) != len(dates) * N:
            raise ValueError("history must cover the full grid on every day")
        pad = max(leads)
        all_dates = dates.append(pd.date_range(issue + pd.Timedelta(days=1), periods=pad))
        cubes = {}
        for v in CUBE_VARS + C.FLAGS:
            arr = h[v].to_numpy(np.float32).reshape(len(dates), N) if v in h else np.zeros((len(dates), N), np.float32)
            cubes[v] = np.vstack([arr, np.full((pad, N), np.nan, np.float32)])
        fb = FeatureBuilder(cubes, all_dates, self.stats.static)
        cells = np.arange(N) if cells is None else np.asarray(cells)
        out = {}
        for t in C.TARGETS:
            parts = [fb.table(t, L, self.stats, cells, days=np.array([len(dates) - 1 + L])) for L in leads]
            out[t] = pd.concat(parts, ignore_index=True)
        return out

    # ------------------------------------------------------------ prediction
    def predict(self, feats: dict[str, pd.DataFrame], nwp: dict | None = None, nwp_weight: dict | None = None) -> pd.DataFrame:
        base = None
        for t, df in feats.items():
            m = self.models[t]
            res = pd.DataFrame({"cell": df["cell"], "latitude": df["latitude"], "longitude": df["longitude"],
                                "target_date": df["date"], "lead_days": df["lead_days"]})
            p, w = m["gating"].predict(df, return_weights=True)
            w_nwp = float((nwp_weight or {}).get(t, 0.0))
            if nwp and t in nwp and w_nwp > 0:
                f = nwp[t].set_index(["latitude", "longitude"])["value"]
                raw = f.reindex(pd.MultiIndex.from_arrays([df["latitude"], df["longitude"]])).to_numpy()
                p = np.where(np.isfinite(raw), (1 - w_nwp) * p + w_nwp * raw, p)
            res[f"pred_{t}"] = p
            res[f"nwp_weight_{t}"] = w_nwp
            res[f"lgbm_{t}"] = m["lgbm"].predict(df)
            res[f"static_{t}"] = m["static"].predict(df)
            for k, e in enumerate(C.EXPERTS):
                res[f"w_{e}_{t}"] = w[:, k]
                res[f"exp_{e}_{t}"] = df[f"exp_{e}"].to_numpy()
            res[f"spread_{t}"] = df["expert_spread"].to_numpy()
            df = df.assign(blend_pred=p)
            if "quantile" in m:
                qm = m["quantile"]
                Q = qm.predict_raw(df)
                for j, q in enumerate(qm.levels):
                    res[f"q{int(round(q * 100)):02d}_{t}"] = Q[:, j]
                res[f"lo80_{t}"], res[f"hi80_{t}"] = qm.interval(df, "80", Q=Q)
            for f, clf in m["clf"].items():
                res[f"prob_{f}"] = clf.predict_proba(df)
            res["regime"] = np.array(C.REGIMES)[df["regime_id"].to_numpy()]
            base = res if base is None else base.merge(
                res.drop(columns=["latitude", "longitude", "regime"]), on=["cell", "target_date", "lead_days"])
        return base

    def forecast(self, history: pd.DataFrame, issue_date, leads=C.LEADS, cells=None, nwp=None, nwp_weight=None) -> pd.DataFrame:
        return self.predict(self.build_features(history, issue_date, leads, cells), nwp, nwp_weight)

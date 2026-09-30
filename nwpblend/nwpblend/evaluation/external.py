"""External / out-of-domain checks that do not share data with the ERA5 training set.

1. Meteostat station (2025): which ERA5 cell it matches, how well ERA5 "truth" agrees with a
   real gauge, and how the ERA5-trained forecasts score against the station on test days.
2. IMD statewise rainfall (Aug 19 - Sep 24 2026): a genuinely unseen year. Experts rebuilt
   from IMD data (climatology = IMD long-period daily normal), blended with the weights the
   gating network learned for monsoon rainfall, compared against every single expert.
3. ECMWF 0 h snapshot (2026-09-24): documents why no skill number can be computed.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .. import config as C
from ..features import FeatureBuilder, load_cubes
from ..ingest import load_ecmwf_snapshot, load_imd_statewise, load_meteostat
from ..models.gating import GatingBlender
from ..models.lgbm_season import SeasonalLGBM
from ..splits import SPLITTERS
from . import metrics as M


def meteostat_check(split: str = "blocked") -> dict[str, pd.DataFrame]:
    st = load_meteostat()
    st = st[(st.station_outage_flag == 0) & (st.possible_outlier_flag == 0)].set_index("date")
    cubes, dates, static = load_cubes()
    t = cubes["t2m_C_mean"]
    obs = st["temp_avg_C"].reindex(dates).to_numpy()
    ok = np.isfinite(obs)
    tz = (t[ok] - t[ok].mean(0)) / (t[ok].std(0) + 1e-6)
    oz = (obs[ok] - obs[ok].mean()) / obs[ok].std()
    corr = (tz * oz[:, None]).mean(0)
    # temperature correlation alone favours broad regions; combine with mean-level agreement
    score = corr - 0.05 * np.abs(t[ok].mean(0) - obs[ok].mean())
    cell = int(np.nanargmax(score))
    lat, lon = float(static.loc[cell, "latitude"]), float(static.loc[cell, "longitude"])
    pairs = {
        "t2m_C_mean": ("temp_avg_C", 1.0),
        "tp_mm": ("precip_mm", 1.0),
        "wind_speed_mean": ("wind_speed_kmh", 1 / 3.6),
    }
    agree = []
    for v, (col, k) in pairs.items():
        o = st[col].reindex(dates).to_numpy() * k
        e = cubes[v][:, cell]
        m = np.isfinite(o)
        agree.append({"variable": v, "n_days": int(m.sum()), "station_mean": float(o[m].mean()),
                      "era5_mean": float(e[m].mean()), "bias_era5_minus_station": float((e[m] - o[m]).mean()),
                      "rmse": M.rmse(o[m], e[m]), "pearson_r": float(np.corrcoef(o[m], e[m])[0, 1])})
    agree = pd.DataFrame(agree)
    agree.insert(0, "matched_cell", f"{lat:.2f}N {lon:.2f}E (t2m r={corr[cell]:.3f})")

    # forecast skill at that cell against the station, on TEST days
    with open(C.FEATURES_DIR / split / "fitted_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    splits = SPLITTERS[split](dates)
    fb = FeatureBuilder(cubes, dates, static)
    rows = []
    for v, (col, k) in pairs.items():
        art = C.ARTIFACTS_DIR / split / v
        gate, lgbm = GatingBlender.load(art, "gating"), SeasonalLGBM.load(art)
        for h in C.LEADS:
            tab = fb.table(v, h, stats, np.array([cell]), splits)
            tab = tab[tab.split == "test"].copy()
            tab["station"] = st[col].reindex(pd.DatetimeIndex(tab["date"])).to_numpy() * k
            tab = tab[np.isfinite(tab["station"])]
            preds = {f"{e}": tab[f"exp_{e}"] for e in C.EXPERTS}
            preds["gating_adaptive"] = gate.predict(tab)
            preds["lgbm_season"] = lgbm.predict(tab)
            for name, p in preds.items():
                rows.append({"variable": v, "lead_days": h, "model": name, "n": len(tab),
                             "rmse_vs_station": M.rmse(tab["station"], p), "rmse_vs_era5": M.rmse(tab["target"], p)})
    skill = pd.DataFrame(rows).groupby(["variable", "model"], as_index=False)[["n", "rmse_vs_station", "rmse_vs_era5"]].mean()
    return {"agreement": agree, "skill": skill}


def imd_transfer(split: str = "blocked") -> dict[str, pd.DataFrame]:
    imd = load_imd_statewise()
    imd = imd[imd.state_type == "state_ut"].sort_values(["state", "date"])
    # learned monsoon rainfall weights (mean adaptive gating weights on monsoon test rows, per lead)
    preds = pd.read_parquet(C.RESULTS_DIR / split / "preds_tp_mm_subgrid.parquet",
                            columns=["season_id", "lead_days"] + [f"w_{e}" for e in C.EXPERTS])
    mw = preds[preds.season_id == C.SEASONS.index("monsoon")].groupby("lead_days")[[f"w_{e}" for e in C.EXPERTS]].mean()
    static = GatingBlender.load(C.ARTIFACTS_DIR / split / "tp_mm", "gating_static")
    probe = pd.DataFrame({"season_id": [C.SEASONS.index("monsoon")] * 3, "lead_days": C.LEADS,
                          **{f"exp_{e}": [0.0] * 3 for e in C.EXPERTS}})
    sw = pd.DataFrame(static.weights(probe), index=C.LEADS, columns=[f"w_{e}" for e in C.EXPERTS])
    rows = []
    for h in C.LEADS:
        parts = []
        for s, g in imd.groupby("state"):
            a, nrm = g["daily_actual_mm"].to_numpy(), g["daily_normal_mm"].to_numpy()
            n = len(a)
            if n <= h + 2:
                continue
            idx = np.arange(h + 2, n)
            ex = {"climatology": nrm[idx], "persistence": a[idx - h],
                  "recent3": np.array([a[i - h - 2:i - h + 1].mean() for i in idx]),
                  "anom_persistence": np.clip(nrm[idx] + a[idx - h] - nrm[idx - h], 0, None)}
            parts.append(pd.DataFrame({"state": s, "y": a[idx], **ex}))
        df = pd.concat(parts)
        E = df[C.EXPERTS].to_numpy()
        df["blend_learned_adaptive_mean_w"] = E @ mw.loc[h].to_numpy()
        df["blend_learned_static_w"] = E @ sw.loc[h].to_numpy()
        for mcol in C.EXPERTS + ["blend_learned_adaptive_mean_w", "blend_learned_static_w"]:
            rows.append({"lead_days": h, "model": mcol, "n_state_days": len(df), "n_states": df.state.nunique(),
                         "rmse": M.rmse(df.y, df[mcol]), "mae": M.mae(df.y, df[mcol])})
    res = pd.DataFrame(rows)
    weights = pd.concat({"adaptive_mean": mw, "static": sw}, names=["kind", "lead_days"]).reset_index()
    return {"skill": res, "weights": weights}


def ecmwf_note() -> pd.DataFrame:
    fc = load_ecmwf_snapshot()
    return pd.DataFrame([{
        "grid_points": len(fc), "init_time": str(fc.init_time.iloc[0].date()), "valid_time": str(fc.valid_time.iloc[0].date()),
        "lead_hours_unique": ",".join(map(str, sorted(fc.lead_hours.unique()))),
        "tp_mm_max": float(fc.tp_mm.max()), "ssrd_max": float(fc.ssrd_MJ.max()),
        "truth_available_at_valid_time": "none gridded (ERA5 is 2025; IMD is statewise and ends at this date, daily accumulations)",
        "verdict": "no verifiable comparison point -> ECMWF cannot be scored or used as a trained expert",
    }])

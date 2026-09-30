"""Benchmark tables. Every function reads the saved test predictions and writes a CSV under
``reports/results/<split>/``; ``report.py`` renders those CSVs into the markdown report."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config as C
from ..features import load_table, region_name
from . import metrics as M

EXPERT_MODELS = {f"exp_{e}": e for e in C.EXPERTS}
BLEND_MODELS = {"static_blend": "static_blend", "gating": "gating_adaptive", "lgbm_season": "lgbm_season", "q50": "quantile_median"}
POINT_MODELS = {**EXPERT_MODELS, **BLEND_MODELS}
TARGET_UNITS = {"tp_mm": "mm/day", "t2m_C_mean": "degC", "wind_speed_mean": "m/s"}


def load_preds(split: str, target: str, cellset: str = "subgrid") -> pd.DataFrame:
    df = pd.read_parquet(C.RESULTS_DIR / split / f"preds_{target}_{cellset}.parquet")
    df["season"] = pd.Categorical(np.array(C.SEASONS)[df["season_id"].to_numpy()], categories=C.SEASONS)
    df["regime"] = pd.Categorical(np.array(C.REGIMES)[df["regime_id"].to_numpy()], categories=C.REGIMES)
    df["zone"] = np.where(df.is_ocean == 1, "ocean", np.where(df.coastal == 1, "coastal_land", "inland"))
    return df


def _models_in(df):
    return {k: v for k, v in POINT_MODELS.items() if k in df.columns}


# ------------------------------------------------------------------ deterministic
def point_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    models = _models_in(df)
    work = pd.DataFrame({c: df[c] for c in by})
    for col, name in models.items():
        e = df[col].to_numpy(np.float64) - df["target"].to_numpy(np.float64)
        work[f"se|{name}"] = e * e
        work[f"ae|{name}"] = np.abs(e)
    g = work.groupby(by, observed=True)
    mean = g.mean()
    n = g.size().rename("n")
    rows = []
    for key, r in mean.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        for name in models.values():
            rows.append(dict(zip(by, key), model=name, rmse=np.sqrt(r[f"se|{name}"]), mae=r[f"ae|{name}"], n=int(n.loc[key if len(key) > 1 else key[0]])))
    return pd.DataFrame(rows)


def improvement_table(df: pd.DataFrame, by: list[str], blends=("gating_adaptive", "lgbm_season", "static_blend"),
                      bootstrap: bool = True) -> pd.DataFrame:
    """% improvement of each blend over EACH expert and over the best single expert in the slice."""
    pm = point_metrics(df, by)
    inv = {v: k for k, v in POINT_MODELS.items()}
    rows = []
    for key, g in pm.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        sel = df
        for c, v in zip(by, key):
            sel = sel[sel[c] == v]
        g = g.set_index("model")
        experts = [e for e in C.EXPERTS if e in g.index]
        best = min(experts, key=lambda e: g.loc[e, "rmse"])
        for b in blends:
            if b not in g.index:
                continue
            r = dict(zip(by, key), blend=b, rmse=g.loc[b, "rmse"], mae=g.loc[b, "mae"], best_expert=best, n=int(g.loc[b, "n"]))
            for e in experts:
                r[f"rmse_gain_vs_{e}_%"] = M.improvement_pct(g.loc[b, "rmse"], g.loc[e, "rmse"])
                r[f"mae_gain_vs_{e}_%"] = M.improvement_pct(g.loc[b, "mae"], g.loc[e, "mae"])
            r["rmse_gain_vs_best_%"] = M.improvement_pct(g.loc[b, "rmse"], g.loc[best, "rmse"])
            r["beats_every_expert_rmse"] = bool(all(g.loc[b, "rmse"] < g.loc[e, "rmse"] for e in experts))
            r["beats_every_expert_mae"] = bool(all(g.loc[b, "mae"] < g.loc[e, "mae"] for e in experts))
            if bootstrap:
                lo, hi = M.day_block_bootstrap_rmse_gain(sel["target"].to_numpy(), sel[inv[b]].to_numpy(),
                                                         sel[inv[best]].to_numpy(), sel["date"].to_numpy())
                r["gain_vs_best_ci95_lo"], r["gain_vs_best_ci95_hi"] = lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ probabilistic
def climatological_quantile_baselines(split: str, target: str, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Two reference probabilistic forecasts, fitted on TRAIN rows only:
    * clim_quantiles: per-cell x season empirical quantiles of the truth
    * persistence_resid: persistence + per season x lead empirical quantiles of (truth - persistence)
    """
    tr = load_table(split, target, splits_keep=["train"], columns=["cell", "season_id", "lead_days", "target", "exp_persistence", "split"])
    lv = C.QUANTILES
    cq = tr[tr.lead_days == 1].groupby(["cell", "season_id"])["target"].quantile(lv).unstack()
    key = pd.MultiIndex.from_arrays([test["cell"], test["season_id"]])
    clim = cq.reindex(key).to_numpy()
    tr["res"] = tr["target"] - tr["exp_persistence"]
    rq = tr.groupby(["season_id", "lead_days"])["res"].quantile(lv).unstack()
    key2 = pd.MultiIndex.from_arrays([test["season_id"], test["lead_days"]])
    pers = test["exp_persistence"].to_numpy()[:, None] + rq.reindex(key2).to_numpy()
    if target == "tp_mm":
        pers = np.clip(pers, 0, None)
    return {"clim_quantiles": np.sort(clim, 1), "persistence_resid": np.sort(pers, 1)}


def quantile_metrics(split: str, target: str, df: pd.DataFrame) -> pd.DataFrame:
    lv = C.QUANTILES
    qcols = [f"q{int(round(q * 100)):02d}" for q in lv]
    base = climatological_quantile_baselines(split, target, df)
    rows = []
    for s in C.SEASONS + ["ALL"]:
        m = np.ones(len(df), bool) if s == "ALL" else (df.season == s).to_numpy()
        if not m.any():
            continue
        y = df["target"].to_numpy()[m]
        Qm = df.loc[m, qcols].to_numpy()
        r = {"season": s, "n": int(m.sum())}
        r["crps_model"] = M.crps_from_quantiles(y, Qm, lv)
        for bname, Qb in base.items():
            r[f"crps_{bname}"] = M.crps_from_quantiles(y, Qb[m], lv)
            r[f"crpss_vs_{bname}_%"] = M.improvement_pct(r["crps_model"], r[f"crps_{bname}"])
        for j, q in enumerate(lv):
            r[f"pinball_q{int(q * 100):02d}"] = M.pinball(y, Qm[:, j], q)
        r["cov80_raw"] = M.coverage(y, df.loc[m, "q10"], df.loc[m, "q90"])
        r["cov80_cqr"] = M.coverage(y, df.loc[m, "cqr80_lo"], df.loc[m, "cqr80_hi"])
        r["cov90_raw"] = M.coverage(y, df.loc[m, "q05"], df.loc[m, "q95"])
        r["cov90_cqr"] = M.coverage(y, df.loc[m, "cqr90_lo"], df.loc[m, "cqr90_hi"])
        r["width80_raw"] = float((df.loc[m, "q90"] - df.loc[m, "q10"]).mean())
        r["width80_cqr"] = float((df.loc[m, "cqr80_hi"] - df.loc[m, "cqr80_lo"]).mean())
        r["cov80_clim"] = M.coverage(y, base["clim_quantiles"][m, 1], base["clim_quantiles"][m, 5])
        r["width80_clim"] = float(np.nanmean(base["clim_quantiles"][m, 5] - base["clim_quantiles"][m, 1]))
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ extremes
PERSIST_RULE = {
    "heavy_rain_day_flag": lambda t: t["now_tp_mm"] >= C.HEAVY_RAIN_MM,
    "heatwave_day_flag": lambda t: (t["now_t2m_C_max"] >= C.HEATWAVE_TMAX_C) & (t["is_ocean"] == 0),
    "high_wind_day_flag": lambda t: t["now_wind_speed_max"] >= C.HIGH_WIND_MAX_MS,
}


def extreme_metrics(split: str, flag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = C.FLAG_TABLE[flag]
    df = load_preds(split, target)
    now = load_table(split, target, splits_keep=["test"],
                     columns=["date", "cell", "is_ocean", "now_tp_mm", "now_t2m_C_max", "now_wind_speed_max", "split"])
    assert (now["cell"].to_numpy() == df["cell"].to_numpy()).all(), "row alignment"
    df["persist_flag"] = PERSIST_RULE[flag](now).astype(float).to_numpy()
    rows, rel = [], []
    for s in C.SEASONS + ["ALL"]:
        for lead in ["all"] + C.LEADS:
            m = np.ones(len(df), bool) if s == "ALL" else (df.season == s).to_numpy().copy()
            if lead != "all":
                m = m & (df.lead_days == lead).to_numpy()
            if not m.any():
                continue
            y, day = df.loc[m, flag].to_numpy(), df.loc[m, "date"].to_numpy()
            for name, col in [("calibrated", f"prob_{flag}"), ("uncalibrated", f"rawprob_{flag}"), ("persistence_flag", "persist_flag")]:
                r = M.binary_metrics(y, df.loc[m, col].to_numpy(), day)
                rows.append({"flag": flag, "season": s, "lead": lead, "model": name, **r})
            if lead == "all" and s == "ALL":
                for name, col in [("calibrated", f"prob_{flag}"), ("uncalibrated", f"rawprob_{flag}")]:
                    t = M.reliability_table(y, df.loc[m, col].to_numpy())
                    t.insert(0, "model", name)
                    t.insert(0, "flag", flag)
                    rel.append(t)
    return pd.DataFrame(rows), pd.concat(rel, ignore_index=True)


# ------------------------------------------------------------------ disagreement
def disagreement_metrics(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    err = np.abs(df["gating"] - df["target"])
    for s in C.SEASONS + ["ALL"]:
        m = np.ones(len(df), bool) if s == "ALL" else (df.season == s).to_numpy()
        sub = df[m]
        e = err[m]
        rho = spearmanr(sub["expert_spread"], e).statistic
        q = pd.qcut(sub["expert_spread"].rank(method="first"), 5, labels=False)
        by = pd.DataFrame({"q": q, "se": e ** 2, "sp": sub["expert_spread"],
                           "w80": sub["cqr80_hi"] - sub["cqr80_lo"],
                           "cov": (sub["target"] >= sub["cqr80_lo"]) & (sub["target"] <= sub["cqr80_hi"])}).groupby("q")
        agg = by.agg(spread=("sp", "mean"), rmse=("se", lambda v: np.sqrt(v.mean())), width80=("w80", "mean"), cov80=("cov", "mean"))
        r = {"target": target, "season": s, "spearman_spread_vs_abs_err": float(rho)}
        for qi, a in agg.iterrows():
            r[f"Q{qi + 1}_spread"], r[f"Q{qi + 1}_rmse"] = a.spread, a.rmse
            r[f"Q{qi + 1}_width80"], r[f"Q{qi + 1}_cov80"] = a.width80, a.cov80
        r["rmse_ratio_top_vs_bottom_quintile"] = agg.rmse.iloc[-1] / agg.rmse.iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ weight maps
def gating_weight_maps(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    wcols = [f"w_{e}" for e in C.EXPERTS]
    out = {}
    out["season_region"] = df.groupby(["season", "region_id"], observed=True)[wcols].mean().reset_index()
    out["season_region"]["region"] = out["season_region"]["region_id"].map(region_name)
    out["season_zone"] = df.groupby(["season", "zone"], observed=True)[wcols].mean().reset_index()
    out["season_lead"] = df.groupby(["season", "lead_days"], observed=True)[wcols].mean().reset_index()
    out["regime"] = df.groupby(["regime"], observed=True)[wcols + ["target"]].agg(
        {**{w: "mean" for w in wcols}, "target": "size"}).rename(columns={"target": "n"}).reset_index()
    out["cell"] = df[df.lead_days == 1].groupby(["season", "latitude", "longitude"], observed=True)[wcols].mean().reset_index()
    for k in out:
        out[k].insert(0, "target", target)
    return out


def weight_sentences(zone_tab: pd.DataFrame, target: str) -> list[str]:
    """Auto-generated plain-English readings of the weight maps."""
    out = []
    for s, g in zone_tab.groupby("season", observed=True):
        g = g.set_index("zone")
        for z in g.index:
            w = g.loc[z, [f"w_{e}" for e in C.EXPERTS]].astype(float)
            top = w.idxmax().replace("w_", "")
            wc = float(g.loc[z, "w_climatology"])
            wp = float(g.loc[z, "w_persistence"])
            ratio = f"{wp / wc:.1f}x" if wc > 0.01 else "far more than"
            out.append(f"{target} / {s} / {z}: top expert = {top} ({w.max():.0%}); persistence weighted {ratio} climatology "
                       f"(pers {wp:.0%}, clim {wc:.0%}, recent3 {float(g.loc[z, 'w_recent3']):.0%}, anom-pers {float(g.loc[z, 'w_anom_persistence']):.0%})")
    return out


# ------------------------------------------------------------------ driver
def run_all(split: str = "blocked", full: bool = True) -> None:
    out = C.RESULTS_DIR / split
    agg: dict[str, list] = {k: [] for k in ["pm_season", "pm_season_lead", "pm_all", "imp_season", "imp_lead",
                                             "imp_regime", "imp_zone", "hold_pm", "hold_imp", "quant", "disagree",
                                             "w_season_region", "w_season_zone", "w_season_lead", "w_regime", "w_cell"]}
    for t in C.TARGETS:
        df = load_preds(split, t)
        if not full:
            df = df.drop(columns=["q50"], errors="ignore")
        for key, fn in [("pm_season", lambda d: point_metrics(d, ["season"])),
                        ("pm_season_lead", lambda d: point_metrics(d, ["season", "lead_days"])),
                        ("pm_all", lambda d: point_metrics(d.assign(all="ALL"), ["all"])),
                        ("imp_season", lambda d: improvement_table(d, ["season"])),
                        ("imp_lead", lambda d: improvement_table(d, ["season", "lead_days"], bootstrap=False))]:
            agg[key].append(fn(df).assign(target=t))
        if full:
            agg["imp_regime"].append(improvement_table(df, ["regime"], bootstrap=False).assign(target=t))
            agg["imp_zone"].append(improvement_table(df, ["season", "zone"], bootstrap=False).assign(target=t))
            agg["quant"].append(quantile_metrics(split, t, df).assign(target=t))
            agg["disagree"].append(disagreement_metrics(df, t))
            for k, v in gating_weight_maps(df, t).items():
                agg[f"w_{k}"].append(v)
            h = load_preds(split, t, "holdout")
            agg["hold_pm"].append(point_metrics(h, ["season"]).assign(target=t))
            agg["hold_imp"].append(improvement_table(h, ["season"], bootstrap=False).assign(target=t))
        del df
    for k, v in agg.items():
        if v:
            pd.concat(v, ignore_index=True).to_csv(out / f"{k}.csv", index=False)
    if full:
        ex, rel = zip(*[extreme_metrics(split, f) for f in C.FLAGS])
        pd.concat(ex, ignore_index=True).to_csv(out / "extremes.csv", index=False)
        pd.concat(rel, ignore_index=True).to_csv(out / "reliability.csv", index=False)

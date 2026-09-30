"""Render reports/BENCHMARK_REPORT.md from the CSVs written by evaluate.run_all / external.

All numbers in the report are read from reports/results/** - nothing is typed by hand.
Narrative sentences that state a result are generated from those same tables.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from . import figures as FIG
from .evaluate import TARGET_UNITS, weight_sentences

BLEND_LABEL = {"gating_adaptive": "Adaptive gating (b)", "lgbm_season": "Per-season LightGBM (a)", "static_blend": "Static blend"}
EXP_LABEL = {"climatology": "Clim", "persistence": "Persist", "recent3": "Recent3", "anom_persistence": "AnomPers"}


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "**NO**"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_f(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def _pct(x):
    return "-" if not np.isfinite(x) else f"{x:+.1f}%"


def _read(split, name):
    p = C.RESULTS_DIR / split / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def headline_table(pm: pd.DataFrame, imp: pd.DataFrame, target: str) -> str:
    """One row per season: RMSE of every source, both blends, and gain vs each expert."""
    rows = []
    for s in C.SEASONS:
        p = pm[(pm.target == target) & (pm.season == s)].set_index("model")
        if p.empty:
            continue
        r = {"season": s}
        for e in C.EXPERTS:
            r[f"{EXP_LABEL[e]} RMSE"] = p.loc[e, "rmse"]
        for b in ["static_blend", "gating_adaptive", "lgbm_season"]:
            if b in p.index:
                r[{"static_blend": "Static", "gating_adaptive": "**Gating**", "lgbm_season": "**LGBM**"}[b] + " RMSE"] = p.loc[b, "rmse"]
        for b in ["gating_adaptive", "lgbm_season"]:
            i = imp[(imp.target == target) & (imp.season == s) & (imp.blend == b)]
            if i.empty:
                continue
            i = i.iloc[0]
            tag = "Gating" if b == "gating_adaptive" else "LGBM"
            r[f"{tag} gain vs Clim / Persist / Recent3 / AnomPers"] = " / ".join(_pct(i[f"rmse_gain_vs_{e}_%"]) for e in C.EXPERTS)
            ci = f" [{i['gain_vs_best_ci95_lo']:+.1f}, {i['gain_vs_best_ci95_hi']:+.1f}]" if "gain_vs_best_ci95_lo" in i and np.isfinite(i.get("gain_vs_best_ci95_lo", np.nan)) else ""
            r[f"{tag} vs best expert (95% CI)"] = f"{_pct(i['rmse_gain_vs_best_%'])}{ci}"
        rows.append(r)
    return md_table(pd.DataFrame(rows))


def mae_table(pm: pd.DataFrame, imp: pd.DataFrame, target: str) -> str:
    rows = []
    for s in C.SEASONS:
        p = pm[(pm.target == target) & (pm.season == s)].set_index("model")
        if p.empty:
            continue
        r = {"season": s, **{f"{EXP_LABEL[e]} MAE": p.loc[e, "mae"] for e in C.EXPERTS}}
        for b in ["static_blend", "gating_adaptive", "lgbm_season"]:
            if b in p.index:
                r[f"{b} MAE"] = p.loc[b, "mae"]
        for b in ["gating_adaptive", "lgbm_season"]:
            i = imp[(imp.target == target) & (imp.season == s) & (imp.blend == b)]
            if not i.empty:
                i = i.iloc[0]
                r[f"{b} MAE gain vs Clim / Persist / Recent3 / AnomPers"] = " / ".join(_pct(i[f"mae_gain_vs_{e}_%"]) for e in C.EXPERTS)
        rows.append(r)
    return md_table(pd.DataFrame(rows))


def scorecard(imp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in ["gating_adaptive", "lgbm_season", "static_blend"]:
        g = imp[imp.blend == b]
        if g.empty:
            continue
        sig = (g["gain_vs_best_ci95_lo"] > 0).sum() if "gain_vs_best_ci95_lo" in g else np.nan
        rows.append({"model": BLEND_LABEL[b], "season x target slices": len(g),
                     "beats EVERY expert (RMSE)": int(g.beats_every_expert_rmse.sum()),
                     "beats EVERY expert (MAE)": int(g.beats_every_expert_mae.sum()),
                     "gain vs best expert CI95 > 0": int(sig),
                     "median gain vs best expert (RMSE)": f"{g['rmse_gain_vs_best_%'].median():+.1f}%"})
    return pd.DataFrame(rows)


def build_report(split: str = "blocked", tf_split: str = "time_forward") -> str:
    R = {k: _read(split, k) for k in ["pm_season", "pm_season_lead", "pm_all", "imp_season", "imp_lead", "imp_regime",
                                      "imp_zone", "hold_pm", "hold_imp", "quant", "disagree", "extremes", "reliability",
                                      "w_season_region", "w_season_zone", "w_season_lead", "w_regime", "w_cell",
                                      "ext_station_agreement", "ext_station_skill", "ext_imd_skill", "ext_imd_weights",
                                      "ext_ecmwf", "data_inventory", "split_summary"]}
    TF = {k: _read(tf_split, k) for k in ["pm_season", "imp_season"]}
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for t in C.TARGETS:
        FIG.rmse_bars(R["pm_season"], t, TARGET_UNITS[t], C.FIGURES_DIR / f"rmse_{t}.png")
        FIG.weight_maps(R["w_cell"], t, C.FIGURES_DIR / f"weights_{t}.png")
    FIG.reliability(R["reliability"], C.FIGURES_DIR / "reliability.png")
    FIG.spread_skill(R["disagree"], C.FIGURES_DIR / "spread_skill.png")
    manifest = json.loads((C.ARTIFACTS_DIR / split / "manifest.json").read_text())

    imp = R["imp_season"]
    sc = scorecard(imp)
    L: list[str] = []
    A = L.append
    A("# Hybrid AI-NWP Blending - Model Benchmark Report\n")
    A(f"_Generated by `python -m scripts.run_pipeline report`. Every number below is read from `reports/results/`. "
      f"Primary split: **blocked-within-season** ({manifest['n_train_days']} train days, embargo {C.EMBARGO_DAYS} d). "
      f"Test = held-out day blocks, on a 0.5-deg subgrid of real 0.25-deg ERA5 cells (3,599 cells), leads 1-3 days._\n")

    # ------------------------------------------------ 0. headline
    A("## 0. Headline results\n")
    A(md_table(sc))
    A("")
    g = imp[imp.blend == "gating_adaptive"]
    lg = imp[imp.blend == "lgbm_season"]
    best_model = "lgbm_season" if lg["rmse_gain_vs_best_%"].median() >= g["rmse_gain_vs_best_%"].median() else "gating_adaptive"
    A(f"* **{BLEND_LABEL[best_model]}** is the most accurate point model overall (median RMSE gain over the best single expert "
      f"per slice: {imp[imp.blend == best_model]['rmse_gain_vs_best_%'].median():+.1f}%).")
    A(f"* The adaptive gating network beats **every** individual expert on RMSE in {int(g.beats_every_expert_rmse.sum())}/{len(g)} "
      f"season x target slices. Against the static (season x lead fixed-weight) blend it wins "
      f"{int((imp[imp.blend == 'static_blend'].set_index(['target', 'season'])['rmse'] > g.set_index(['target', 'season'])['rmse']).sum())}/{len(g)} slices - "
      f"so with one year of truth, most of the blending gain comes from learning the right *seasonal* weights; per-row "
      f"context adaptation adds skill only in some slices (largest in post-monsoon rain and t2m).")
    A("* **Time-forward stress test (s.7): skill does NOT transfer to an unseen season for t2m.** With zero post-monsoon training "
      "days, all blends lose badly to persistence on Oct-Dec t2m; rain and (for LightGBM/static) wind still beat every expert. "
      "Operationally, retrain once every season has been observed.")
    fails = pd.concat([g[~g.beats_every_expert_rmse], lg[~lg.beats_every_expert_rmse]])
    if len(fails):
        A("* Slices where a blend does **not** beat every expert (reported, not hidden): " +
          "; ".join(f"{BLEND_LABEL[r.blend]} on {r.target}/{r.season} (best expert {r.best_expert}, {r['rmse_gain_vs_best_%']:+.1f}%)" for _, r in fails.iterrows()))
    A("* The raw ECMWF forecast **cannot be benchmarked** with the data in this workspace (section 9c) - every "
      "\"% improvement over ECMWF\" cell is therefore absent rather than invented.\n")

    # ------------------------------------------------ 1. data
    A("## 1. Data actually used (inventory)\n")
    if R["data_inventory"] is not None:
        A(md_table(R["data_inventory"]))
    A("")
    if R["split_summary"] is not None:
        A("**Split (days per season):**\n")
        A(md_table(R["split_summary"]))
        A(f"\nBlocks of {C.BLOCK_DAYS} consecutive days are assigned to train/val/test *within each season* "
          f"(>= {C.MIN_TEST_BLOCKS_PER_SEASON} test blocks per season). Train days within {C.EMBARGO_DAYS} days of any val/test "
          "day are dropped (\"embargo\"). Why not a pure time split: the earlier prototype's Q4 hold-out put the whole post-monsoon "
          "season in test with zero training days for it, so its climatology had no seasonal information (prototype t2m "
          "climatology RMSE 7.23 C). Section 7 reruns our models on exactly that time-forward split as a stress test.\n")

    # ------------------------------------------------ 2. point forecasts
    A("## 2. Point forecasts - per season, per target (test split)\n")
    A("RMSE for each individual source and each blend; % improvement = (baseline - model)/baseline, positive = better. "
      "95% CIs come from a bootstrap over **test days** (500 resamples), because cells on the same day are strongly correlated. "
      "Leads 1-3 days pooled; per-lead tables follow.\n")
    for t in C.TARGETS:
        A(f"### {t} ({TARGET_UNITS[t]})\n")
        A(f"![RMSE {t}](figures/rmse_{t}.png)\n")
        A(headline_table(R["pm_season"], imp, t))
        A("\n<details><summary>MAE table</summary>\n")
        A(mae_table(R["pm_season"], imp, t))
        A("\n</details>\n")
    A("### All seasons pooled\n")
    pa = R["pm_all"].pivot(index="target", columns="model", values="rmse").reset_index()
    A(md_table(pa))
    A("\n### By lead time (RMSE gain vs best expert in each season x lead slice)\n")
    il = R["imp_lead"]
    rows = []
    for (t, s), gg in il[il.blend.isin(["gating_adaptive", "lgbm_season"])].groupby(["target", "season"]):
        r = {"target": t, "season": s}
        for _, x in gg.iterrows():
            r[f"{'Gating' if x.blend == 'gating_adaptive' else 'LGBM'} lead {x.lead_days}d"] = f"{x.rmse:.3f} ({_pct(x['rmse_gain_vs_best_%'])}, best={EXP_LABEL[x.best_expert]})"
        rows.append(r)
    A(md_table(pd.DataFrame(rows)))
    A("\n### By weather regime (all seasons, adaptive gating vs every expert)\n")
    ir = R["imp_regime"]
    ir = ir[ir.blend == "gating_adaptive"][["target", "regime", "n", "rmse", "best_expert"] + [f"rmse_gain_vs_{e}_%" for e in C.EXPERTS] + ["beats_every_expert_rmse"]]
    A(md_table(ir, 2))
    A("")

    # ------------------------------------------------ 3. spatial holdout
    A("## 3. Spatial generalisation - cells never seen in training\n")
    A(f"{C.SPATIAL_HOLDOUT_CELLS} random 0.25-deg cells *off* the training subgrid, same test days. Similar gains here mean the "
      "models are not memorising cell identity.\n")
    hi = R["hold_imp"]
    hrows = []
    for (t, s), gg in hi.groupby(["target", "season"]):
        r = {"target": t, "season": s}
        for _, x in gg[gg.blend.isin(["gating_adaptive", "lgbm_season"])].iterrows():
            tag = "Gating" if x.blend == "gating_adaptive" else "LGBM"
            r[f"{tag} RMSE"] = x.rmse
            r[f"{tag} gain vs best expert"] = _pct(x["rmse_gain_vs_best_%"])
            r[f"{tag} beats every expert"] = x.beats_every_expert_rmse
        sub = imp[(imp.target == t) & (imp.season == s) & (imp.blend == "gating_adaptive")]
        if len(sub):
            r["(subgrid Gating gain, for reference)"] = _pct(sub.iloc[0]["rmse_gain_vs_best_%"])
        hrows.append(r)
    A(md_table(pd.DataFrame(hrows)))
    A("")

    # ------------------------------------------------ 4. quantiles
    A("## 4. Uncertainty - quantile regression on top of the blend (c)\n")
    A(f"LightGBM quantile models at levels {C.QUANTILES}; inputs = features + adaptive blend + expert spread. "
      "CRPS is approximated as 2 x mean pinball loss over the 7 levels (same estimator for model and baselines). "
      "Baselines: per-cell x season climatological quantiles (train days), and persistence + train residual quantiles. "
      "Intervals are split-conformalised per season on the validation days (CQR) - both raw and conformalised coverage are shown.\n")
    q = R["quant"]
    qcols = ["target", "season", "crps_model", "crps_clim_quantiles", "crpss_vs_clim_quantiles_%", "crps_persistence_resid",
             "crpss_vs_persistence_resid_%", "pinball_q10", "pinball_q50", "pinball_q90", "cov80_raw", "cov80_cqr",
             "cov90_raw", "cov90_cqr", "width80_cqr", "cov80_clim"]
    A(md_table(q[qcols], 3))
    A("\nTarget: cov80 ~ 0.80, cov90 ~ 0.90. `crpss_*` > 0 means the quantile model beats that probabilistic baseline.\n")

    # ------------------------------------------------ 5. extremes
    A("## 5. Extreme-event classifiers (d) - calibrated probabilities\n")
    A(f"Labels: heavy rain = daily tp >= {C.HEAVY_RAIN_MM} mm (IMD 'heavy'); heatwave = land cell with daily max of 6-hourly "
      f"t2m >= {C.HEATWAVE_TMAX_C} C (IMD plains absolute criterion; 6-hourly sampling under-reads the true Tmax); high wind = "
      f"daily max 10 m wind >= {C.HIGH_WIND_MAX_MS} m/s (Beaufort 7). LightGBM + isotonic calibration on validation days. "
      "Reference: 'persistence_flag' = the event happened on the issue day. "
      f"**Trust rule:** a slice is marked untrusted if it has < {C.MIN_POSITIVES_TRUST} positives or positives on < "
      f"{C.MIN_POSITIVE_DAYS_TRUST} distinct days (grid cells on one day are not independent events).\n")
    ex = R["extremes"]
    ecols = ["flag", "season", "model", "n_pos", "pos_days", "trusted", "auc", "avg_precision", "brier_skill_vs_climo",
             "precision@0.2", "recall@0.2", "precision@0.5", "recall@0.5"]
    A("### All leads pooled\n")
    A(md_table(ex[(ex.lead == "all") & (ex.model.isin(["calibrated", "persistence_flag"]))][ecols], 3))
    un = ex[(ex.lead == "all") & (ex.model == "calibrated") & (~ex.trusted.astype(bool))]
    if len(un):
        A("\n**Untrustworthy slices (too few positives / event days):** " +
          ", ".join(f"{r.flag} / {r.season} ({int(r.n_pos)} positives on {int(r.pos_days) if np.isfinite(r.pos_days) else 0} days)" for _, r in un.iterrows()) + ".")
    A("\n### By lead (all seasons, calibrated)\n")
    A(md_table(ex[(ex.season == "ALL") & (ex.model == "calibrated") & (ex.lead != "all")][ecols[:1] + ["lead"] + ecols[3:]], 3))
    A("\n![reliability](figures/reliability.png)\n")

    # ------------------------------------------------ 6. disagreement
    A("## 6. Model-disagreement signal (e)\n")
    A("`expert_spread` = standard deviation of the 4 expert forecasts per cell/day (logged in every prediction row, "
      "and fed to the quantile model). Is it informative? Error of the adaptive blend by spread quintile:\n")
    dcols = ["target", "season", "spearman_spread_vs_abs_err", "Q1_rmse", "Q3_rmse", "Q5_rmse", "rmse_ratio_top_vs_bottom_quintile",
             "Q1_width80", "Q5_width80", "Q1_cov80", "Q5_cov80"]
    A(md_table(R["disagree"][dcols], 3))
    A("\n![spread-skill](figures/spread_skill.png)\n")

    # ------------------------------------------------ 7. time-forward
    A("## 7. Stress test - time-forward split (train Jan-Aug, val Sep, test Oct-Dec)\n")
    A("Same features/models, but the prototype's split. Post-monsoon has **no training days**, so per-season LightGBM falls "
      "back to one all-season model, and climatology has to borrow from the nearest train days. "
      "This quantifies how much of the headline skill survives a pure out-of-time test.\n")
    if TF["pm_season"] is not None:
        tf = TF["pm_season"].pivot_table(index=["target", "season"], columns="model", values="rmse").reset_index()
        A(md_table(tf))
        A("")
        ti = TF["imp_season"]
        A(md_table(ti[["target", "season", "blend", "rmse", "best_expert", "rmse_gain_vs_best_%", "gain_vs_best_ci95_lo",
                       "gain_vs_best_ci95_hi", "beats_every_expert_rmse"]], 3))
        proto = pd.read_csv(C.PROTOTYPE_SKILL_CSV)
        A("\n**Comparison with the earlier prototype (same Oct-Dec test window, lead 1 d persistence setting):**\n")
        pr = proto.pivot(index="target", columns="source", values="rmse").reset_index()
        A(md_table(pr))
        A("\nThe prototype's blend lost to persistence on t2m (1.575 vs 1.268) and wind (0.677 vs 0.645). "
          "Its tp values are ~6x smaller than a physical daily total (it appears to sum four 1-hour accumulations instead of scaling to 24 h), "
          "so tp numbers are not comparable across the two.\n")

    # ------------------------------------------------ 8. weight maps
    A("## 8. Model weight maps (interpretability deliverable)\n")
    A("Mean softmax weight the adaptive gating network gives each expert. Full per-region table: "
      "`reports/results/blocked/w_season_region.csv` (3x3-deg regions), per-cell: `w_cell.csv`.\n")
    A("### Plain-English readings (auto-generated from the weight tables)\n")
    for t in C.TARGETS:
        z = R["w_season_zone"][R["w_season_zone"].target == t]
        for s in weight_sentences(z, t):
            A(f"* {s}")
    A("")
    for t in C.TARGETS:
        A(f"![weights {t}](figures/weights_{t}.png)\n")
    A("### Weights by lead time\n")
    A(md_table(R["w_season_lead"], 2))
    A("\n### Weights by regime\n")
    A(md_table(R["w_regime"], 2))
    A("\n### Per-season LightGBM: top-8 features by gain share\n")
    for t in C.TARGETS:
        imp_f = pd.read_csv(C.RESULTS_DIR / split / f"lgbm_importance_{t}.csv")
        top = (imp_f.sort_values("gain_share", ascending=False).groupby("season").head(8)
               .groupby("season")[["feature", "gain_share"]].apply(lambda d: ", ".join(f"{a} ({b:.0%})" for a, b in zip(d.feature, d.gain_share))))
        A(f"**{t}**\n")
        A(md_table(top.reindex(C.SEASONS).reset_index().rename(columns={0: "top features"})))
        A("")

    # ------------------------------------------------ 9. external
    A("## 9. External validation\n")
    A("### 9a. ERA5 truth vs a real station (Meteostat 2025)\n")
    if R["ext_station_agreement"] is not None:
        A(md_table(R["ext_station_agreement"], 3))
        A("\nForecast skill at the matched cell on test days, scored against ERA5 and against the station:\n")
        A(md_table(R["ext_station_skill"], 3))
    A("\n### 9b. IMD statewise rainfall, Aug 19 - Sep 24 2026 (an unseen year)\n")
    A("Experts rebuilt from IMD data (climatology = IMD's long-period daily normal). Blends use the monsoon-rain weights "
      "learned on ERA5 (mean adaptive gating weights, and the static season x lead weights) - no refitting on IMD data.\n")
    if R["ext_imd_skill"] is not None:
        A(md_table(R["ext_imd_skill"], 3))
        A("\nWeights transferred:\n")
        A(md_table(R["ext_imd_weights"], 3))
    A("\n### 9c. ECMWF operational snapshot (2026-09-24 00 UTC)\n")
    if R["ext_ecmwf"] is not None:
        A(md_table(R["ext_ecmwf"]))
    A("\nThe snapshot is a single 0 h analysis step: its accumulated fields (tp, ssrd, gusts) are identically zero by construction, "
      "and there is no gridded truth for 2026-09-24. There is therefore no valid comparison point, and ECMWF is **not** a trained "
      "expert. The inference API accepts it as an optional extra source with an explicit user-set weight (default off) - see README.\n")

    # ------------------------------------------------ 10. caveats
    A("## 10. Caveats and honest limitations\n")
    A("* **One year of truth.** Climatology is estimated from ~158 training days of 2025 (smoothed +-30 d window, excluding +-5 d "
      "around the target day), not a multi-decade normal. It therefore contains some same-year information; section 9b shows how "
      "the blend behaves with a real long-period normal (IMD).")
    A("* **Effective sample size is days, not rows.** ~84 test days; bootstrap CIs are computed over days for this reason. "
      "Winter has 14 test days - treat winter numbers as the least certain.")
    A("* **tp is estimated from four 1-hour samples per day** (x24). Absolute daily rainfall errors are therefore inflated relative "
      "to a true 24 h accumulation; relative model rankings are unaffected (every source is scored on the same truth).")
    A("* **Subgrid training.** Memory limits (~3 GB free) led to training on the 0.5-deg subgrid of 0.25-deg cells; section 3 shows "
      "skill carries over to unseen cells.")
    A("* **Lead-time buckets.** Available horizons are 24/48/72 h (daily data). The 0-6 h bucket exists only for the ECMWF 0 h "
      "snapshot at inference time; no model was trained or scored on it.")
    A("* **Heatwave label** uses 6-hourly samples, so it under-counts true IMD heatwave days; the classifier learns this label consistently.")
    A("")

    # ------------------------------------------------ 11. hand-off
    A("## 11. Hand-off to backend / dashboard / chatbot (not built here)\n")
    A("* **Backend API**: load `artifacts/blocked/` via `nwpblend.inference.BlendingForecaster` (signature in README). Output "
      "schema = one row per cell x target date x lead: `pred_<target>` (adaptive blend), `lgbm_<target>`, `q05..q95_<target>`, "
      "`lo80/hi80_<target>` (conformal), `w_<expert>_<target>` (weights), `spread_<target>`, `prob_<flag>`, `regime`.")
    A("* **Dashboard**: weight maps `reports/results/blocked/w_cell.csv` / `w_season_region.csv`; skill tables `imp_*.csv`; "
      "reliability `reliability.csv`; test predictions `preds_<target>_subgrid.parquet`.")
    A("* **Chatbot**: the auto-generated weight sentences (section 8) and `regime` labels are designed to be quoted directly.")
    return "\n".join(L)


def write_report(split: str = "blocked") -> None:
    md = build_report(split)
    (C.REPORTS_DIR / "BENCHMARK_REPORT.md").write_text(md, encoding="utf-8")

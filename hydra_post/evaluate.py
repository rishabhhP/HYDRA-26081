"""Full before/after evaluation against every point in HYDRA_V3_DIAGNOSTIC.md.

    python -m hydra_post.evaluate            -> reports/HYDRA_V3_1_POSTPROCESS_EVALUATION.md + runtime/hydra_v3_1_evaluation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from . import rolling as R

ROOT = Path(__file__).resolve().parents[1]
PROB_TARGETS = {"state_10": ("y", 10.0, "pa_20"), "state_20": ("y", 20.0, "pa_20"), "state_64.5": ("y", 64.5, "pa_64.5"),
                "local_64.5": ("y_local_max", 64.5, "pm_64.5")}


def r(x, d=3):
    if isinstance(x, dict):
        return {k: r(v, d) for k, v in x.items()}
    if isinstance(x, list):
        return [r(v, d) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), d)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def state_table(res, col):
    out = {}
    for st, g in res.groupby("state"):
        m = M.point(g["y"], g[col])
        out[st] = {"n": len(g), "mae": m["mae"], "bias": m["bias"], "recall_20": m["recall_20"], "events_20": m["events_20"],
                   "mae_skill_persistence": M.skill(g["y"], g[col], g["e_persistence"])["mae_skill"],
                   "mae_skill_climatology": M.skill(g["y"], g[col], g["e_climatology"])["mae_skill"]}
    return out


def main(replay: Path | None = None, lead: int = 1, truth: str = "era5", imd_csv: Path | None = None) -> dict:
    df, payload = D.load(replay or D.REPLAY, lead, truth, imd_csv)
    print("rolling-origin evaluation (5 monthly folds)...", flush=True)
    res, log = R.run(df)
    y = res["y"].to_numpy()
    report = {"design": {"test_period": [str(res["date"].min().date()), str(res["date"].max().date())], "test_state_days": len(res),
                         "folds": log["folds"], "history_selection": log["history_selection"], "gate_params": log["gate_params"],
                         "note": "Every number is out-of-sample: each month is predicted by models fitted only on earlier months (1-day embargo). "
                                 "July 2025 is training-only, so the comparison window is Aug-Dec 2025 for both raw HYDRA and the fixes."}}
    amounts = {"raw HYDRA v3": "hydra", "persistence": "e_persistence", "equal-weight experts": "equal_avg", "skill gate": "gate",
               "log1p extreme head": "amt_log_extreme", "Poisson head": "amt_poisson", "heavy/non-heavy mixture": "amt_mixture",
               "gate + mixture": "ens_gate_mixture", "FINAL (history-selected)": "final_amount"}
    report["amount"] = {}
    for name, col in amounts.items():
        m = M.point(y, res[col])
        m.update({f"skill_vs_persistence_{k}": v for k, v in M.skill(y, res[col], res["e_persistence"]).items()})
        m.update({f"skill_vs_climatology_{k}": v for k, v in M.skill(y, res[col], res["e_climatology"]).items()})
        report["amount"][name] = m
    print("bootstrapping confidence intervals...", flush=True)
    mae_d = lambda d: np.abs(d["final_amount"] - d["y"]).mean() - np.abs(d["hydra"] - d["y"]).mean()  # noqa: E731
    rmse_d = lambda d: np.sqrt(((d["final_amount"] - d["y"]) ** 2).mean()) - np.sqrt(((d["hydra"] - d["y"]) ** 2).mean())  # noqa: E731
    skill_p = lambda d: 1 - np.abs(d["final_amount"] - d["y"]).mean() / np.abs(d["e_persistence"] - d["y"]).mean()  # noqa: E731
    report["confidence_intervals_95"] = report.get("confidence_intervals_95", {}) | {"final_minus_raw_mae": M.bootstrap(res, mae_d), "final_minus_raw_rmse": M.bootstrap(res, rmse_d),
                                         "final_mae_skill_vs_persistence": M.bootstrap(res, skill_p)}
    report["interval"] = {"raw HYDRA v3 (symmetric conformal)": M.interval(y, res["lo80"], res["hi80"]),
                          "asymmetric split conformal": M.interval(y, res["lo_asym"], res["hi_asym"])}
    if "lo_aci" in res:
        report["interval"]["adaptive conformal (ACI)"] = M.interval(y, res["lo_aci"], res["hi_aci"])
        report["interval_final"] = "adaptive conformal (ACI)"
        report.setdefault("confidence_intervals_95", {})
    else:
        report["interval_final"] = "asymmetric split conformal"
    if "lo_aci" in res:
        report["confidence_intervals_95"]["aci_coverage"] = M.bootstrap(res, lambda d: ((d["y"] >= d["lo_aci"]) & (d["y"] <= d["hi_aci"])).mean())
    report["confidence_intervals_95"]["asym_coverage"] = M.bootstrap(res, lambda d: ((d["y"] >= d["lo_asym"]) & (d["y"] <= d["hi_asym"])).mean())
    report["confidence_intervals_95"]["asym_above"] = M.bootstrap(res, lambda d: (d["y"] > d["hi_asym"]).mean())
    report["probability"] = {}
    for name, (col, thr, native) in PROB_TARGETS.items():
        obs = (res[col] >= thr).to_numpy()
        entry = {"raw HYDRA head (" + native + ")": M.prob(obs, res[native].fillna(0)), "FINAL calibrated": M.prob(obs, res[f"final_p_{name}"])}
        entry["tiers_final"] = M.tiers(obs, res[f"final_p_{name}"])
        entry["point_forecast_detection"] = M.contingency(res[col].to_numpy() >= thr, (res["hydra"] if col == "y" else res["hydra"]).to_numpy() >= thr) if col == "y" else None
        report["probability"][name] = entry
    report["states"] = {"raw": state_table(res, "hydra"), "final": state_table(res, "final_amount")}
    report["states_nonpositive_mae_skill_vs_persistence"] = {k: sum(1 for v in report["states"][k].values() if v["mae_skill_persistence"] <= 0) for k in ("raw", "final")}
    report["states_nonpositive_mae_skill_vs_climatology"] = {k: sum(1 for v in report["states"][k].values() if v["mae_skill_climatology"] <= 0) for k in ("raw", "final")}
    raw_up = res.assign(a=res["y"] > res["hi80"]).groupby("state")["a"].mean()
    new_up = res.assign(a=res["y"] > res["hi_asym"]).groupby("state")["a"].mean()
    report["upper_miss_by_state"] = {st: {"raw": raw_up[st], "asym": new_up[st]} for st in raw_up.sort_values(ascending=False).index[:8]}
    w_cols = [c for c in df.columns if c.startswith("w_")]
    test = df[df["date"] >= res["date"].min()]
    report["gate"] = {"v3_trained_gate_mean_weight": test[w_cols].mean().rename(lambda c: c[2:]).to_dict(), "skill_gate_mean_weight": log["gate_weight_mean"],
                      "skill_gate_mean_max_weight": log["gate_max_weight_share"]}
    report["occurrence_definition"] = {
        "p_rain>=0.5 vs obs>=1": M.contingency(y >= 1, res.merge(df[["state", "date", "p_rain"]], on=["state", "date"])["p_rain"].to_numpy() >= 0.5),
        "hydra>=1 vs obs>=1": M.contingency(y >= 1, res["hydra"].to_numpy() >= 1), "final>=1 vs obs>=1": M.contingency(y >= 1, res["final_amount"].to_numpy() >= 1)}
    out = r(report)
    (ROOT / "runtime").mkdir(exist_ok=True)
    suffix = ("" if lead == 1 else f"_lead{lead}") + ("" if truth == "era5" else f"_{truth}")
    out["design"]["lead_days"] = lead
    out["design"]["truth"] = truth
    (ROOT / "runtime" / f"hydra_v3_1_evaluation{suffix}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    res.to_csv(ROOT / "runtime" / f"hydra_v3_1_oos_predictions{suffix}.csv", index=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"HYDRA_V3_1_POSTPROCESS_EVALUATION{suffix.upper()}.md").write_text(markdown(out), encoding="utf-8")
    return out


def markdown(o: dict) -> str:
    A = o["amount"]
    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
    pct = lambda v, d=1: "n/a" if v is None else f"{v * 100:.{d}f}%"  # noqa: E731
    L = ["# HYDRA v3.1 post-processing: out-of-sample evaluation", "",
         f"Test window {o['design']['test_period'][0]} to {o['design']['test_period'][1]}, {o['design']['test_state_days']} state-days. {o['design']['note']}", "",
         "## Amount forecast", "", "| Model | MAE | RMSE | Bias | ≥10 recall | ≥20 recall | ≥20 CSI | Bias 20-64.5 | Bias ≥64.5 | Top-5 ratio | MAE skill vs persistence | RMSE skill vs persistence |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in A.items():
        L.append(f"| {name} | {f(m['mae'])} | {f(m['rmse'])} | {f(m['bias'])} | {pct(m['recall_10'])} | {pct(m['recall_20'])} | {f(m['csi_20'], 3)} | "
                 f"{f(m['bias[20-64.5]'], 1)} | {f(m['bias[>=64.5]'], 1)} | {f(m['top5_ratio'])} | {pct(m['skill_vs_persistence_mae_skill'])} | {pct(m['skill_vs_persistence_rmse_skill'])} |")
    ci = o["confidence_intervals_95"]
    L += ["", f"95% day-block bootstrap: final − raw MAE {f(ci['final_minus_raw_mae'][0])} to {f(ci['final_minus_raw_mae'][1])} mm/day; "
              f"final − raw RMSE {f(ci['final_minus_raw_rmse'][0])} to {f(ci['final_minus_raw_rmse'][1])}; final MAE skill vs persistence "
              f"{pct(ci['final_mae_skill_vs_persistence'][0])} to {pct(ci['final_mae_skill_vs_persistence'][1])}.", "",
          f"States with non-positive MAE skill vs persistence: raw {o['states_nonpositive_mae_skill_vs_persistence']['raw']}/36, final {o['states_nonpositive_mae_skill_vs_persistence']['final']}/36. "
          f"Vs climatology: raw {o['states_nonpositive_mae_skill_vs_climatology']['raw']}/36, final {o['states_nonpositive_mae_skill_vs_climatology']['final']}/36.", "",
          "## 80% interval", "", "| Interval | Coverage | Above upper | Below lower | Mean width |", "|---|---|---|---|---|"]
    for name, m in o["interval"].items():
        L.append(f"| {name} | {pct(m['coverage'])} | {pct(m['above'])} | {pct(m['below'])} | {f(m['width'], 1)} mm |")
    if "aci_coverage" in ci:
        L += ["", f"Adaptive (published) coverage 95% CI {pct(ci['aci_coverage'][0])} to {pct(ci['aci_coverage'][1])}."]
    L += ["", f"Asymmetric coverage 95% CI {pct(ci['asym_coverage'][0])} to {pct(ci['asym_coverage'][1])}; above-upper CI {pct(ci['asym_above'][0])} to {pct(ci['asym_above'][1])}.", "",
          "## Heavy-rain probabilities", "", "| Event | Events | Source | Brier | Brier skill | AUC |", "|---|---|---|---|---|---|"]
    for name, e in o["probability"].items():
        for src, m in e.items():
            if isinstance(m, dict) and "brier" in m:
                L.append(f"| {name} | {m['events']} | {src} | {f(m['brier'], 4)} | {f(m['bss'], 3)} | {f(m['auc'], 3)} |")
    L += ["", "### Alert tiers (final calibrated probabilities)", "", "| Event | Tier | P ≥ | Issued | Hit rate | Events captured |", "|---|---|---|---|---|---|"]
    for name, e in o["probability"].items():
        for t in e["tiers_final"]:
            L.append(f"| {name} | {t['tier']} | {t['min_prob']} | {t['issued']} | {pct(t['hit_rate'])} | {pct(t['events_captured'])} |")
    G = o["gate"]
    L += ["", "## Gate allocation", "", "| Expert | v3 trained gate | Skill-aware gate |", "|---|---|---|"]
    for k, v in G["v3_trained_gate_mean_weight"].items():
        L.append(f"| {k} | {pct(v)} | {pct(G['skill_gate_mean_weight'].get(k))} |")
    L += [f"| hydra blend (as one input) | n/a | {pct(G['skill_gate_mean_weight'].get('hydra'))} |", "", "## Model choice per month (by earlier out-of-sample record only)", ""]
    for s in o["design"]["history_selection"]:
        L.append(f"- {s['month']}: amount `{s['amount']}`; probabilities {', '.join(f'{k}: `{v}`' for k, v in s['probabilities'].items())}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--truth", choices=["era5", "imd"], default="era5")
    ap.add_argument("--imd-csv", type=Path)
    a = ap.parse_args()
    main(a.replay, a.lead, a.truth, a.imd_csv)

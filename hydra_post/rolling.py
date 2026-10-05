"""Rolling-origin evaluation: train on every month before the test month, test on that month, repeat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .aci import AdaptiveInterval
from .models import AmountHead, AsymmetricInterval, NativeCalibration, ProbabilityHeads, skill_gate, split_time

CRITERION = "mae+rmse"   # declared before looking at any test month


def score(y, p):
    e = np.asarray(p) - np.asarray(y)
    return np.abs(e).mean() + np.sqrt((e ** 2).mean())


def select_amount(tr: pd.DataFrame, cols, gate_pred_tr: np.ndarray) -> tuple[str, dict]:
    """Nested choice of the amount forecast using only the training window (inner time split)."""
    inner_fit, inner_val = split_time(tr, 0.25)
    scores = {"raw_hydra": score(inner_val["y"], inner_val["hydra"]),
              "skill_gate": score(inner_val["y"], gate_pred_tr[tr.index.get_indexer(inner_val.index)])}
    for v in AmountHead.VARIANTS:
        try:
            scores[v] = score(inner_val["y"], AmountHead(cols, v).fit(inner_fit).predict(inner_val))
        except Exception:
            scores[v] = np.inf
    return min(scores, key=scores.get), scores


def tune_gate(df: pd.DataFrame, experts, months_before: pd.Timestamp) -> dict:
    """Pick Hedge eta/decay on data before the first test month only."""
    best, best_s = None, np.inf
    hist = df[df["date"] < months_before]
    for eta in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        for decay in (0.8, 0.9, 0.95, 0.99):
            p, _ = skill_gate(hist, experts, eta, decay)
            s = score(hist["y"], p)
            if s < best_s:
                best, best_s = {"eta": eta, "decay": decay}, s
    return best


def run(df: pd.DataFrame, test_months=("2025-08", "2025-09", "2025-10", "2025-11", "2025-12")) -> tuple[pd.DataFrame, dict]:
    experts = D.experts(df)
    first = pd.Timestamp(test_months[0] + "-01")
    gate_params = tune_gate(df, experts, first)
    gate_pred, gate_w = skill_gate(df, experts, **gate_params)
    df = df.assign(gate=gate_pred)
    out, log = [], {"gate_params": gate_params, "folds": []}
    aci_state = None
    for month in test_months:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthEnd(0)
        tr = df[df["date"] < start - pd.Timedelta(days=1)]          # 1-day embargo
        te = df[(df["date"] >= start) & (df["date"] <= end)]
        tr = D.add_state_climate(tr, tr)
        te = D.add_state_climate(tr, te)
        cols = [c for c in D.feature_columns(df) if c in tr]
        choice, inner = select_amount(tr, cols, tr["gate"].to_numpy())
        fold = te[["state", "date", "y", "y_local_max", "hydra", "lo80", "hi80", "e_persistence", "e_climatology", "gate"]].copy()
        for v in AmountHead.VARIANTS:
            fold[f"amt_{v}"] = AmountHead(cols, v).fit(tr).predict(te)
        fold["amt_selected"] = fold["hydra"] if choice == "raw_hydra" else fold["gate"] if choice == "skill_gate" else fold[f"amt_{choice}"]
        probs = ProbabilityHeads(cols).fit(tr)
        for name, p in probs.predict(te).items():
            fold[f"p_{name}"] = p
        for name, d in NativeCalibration().fit(tr).predict(te).items():
            fold[f"iso_{name}"], fold[f"stack_{name}"] = d["iso"], d["stack"]
        fold["equal_avg"] = te[[f"e_{e}" for e in experts]].mean(axis=1).to_numpy()
        fold["ens_gate_mixture"] = 0.5 * fold["gate"] + 0.5 * fold["amt_mixture"]
        fold["ens_gate_log"] = 0.5 * fold["gate"] + 0.5 * fold["amt_log_extreme"]
        lo, hi = AsymmetricInterval(cols).fit(tr).predict(te)
        fold["lo_asym"], fold["hi_asym"] = lo, hi
        aci = AdaptiveInterval(cols).fit(tr, aci_state)
        lo2, hi2, trace = aci.run(te)
        aci_state = aci.state()
        fold["lo_aci"], fold["hi_aci"] = lo2, hi2
        # v3.4: the adaptive interval is the published range (declared before any test-period scoring)
        fold["lo_final"], fold["hi_final"] = lo2, hi2
        fold["pa_20"] = te["pa_20"].to_numpy()
        fold["pa_64.5"] = te["pa_64.5"].to_numpy()
        fold["pm_64.5"] = te["pm_64.5"].to_numpy()
        fold["fold"] = month
        out.append(fold)
        log["folds"].append({"test_month": month, "train_days": int(tr["date"].nunique()), "selected_amount": choice,
                             "aci_levels_end": {k: round(v, 4) for k, v in aci_state.items()},
                             "inner_scores": {k: round(float(v), 3) for k, v in inner.items()}, "probability_models": probs.kind})
        print(f"  fold {month}: trained on {tr['date'].nunique()} days, amount -> {choice}", flush=True)
    res = pd.concat(out).sort_values(["state", "date"])
    res = select_by_history(res, log)
    log["gate_weight_mean"] = dict(zip(experts + ["hydra"], np.nanmean(gate_w[df["date"] >= first], 0).round(3).tolist()))
    log["gate_max_weight_share"] = float(np.nanmean(np.nanmax(gate_w[df["date"] >= first], 1)))
    return res, log


AMOUNT_CANDIDATES = ("hydra", "equal_avg", "gate", "amt_log_extreme", "amt_poisson", "amt_mixture", "ens_gate_mixture", "ens_gate_log")
PROB_CANDIDATES = ("p_{}", "iso_{}", "stack_{}")


def select_by_history(res: pd.DataFrame, log: dict) -> pd.DataFrame:
    """For each test month, use the candidate with the best score on earlier test months only.

    The first test month has no out-of-sample history, so it falls back to the pre-declared defaults:
    raw HYDRA for the amount and the isotonic-calibrated native head for probabilities.
    """
    res = res.copy()
    months = sorted(res["fold"].unique())
    res["final_amount"] = np.nan
    targets = {"state_10": ("y", 10.0), "state_20": ("y", 20.0), "state_64.5": ("y", 64.5), "local_64.5": ("y_local_max", 64.5)}
    for name in targets:
        res[f"final_p_{name}"] = np.nan
    log["history_selection"] = []
    for k, month in enumerate(months):
        past = res[res["fold"].isin(months[:k])]
        cur = res["fold"] == month
        if past.empty:
            amount, pick = "hydra", {n: "iso_{}" for n in targets}
        else:
            amount = min(AMOUNT_CANDIDATES, key=lambda c: score(past["y"], past[c]))
            pick = {}
            for n, (col, thr) in targets.items():
                obs = (past[col] >= thr).to_numpy()
                pick[n] = min(PROB_CANDIDATES, key=lambda c: float(((past[c.format(n)] - obs) ** 2).mean()))
        res.loc[cur, "final_amount"] = res.loc[cur, amount]
        for n in targets:
            res.loc[cur, f"final_p_{n}"] = res.loc[cur, pick[n].format(n)]
        log["history_selection"].append({"month": month, "amount": amount, "probabilities": {n: pick[n].format(n) for n in targets}})
    return res

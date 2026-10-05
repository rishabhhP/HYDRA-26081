"""Frozen-candidate harness: test a promising HYDRA configuration without letting the test data choose it.

A candidate whose settings were picked by looking at a test period will look better on that period than it will in
use. This harness only lets a candidate be scored on data that played no part in choosing it:

    python -m hydra_post.candidates freeze candidates/my_candidate.json
        writes the settings with a timestamp and a SHA-256 fingerprint; any later edit is detected and refused

    python -m hydra_post.candidates holdout candidates/my_candidate.json --start 2024-06-01 --end 2024-09-30
        fixed settings, scored on a period that must not overlap the spec's "chosen_on" dates; production is scored
        on the same days for comparison (models always trained only on days before each holdout month)

    python -m hydra_post.candidates nested candidates/my_grid.json
        tests the selection procedure instead of the chosen values: each month picks among the spec's "grid" of
        variants using only earlier months, exactly like production's history-based selection

Promotion rule (printed in the report): candidate MAE and RMSE better than production on the independent data,
the MAE improvement's 95% day-block bootstrap interval excludes zero, and the release-gate interval and calibration
checks pass for the candidate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .aci import AdaptiveInterval
from .models import AmountHead, AsymmetricInterval, NativeCalibration, ProbabilityHeads, skill_gate
from .rolling import score, tune_gate

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = {"name": "production", "description": "v3.4 production defaults",
              "amount": {"method": "blend", "blend_weight_gate": 0.5, "extreme_alpha": 3.0},
              "gate": {"eta": "auto", "decay": "auto", "cap": 0.5}, "probabilities": "stack",
              "interval": {"method": "aci", "gamma": 0.03, "buffer_days": 45}}
AMOUNT_METHODS = {"hydra", "gate", "equal_avg", "mixture", "log_extreme", "poisson", "blend"}
EVENTS = {"state_10": ("y", 10.0), "state_20": ("y", 20.0), "state_64.5": ("y", 64.5), "local_64.5": ("y_local_max", 64.5)}


# ------------------------------------------------------------------ freezing
def _fingerprint(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k not in ("sha256", "frozen_at")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def freeze(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec.get("sha256"):
        verify(spec)
        print(f"already frozen at {spec['frozen_at']}")
        return spec
    validate(spec)
    spec["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec["sha256"] = _fingerprint(spec)
    Path(path).write_text(json.dumps(spec, indent=1))
    print(f"frozen {spec['name']} at {spec['frozen_at']}, sha256 {spec['sha256'][:12]}")
    return spec


def verify(spec: dict) -> None:
    if not spec.get("sha256"):
        raise SystemExit("spec is not frozen; run: python -m hydra_post.candidates freeze <spec>")
    if _fingerprint(spec) != spec["sha256"]:
        raise SystemExit("spec was edited after freezing (fingerprint mismatch); freeze a new candidate instead")


def validate(spec: dict) -> None:
    for variant in spec.get("grid", [spec]):
        method = variant.get("amount", {}).get("method", "blend")
        if method not in AMOUNT_METHODS:
            raise SystemExit(f"amount.method must be one of {sorted(AMOUNT_METHODS)}")
        if variant.get("probabilities", "stack") not in ("stack", "iso", "gbm"):
            raise SystemExit("probabilities must be stack, iso or gbm")
        if variant.get("interval", {}).get("method", "aci") not in ("aci", "asym"):
            raise SystemExit("interval.method must be aci or asym")


def merged(variant: dict) -> dict:
    out = copy.deepcopy(PRODUCTION)
    for key in ("amount", "gate", "interval"):
        out[key].update(variant.get(key, {}))
    for key in ("name", "description", "probabilities"):
        if key in variant:
            out[key] = variant[key]
    return out


# ------------------------------------------------------------------ predictions with fixed settings
def predict_months(df: pd.DataFrame, variant: dict, months: list[str]) -> pd.DataFrame:
    """Monthly expanding refits; each month uses only rows dated before it (1-day embargo)."""
    v = merged(variant)
    experts = D.experts(df)
    first = pd.Timestamp(months[0] + "-01")
    g = v["gate"]
    if g["eta"] == "auto" or g["decay"] == "auto":
        tuned = tune_gate(df, experts, first)
        g = {**g, "eta": tuned["eta"] if g["eta"] == "auto" else g["eta"], "decay": tuned["decay"] if g["decay"] == "auto" else g["decay"]}
    gate_pred, _ = skill_gate(df, experts, eta=float(g["eta"]), decay=float(g["decay"]), cap=float(g["cap"]))
    df = df.assign(gate=gate_pred)
    frames, aci_state = [], None
    for month in months:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthEnd(0)
        tr = D.add_state_climate(df[df["date"] < start - pd.Timedelta(days=1)], df[df["date"] < start - pd.Timedelta(days=1)])
        te = D.add_state_climate(tr, df[(df["date"] >= start) & (df["date"] <= end)])
        if te.empty or tr["date"].nunique() < 20:
            continue
        cols = [c for c in D.feature_columns(df) if c in tr]
        a = v["amount"]
        method = a["method"]
        if method == "hydra":
            amount = te["hydra"].to_numpy()
        elif method == "gate":
            amount = te["gate"].to_numpy()
        elif method == "equal_avg":
            amount = te[[f"e_{e}" for e in experts]].mean(axis=1).to_numpy()
        elif method == "blend":
            mix = AmountHead(cols, "mixture", a.get("extreme_alpha", 3.0)).fit(tr).predict(te)
            w = float(a.get("blend_weight_gate", 0.5))
            amount = w * te["gate"].to_numpy() + (1 - w) * mix
        else:
            amount = AmountHead(cols, method, a.get("extreme_alpha", 3.0)).fit(tr).predict(te)
        fold = te[["state", "date", "y", "y_local_max", "hydra", "e_persistence", "e_climatology"]].copy()
        fold["amount"] = np.clip(amount, 0, None)
        if v["probabilities"] == "gbm":
            for name, p in ProbabilityHeads(cols).fit(tr).predict(te).items():
                fold[f"p_{name}"] = p
        else:
            for name, d in NativeCalibration().fit(tr).predict(te).items():
                fold[f"p_{name}"] = d[v["probabilities"]]
        iv = v["interval"]
        if iv["method"] == "aci":
            aci = AdaptiveInterval(cols, gamma=float(iv.get("gamma", 0.03)), buffer_days=int(iv.get("buffer_days", 45))).fit(tr, aci_state)
            lo, hi, _ = aci.run(te)
            aci_state = aci.state()
        else:
            lo, hi = AsymmetricInterval(cols).fit(tr).predict(te)
        fold["lo"], fold["hi"] = lo, hi
        fold["month"] = month
        frames.append(fold)
    if not frames:
        raise SystemExit("no month had enough earlier history to train on")
    return pd.concat(frames, ignore_index=True)


def scores(f: pd.DataFrame) -> dict:
    y = f["y"].to_numpy()
    pt = M.point(y, f["amount"])
    out = {"n": len(f), "days": int(f["date"].nunique()), "mae": pt["mae"], "rmse": pt["rmse"], "bias": pt["bias"],
           "recall_20": pt["recall_20"], "csi_20": pt["csi_20"], "top5_ratio": pt["top5_ratio"], "events_20": pt["events_20"],
           "mae_skill_persistence": M.skill(y, f["amount"], f["e_persistence"])["mae_skill"]}
    out.update({f"interval_{k}": v for k, v in M.interval(y, f["lo"], f["hi"]).items() if k != "nominal"})
    for name, (col, thr) in EVENTS.items():
        out[f"bss_{name}"] = M.prob(f[col].to_numpy() >= thr, f[f"p_{name}"].to_numpy())["bss"]
    return out


def compare(cand: pd.DataFrame, prod: pd.DataFrame, independent: bool) -> dict:
    key = ["state", "date"]
    both = cand.merge(prod, on=key, suffixes=("_c", "_p"))
    both = both.rename(columns={"y_c": "y"})
    diff = lambda d: np.abs(d["amount_c"] - d["y"]).mean() - np.abs(d["amount_p"] - d["y"]).mean()  # noqa: E731
    ci = M.bootstrap(both, diff)
    sc, sp = scores(cand), scores(prod)
    interval_ok = 0.75 <= sc["interval_coverage"] <= 0.85 and abs(sc["interval_above"] - sc["interval_below"]) <= 0.04
    calib_ok = sc["bss_state_10"] > 0 and sc["bss_state_20"] > 0
    better = sc["mae"] < sp["mae"] and sc["rmse"] < sp["rmse"] and ci[1] < 0
    verdict = ("PROMOTE" if better and interval_ok and calib_ok and independent else
               "NOT INDEPENDENT: result cannot justify promotion" if not independent else
               "KEEP PRODUCTION")
    return {"candidate": sc, "production": sp, "mae_difference_ci95": ci, "interval_ok": interval_ok, "calibration_ok": calib_ok,
            "independent": independent, "verdict": verdict}


def overlaps(spec: dict, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    chosen = spec.get("chosen_on")
    if not chosen:
        return True          # unknown: treat as overlapping, the honest default
    a, b = pd.Timestamp(chosen["start"]), pd.Timestamp(chosen["end"])
    return not (end < a or start > b)


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def run_holdout(spec: dict, start: str, end: str, truth: str = "era5", force: bool = False) -> dict:
    verify(spec)
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    independent = not overlaps(spec, a, b)
    if not independent and not force:
        raise SystemExit(f"holdout {start}..{end} overlaps the dates used to choose this candidate ({spec.get('chosen_on')}). "
                         "Pick a holdout outside them (for example a season from the multi-year replay), use 'nested' mode, "
                         "or pass --force to run anyway (the result is then marked NOT INDEPENDENT).")
    df, _ = D.load(truth=truth)
    if a < df["date"].min() + pd.Timedelta(days=30) or b > df["date"].max():
        raise SystemExit(f"holdout must lie inside the replay ({df['date'].min().date()} to {df['date'].max().date()}) with 30 days of history before it")
    months = months_between(a, b)
    cand = predict_months(df, spec, months)
    prod = predict_months(df, PRODUCTION, months)
    window = lambda f: f[(f["date"] >= a) & (f["date"] <= b)]  # noqa: E731
    return {"mode": "holdout", "holdout": [start, end], "truth": truth, **compare(window(cand), window(prod), independent)}


def run_nested(spec: dict, truth: str = "era5", months: list[str] | None = None) -> dict:
    verify(spec)
    grid = spec.get("grid")
    if not grid:
        raise SystemExit("nested mode needs a 'grid' list of variants in the spec")
    df, _ = D.load(truth=truth)
    months = months or months_between(df["date"].min() + pd.offsets.MonthBegin(1), df["date"].max())
    preds = {k: predict_months(df, variant, months) for k, variant in enumerate(grid)}
    picks, chosen = [], []
    for i, month in enumerate(sorted({m for p in preds.values() for m in p["month"]})):
        past = [m for m in sorted(preds[0]["month"].unique()) if m < month]
        if not past:
            k = 0                                                    # first month: the grid's first entry (declared default)
        else:
            k = min(preds, key=lambda j: score(preds[j][preds[j]["month"].isin(past)]["y"], preds[j][preds[j]["month"].isin(past)]["amount"]))
        picks.append(preds[k][preds[k]["month"] == month])
        chosen.append({"month": month, "variant": grid[k].get("name", f"variant {k}")})
    cand = pd.concat(picks, ignore_index=True)
    prod = predict_months(df, PRODUCTION, months)
    return {"mode": "nested", "truth": truth, "choices": chosen, **compare(cand, prod, independent=True)}


def report(result: dict, spec: dict) -> str:
    c, p = result["candidate"], result["production"]
    f = lambda v, d=3: "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"  # noqa: E731
    rows = [("MAE", "mae"), ("RMSE", "rmse"), ("Bias", "bias"), ("≥20 mm recall", "recall_20"), ("≥20 mm CSI", "csi_20"),
            ("Top-5 ratio", "top5_ratio"), ("MAE skill vs persistence", "mae_skill_persistence"), ("80% coverage", "interval_coverage"),
            ("Above / below", None), ("Brier skill ≥10 mm", "bss_state_10"), ("Brier skill ≥20 mm", "bss_state_20"),
            ("Brier skill ≥64.5 anywhere", "bss_local_64.5")]
    L = [f"# Candidate test: {spec['name']}", "", spec.get("description", ""), "",
         f"Mode: {result['mode']}" + (f", holdout {result['holdout'][0]} to {result['holdout'][1]}" if result["mode"] == "holdout" else "")
         + f"; truth {result['truth'].upper()}; frozen {spec.get('frozen_at')} (sha256 {spec.get('sha256', '')[:12]}).",
         f"Independent of the data used to choose the candidate: {'yes' if result['independent'] else 'NO'}.", "",
         f"**Verdict: {result['verdict']}**", "", "| Metric | Candidate | Production |", "|---|---|---|"]
    for label, key in rows:
        if key is None:
            L.append(f"| {label} | {c['interval_above']:.1%} / {c['interval_below']:.1%} | {p['interval_above']:.1%} / {p['interval_below']:.1%} |")
        else:
            L.append(f"| {label} | {f(c[key])} | {f(p[key])} |")
    ci = result["mae_difference_ci95"]
    L += ["", f"Candidate − production MAE, 95% day-block bootstrap: {ci[0]:+.3f} to {ci[1]:+.3f} mm/day "
              f"({c['n']:,} state-days, {c['days']} days, {c['events_20']} days ≥20 mm).", "",
          "Promotion needs: lower MAE and RMSE, an MAE interval entirely below zero, interval coverage 75-85% with balanced tails, "
          "positive Brier skill at 10 and 20 mm, and independent test data."]
    if result["mode"] == "nested":
        L += ["", "Variant chosen each month (from earlier months only):"] + [f"- {x['month']}: {x['variant']}" for x in result["choices"]]
    return "\n".join(L) + "\n"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("freeze")
    s1.add_argument("spec", type=Path)
    for name in ("holdout", "nested"):
        s = sub.add_parser(name)
        s.add_argument("spec", type=Path)
        s.add_argument("--truth", choices=["era5", "imd"], default="era5")
        if name == "holdout":
            s.add_argument("--start", required=True)
            s.add_argument("--end", required=True)
            s.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.spec)
        return
    spec = json.loads(a.spec.read_text())
    result = run_holdout(spec, a.start, a.end, a.truth, a.force) if a.cmd == "holdout" else run_nested(spec, a.truth)
    tag = f"{spec['name']}_{a.cmd}_{a.truth}"
    (ROOT / "runtime").mkdir(exist_ok=True)
    (ROOT / "runtime" / f"candidate_{tag}.json").write_text(json.dumps(result, indent=1, default=float))
    (ROOT / "reports").mkdir(exist_ok=True)
    md = report(result, spec)
    (ROOT / "reports" / f"CANDIDATE_{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

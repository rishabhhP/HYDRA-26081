"""Production post-processor: fit on all verified history, then enhance new HYDRA state-day forecasts.

    python -m hydra_post.apply      -> runtime/hydra_v3_1_postprocessed.json (+ model files in runtime/hydra_v3_1/)

Output per state-day: improved amount, asymmetric 80% interval, calibrated probabilities for >=10 / >=20 / >=64.5 mm
(state mean) and >=64.5 mm anywhere in the state, alert tier, the expected amount if heavy rain happens, and a
per-state reliability card with sample counts.
"""
from __future__ import annotations

import json
import pickle
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .aci import AdaptiveInterval
from .models import AmountHead, AsymmetricInterval, NativeCalibration, skill_gate
from .rolling import score, tune_gate

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runtime" / "hydra_v3_1"
TIERS = (("warning", 0.6), ("alert", 0.4), ("watch", 0.2))
EVENTS = {"state_10": "state-average rain ≥ 10 mm", "state_20": "state-average rain ≥ 20 mm",
          "state_64.5": "state-average rain ≥ 64.5 mm", "local_64.5": "≥ 64.5 mm somewhere in the state (any grid cell)"}


def tier(p: float) -> str | None:
    return next((name for name, cut in TIERS if p >= cut), None)


class PostProcessor:
    def fit(self, df: pd.DataFrame, evaluation: dict | None = None):
        self.experts = D.experts(df)
        self.gate_params = tune_gate(df, self.experts, df["date"].max() + pd.Timedelta(days=1))
        gate, _ = skill_gate(df, self.experts, **self.gate_params)
        df = df.assign(gate=gate)
        tr = D.add_state_climate(df, df)
        self.climate = tr.groupby("state")[["state_mean_train", "state_p90_train", "state_wetfrac_train"]].first()
        self.cols = [c for c in D.feature_columns(df) if c in tr]
        self.mixture = AmountHead(self.cols, "mixture").fit(tr)
        self.interval = AsymmetricInterval(self.cols).fit(tr)  # retained for comparison
        self.aci_template = AdaptiveInterval(self.cols).fit(tr)  # v3.4 published interval
        self.calib = NativeCalibration().fit(tr)
        # amount method: best out-of-sample record in the rolling evaluation when available
        self.amount_method = "ens_gate_mixture"
        if evaluation:
            names = {"skill gate": "gate", "equal-weight experts": "equal_avg", "gate + mixture": "ens_gate_mixture", "raw HYDRA v3": "hydra"}
            ranked = sorted(((v["mae"] + v["rmse"], names[k]) for k, v in evaluation["amount"].items() if k in names))
            self.amount_method = ranked[0][1]
        return self

    def predict(self, history_and_new: pd.DataFrame, only_from: pd.Timestamp | None = None) -> pd.DataFrame:
        """Pass the full replay (verified history + newest issue days). Gate weights use verified days only."""
        df = history_and_new.sort_values(["state", "date"]).reset_index(drop=True)
        gate, W = skill_gate(df, self.experts, **self.gate_params)
        df = df.assign(gate=gate)
        df = df.drop(columns=[c for c in self.climate.columns if c in df], errors="ignore").merge(self.climate, left_on="state", right_index=True, how="left")
        if only_from is not None:
            keep = df["date"] >= only_from
            df, W = df[keep], W[keep.to_numpy()]
        out = pd.DataFrame({"state": df["state"], "date": df["date"], "observed_mm": df["y"], "hydra_v3_mm": df["hydra"]})
        mix = self.mixture.predict(df)
        heavy_amount = self.mixture.last_heavy_amount
        equal = df[[f"e_{e}" for e in self.experts]].mean(axis=1).to_numpy()
        amount = {"gate": df["gate"].to_numpy(), "equal_avg": equal, "ens_gate_mixture": 0.5 * df["gate"].to_numpy() + 0.5 * mix,
                  "hydra": df["hydra"].to_numpy()}[self.amount_method]
        out["amount_mm"] = np.clip(amount, 0, None)
        # Run forward chronologically: each issued range uses only earlier verified misses.
        aci = copy.deepcopy(self.aci_template)
        lo, hi, _ = aci.run(df)
        self.aci_levels = aci.state()
        out["lower80_mm"], out["upper80_mm"] = lo, hi
        probs = self.calib.predict(df)
        for name, d in probs.items():
            out[f"p_{name}"] = d["stack"]
        out["heavy_amount_if_occurs_mm"] = heavy_amount
        out["gate_max_weight"] = np.nanmax(W, 1)
        return out


def reliability_cards(evaluation: dict) -> dict:
    """Per-state sample counts and out-of-sample skill, with a plain-language reliability flag."""
    cards = {}
    for st, m in evaluation["states"]["final"].items():
        raw = evaluation["states"]["raw"][st]
        sk = m["mae_skill_persistence"] or 0
        flag = ("good" if sk > 0.05 else "fair" if sk > -0.02 else "weak")
        cards[st] = {"test_state_days": m["n"], "heavy_20mm_events": m["events_20"], "mae_mm": m["mae"], "raw_mae_mm": raw["mae"],
                     "bias_mm": m["bias"], "mae_skill_vs_persistence": m["mae_skill_persistence"], "reliability": flag,
                     "heavy_rain_evidence": "insufficient (<10 events)" if (m["events_20"] or 0) < 10 else "limited (single season)"}
    return cards


def _default_truth() -> str:
    """Use independent IMD truth when the matching evaluation and source exist."""
    imd_eval = ROOT / "runtime" / "hydra_v3_1_evaluation_imd.json"
    return "imd" if imd_eval.exists() and D.IMD_CSV.exists() else "era5"


def main(truth: str | None = None, imd_csv: Path | None = None):
    truth = truth or _default_truth()
    if truth not in ("era5", "imd"):
        raise ValueError("truth must be 'era5' or 'imd'")
    df, payload = D.load(truth=truth, imd_csv=imd_csv)
    suffix = "_imd" if truth == "imd" else ""
    eval_path = ROOT / "runtime" / f"hydra_v3_1_evaluation{suffix}.json"
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else None
    pp = PostProcessor().fit(df, evaluation)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "postprocessor.pkl").open("wb") as fh:
        pickle.dump(pp, fh)
    pred = pp.predict(df)
    states = {}
    for st, g in pred.groupby("state"):
        rows = []
        for r in g.to_dict("records"):
            probs = {k: round(float(r[f"p_{k}"]), 4) for k in EVENTS}
            top = max(("state_20", "local_64.5"), key=lambda k: probs[k])
            rows.append({"valid_date": str(r["date"].date()), "observed_mm": r["observed_mm"], "hydra_v3_mm": round(float(r["hydra_v3_mm"]), 3),
                         "amount_mm": round(float(r["amount_mm"]), 3), "interval80": [round(float(r["lower80_mm"]), 3), round(float(r["upper80_mm"]), 3)],
                         "probabilities": probs, "alert": {"state_20": tier(probs["state_20"]), "local_64.5": tier(probs["local_64.5"])},
                         "heavy_amount_if_occurs_mm": round(float(r["heavy_amount_if_occurs_mm"]), 1), "most_likely_heavy_event": top})
        states[st] = rows
    cards = reliability_cards(evaluation) if evaluation else {}
    result = {"kind": "hydra_v3_1_postprocessed", "model_version": "hydra-rain-v3.1-post", "base_model": payload.get("model_version"),
              "truth": truth,
              "amount_method": pp.amount_method, "gate_params": pp.gate_params,
              "events": EVENTS, "tiers": {name: cut for name, cut in TIERS},
              "interval": {"nominal": 0.8,
                           "method": "adaptive conformal (each tail targets 10%; levels update daily from verified misses, 45-day score buffer)",
                           "current_levels": getattr(pp, "aci_levels", None)},
              "notes": [f"Rows for already-verified dates are in-sample fits (the post-processor was trained on them); the honest scores are in runtime/hydra_v3_1_evaluation{suffix}.json.",
                        "Amounts are expected values and still under-state rare peaks; use the probabilities and alert tiers for heavy rain.",
                        "state_64.5 has very few training events; treat its probabilities as indicative only."],
              "reliability": cards, "states": states}
    (ROOT / "runtime" / "hydra_v3_1_postprocessed.json").write_text(json.dumps(result, default=float, separators=(",", ":")), encoding="utf-8")
    print(f"wrote runtime/hydra_v3_1_postprocessed.json ({len(states)} states), truth {truth}, amount method {pp.amount_method}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", choices=("era5", "imd"), help="verification target; defaults to IMD when available")
    parser.add_argument("--imd-csv", type=Path)
    args = parser.parse_args()
    main(args.truth, args.imd_csv)

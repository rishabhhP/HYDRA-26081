"""Score a published HYDRA rainfall replay with the full evaluation contract.

Works on both the legacy (v2, four rainfall-derived experts) and the v3 replay
payloads, so the before/after comparison uses identical metric code.

    python scripts/evaluate_rainfall_replay.py --replay runtime/hydra_rolling_rainfall_replay.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_rain import metrics as M  # noqa: E402

DEFAULT = ROOT / "runtime" / "hydra_rolling_rainfall_replay.json"
SEASON = {6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post_monsoon", 11: "post_monsoon",
          12: "post_monsoon", 1: "winter", 2: "winter", 3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon"}


def _expert(rows, name):
    return np.array([next((e["value"] for e in r["experts"] if e["name"] == name), np.nan) for r in rows], float)


def score_rows(rows: list[dict], nominal: float, thresholds=(10.0, 20.0)) -> dict:
    obs = np.array([r["actual_mm"] for r in rows], float)
    pred = np.array([r["hydra_mm"] for r in rows], float)
    lo = np.array([r["interval80"][0] for r in rows], float)
    hi = np.array([r["interval80"][1] for r in rows], float)
    p_rain = np.array([r["p_rain"] for r in rows], float) if rows and "p_rain" in rows[0] else None
    refs = {name: _expert(rows, name) for name in ("persistence", "climatology")}
    refs = {k: v for k, v in refs.items() if np.isfinite(v).any()}
    return M.full_report(obs, pred, lower=lo, upper=hi, nominal=nominal, p_rain=p_rain,
                         heavy_thresholds=thresholds, references=refs,
                         dates=np.array([r["valid_date"] for r in rows]))


def evaluate(payload: dict) -> dict:
    nominal = float(payload.get("interval", {}).get("nominal", 0.8))
    states = payload["states"]
    per_state = {state: score_rows(rows, nominal) for state, rows in states.items() if rows}
    pooled_rows = [row for rows in states.values() for row in rows]
    weights = np.array([[e["weight"] for e in r["experts"]] for r in pooled_rows], float)
    experts = [e["name"] for e in pooled_rows[0]["experts"]]
    by_season = {}
    for season in sorted({SEASON[int(r["valid_date"][5:7])] for r in pooled_rows}):
        subset = [r for r in pooled_rows if SEASON[int(r["valid_date"][5:7])] == season]
        by_season[season] = score_rows(subset, nominal)
    dyn = {state: M.weight_dynamics(np.array([[e["weight"] for e in r["experts"]] for r in rows]), experts)
           for state, rows in states.items() if rows}
    return {
        "model_version": payload.get("model_version", "hydra-rain-v2-legacy"),
        "note": "State-average replay; thresholds 10 and 20 mm/day apply to state means, not to cells.",
        "pooled": score_rows(pooled_rows, nominal),
        "by_season": by_season,
        "states": per_state,
        "gate_dynamics_pooled": M.weight_dynamics(weights, experts),
        "gate_dynamics_by_state": dyn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state", default="Maharashtra")
    args = parser.parse_args()
    result = evaluate(json.loads(args.replay.read_text(encoding="utf-8")))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    focus = result["states"].get(args.state)
    print(json.dumps({"pooled": result["pooled"], args.state: focus,
                      "gate_dynamics": result["gate_dynamics_by_state"].get(args.state)}, indent=1))


if __name__ == "__main__":
    main()

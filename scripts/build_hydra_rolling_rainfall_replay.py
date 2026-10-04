"""Build a six-month, one-day-ahead HYDRA rainfall replay for the Overview chart.

The replay is deliberately chronological: each row issued on day *t* uses only
the preceding 35 daily state values (t-34 through t) and predicts rainfall on
t+1.  The observed t+1 ERA5 state mean is retained solely for evaluation and
visual comparison.  The neural gate is fitted before the replay period, so it
does not train on July--December targets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ROOT = ROOT / "nwpblend"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from nwpblend.models.gating import GatingBlender
from scripts import train_hydra_daily_mean_state_blend as daily


SOURCE_DIR = ROOT / "data" / "raw" / "era5_2025_full_source"
OUTPUT = ROOT / "runtime" / "hydra_rolling_rainfall_replay.json"
ARTIFACTS = ROOT / "runtime" / "hydra_rolling_rainfall_replay"
TARGET = "tp_mm"
LEAD_DAYS = 1
HISTORY_DAYS = 35
EXPERTS = daily.EXPERTS


def row_with_35_day_history(state: str, arrays: dict[str, np.ndarray], days: list[date], issue_index: int) -> dict:
    """Construct the same trained feature contract with a 35-day history window."""
    if issue_index < HISTORY_DAYS - 1:
        raise ValueError("A 35-day replay row needs 35 prior daily values")
    row = daily._row(state, arrays, days, issue_index, LEAD_DAYS, TARGET)
    target = arrays[TARGET]
    historic = target[issue_index - HISTORY_DAYS + 1:issue_index + 1]
    climatology = float(np.mean(historic))
    persistence = float(target[issue_index])
    recent3 = float(np.mean(target[issue_index - 2:issue_index + 1]))
    prior7 = float(np.mean(target[issue_index - 6:issue_index + 1]))
    anomaly = max(0.0, climatology + (persistence - prior7))
    expert_values = (climatology, persistence, recent3, anomaly)
    row.update({
        "exp_climatology": climatology,
        "exp_persistence": persistence,
        "exp_recent3": recent3,
        "exp_anom_persistence": anomaly,
        "expert_spread": float(np.std(expert_values)),
        "expert_range": float(np.ptp(expert_values)),
        "now_anomaly": persistence - climatology,
    })
    return row


def _error(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = actual - prediction
    return {
        "mae": round(float(np.mean(np.abs(residual))), 4),
        "rmse": round(float(np.sqrt(np.mean(residual ** 2))), 4),
        "abs_error_p90": round(float(np.quantile(np.abs(residual), 0.90)), 4),
        "samples": int(len(actual)),
    }


def _frames_before(series: dict[str, dict[str, np.ndarray]], days: list[date], cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for state, arrays in series.items():
        for issue_index in range(HISTORY_DAYS - 1, len(days) - LEAD_DAYS):
            if days[issue_index + LEAD_DAYS] > cutoff:
                break
            rows.append(row_with_35_day_history(state, arrays, days, issue_index))
    frame = pd.DataFrame(rows)
    issue_days = sorted(frame.issue_date.unique())
    split = issue_days[max(1, int(len(issue_days) * 0.80)) - 1]
    return (frame[frame.issue_date <= split].reset_index(drop=True),
            frame[frame.issue_date > split].reset_index(drop=True))


def build(source_dir: Path, output: Path, artifacts: Path, start: date, end: date) -> dict:
    archives = daily._archives(source_dir)
    series, days, indices, fallback = daily._build_series(archives)
    train_cutoff = start - timedelta(days=1)
    train, calibration = _frames_before(series, days, train_cutoff)
    if train.empty or calibration.empty:
        raise ValueError("Insufficient history before replay start to train and calibrate the HYDRA gate")

    static = GatingBlender(static=True).fit(train, calibration)
    gate = GatingBlender().fit(train, calibration, anchor_init=static.static_logits())
    artifacts.mkdir(parents=True, exist_ok=True)
    static.save(artifacts, "gating_static")
    gate.save(artifacts, "gating")

    calibration_pred, _ = gate.predict(calibration.drop(columns=["state", "issue_date"]), return_weights=True)
    calibration_errors: dict[str, dict[str, float | int]] = {}
    for state, group in calibration.groupby("state"):
        positions = group.index.to_numpy()
        calibration_errors[state] = _error(group.target.to_numpy(), calibration_pred[positions])
    global_error = _error(calibration.target.to_numpy(), calibration_pred)

    states: dict[str, list[dict]] = {}
    for state, arrays in series.items():
        rows: list[dict] = []
        for issue_index in range(HISTORY_DAYS - 1, len(days) - LEAD_DAYS):
            valid_day = days[issue_index + LEAD_DAYS]
            if valid_day < start or valid_day > end:
                continue
            row = row_with_35_day_history(state, arrays, days, issue_index)
            frame = pd.DataFrame([row]).drop(columns=["state", "issue_date"])
            prediction, weights = gate.predict(frame, return_weights=True)
            expert_values = [float(frame.iloc[0][f"exp_{expert}"]) for expert in EXPERTS]
            error = calibration_errors.get(state, global_error)
            margin = float(error["abs_error_p90"])
            value = float(prediction[0])
            rows.append({
                "issue_date": str(days[issue_index]),
                "valid_date": str(valid_day),
                "actual_mm": round(float(arrays[TARGET][issue_index + LEAD_DAYS]), 3),
                "hydra_mm": round(value, 3),
                "interval80": [round(max(0.0, value - margin), 3), round(value + margin, 3)],
                "spread_mm": round(float(np.std(expert_values)), 3),
                "experts": [{"name": expert, "value": round(expert_values[number], 3), "weight": round(float(weights[0, number]), 6)} for number, expert in enumerate(EXPERTS)],
            })
        states[state] = rows

    payload = {
        "status": "available",
        "kind": "hydra_rolling_rainfall_replay",
        "target": TARGET,
        "unit": "mm/day",
        "lead_hours": 24,
        "history_days": HISTORY_DAYS,
        "training_cutoff": str(train_cutoff),
        "calibration": "Chronological pre-replay calibration; p90 absolute residual supplies each displayed 80% empirical interval.",
        "coverage": {"start": str(start), "end": str(end), "days": (end - start).days + 1},
        "source_archives": {str(month): path.name for month, path in archives.items()},
        "message": "Daily HYDRA rainfall replay. Every +24-hour value was issued with only the preceding 35 daily state observations. Actual ERA5 rainfall is retained for the historical comparison only.",
        "states": states,
        "grid_cell_count": {state: int(len(cells)) for state, cells in indices.items()},
        "nearest_grid_fallback": fallback,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    args = parser.parse_args()
    payload = build(args.source_dir, args.output, args.artifacts, args.start, args.end)
    print(f"Published {payload['coverage']['days']} HYDRA rainfall replay days for {len(payload['states'])} states to {args.output}")


if __name__ == "__main__":
    main()

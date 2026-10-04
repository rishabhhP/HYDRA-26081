"""Build the six-month HYDRA rainfall replay for the Overview chart (HYDRA rainfall v3).

The model is trained at 0.25 degree grid-cell level on ERA5 atmospheric predictors with
truth dated before the replay start only. Each replay row is issued on day t and predicts
rainfall on t+1; observed ERA5 rainfall for t+1 is used solely for verification.

Grid predictions are aggregated to each state at the end, alongside the local extremes
the area mean hides: the largest predicted cell value, the expected fraction of the state
above 20 / 64.5 mm/day, and the largest cell heavy-rain probability.

    python scripts/build_hydra_rolling_rainfall_replay.py                    # ERA5 under data/raw/
    python scripts/build_hydra_rolling_rainfall_replay.py --synthetic-dry-run  # pipeline check only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_rain import config as C  # noqa: E402
from hydra_rain import report as R  # noqa: E402
from hydra_rain.experts import EXPERTS  # noqa: E402
from hydra_rain.grid import load_era5, synthetic_cube  # noqa: E402
from hydra_rain.pipeline import HydraRainModel  # noqa: E402

SOURCE_DIR = ROOT / "data" / "raw" / "era5_2025_full_source"
OUTPUT = ROOT / "runtime" / "hydra_rolling_rainfall_replay.json"
ARTIFACTS = ROOT / "runtime" / "hydra_rain_v3"
LEAD_DAYS = 1


def _r(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def replay_rows(states_frame) -> list[dict]:
    rows = []
    heavy_keys = list(C.HEAVY_KEYS)
    for record in states_frame.sort_values("issue_idx").to_dict("records"):
        rows.append({
            "issue_date": record["issue_date"], "valid_date": record["valid_date"],
            "actual_mm": _r(record["actual_mm"]), "hydra_mm": _r(record["hydra_mm"]),
            "interval80": [_r(record["lower80"]), _r(record["upper80"])],
            "interval_stratum": record.get("interval_stratum") or None,
            "spread_mm": _r(record["spread_mm"]),
            "experts": [{"name": e, "value": _r(record[f"exp_{e}"]), "weight": _r(record[f"w_{e}"], 6)} for e in EXPERTS],
            "p_rain": _r(record["p_rain"], 4),
            "wet_amount_mm": _r(record["wet_amount_mm"]),
            "p_heavy_area": {k: _r(record[f"p_heavy_{k}"], 4) for k in heavy_keys},
            "p_heavy_max_cell": {k: _r(record[f"max_p_heavy_{k}"], 4) for k in heavy_keys},
            "local_peak_mm": _r(record["local_peak_mm"]),
            "actual_local_max_mm": _r(record["actual_local_max_mm"]),
            "actual_heavy_area": {f"{t:g}": _r(record[f"actual_heavy_frac_{t:g}"], 4) for t in C.HEAVY_THRESHOLDS_MM},
            "regime": C.REGIMES[int(record["regime_id"])],
        })
    return rows


def build(cube, output: Path, artifacts: Path, start: date, end: date, train_gate: bool = True, label: str = "ERA5") -> dict:
    model = HydraRainModel().fit(cube, np.datetime64(start - timedelta(days=1)), train_gate=train_gate)
    model.save(artifacts)
    first = cube.day_index(np.datetime64(start)) - LEAD_DAYS
    last = min(cube.day_index(np.datetime64(end)) - LEAD_DAYS, cube.n_days - 1 - LEAD_DAYS)
    cells, states = model.predict(cube, np.arange(first, last + 1))
    cells = cells[cells["actual_mm"].notna()]
    states = states[states["actual_mm"].notna()]
    lead1 = states[states["lead_days"] == LEAD_DAYS]
    per_state, metrics_state, metrics_cell = {}, {}, {}
    for name in cube.state_names:
        frame = lead1[lead1["state"] == name]
        per_state[name] = replay_rows(frame)
        metrics_state[name] = {f"lead{lead}": R.state_metrics(states[(states["state"] == name) & (states["lead_days"] == lead)])
                               for lead in C.LEADS}
        member = cells[cells["cell"].isin(cube.state_cells[name]) & (cells["lead_days"] == LEAD_DAYS)]
        metrics_cell[name] = R.cell_metrics(member)
    payload = {
        "status": "available",
        "kind": "hydra_rolling_rainfall_replay",
        "model_version": C.MODEL_VERSION,
        "target": "tp_mm", "unit": "mm/day", "lead_hours": LEAD_DAYS * 24,
        "history_days": 30,
        "training_cutoff": model.meta["cutoff"],
        "training_splits": model.meta["splits"],
        "interval": {**model.state_conformal.summary(), "nominal": 1 - C.ALPHA, "alpha": C.ALPHA,
                     "label": "Conformal central 80% interval",
                     "method": "Stratified split-conformal quantile regression (state x season x lead x regime, "
                               "hierarchical fallback), calibrated on a held-out slice before the replay."},
        "calibration": "Central 80% interval from stratified split-conformal CQR (alpha = 0.20), fitted on the last "
                       "15% of pre-replay days, which neither the experts nor the gate trained on.",
        "coverage": {"start": str(start), "end": str(end), "days": (end - start).days + 1},
        "heavy_thresholds_mm": list(C.HEAVY_THRESHOLDS_MM),
        "state_p95_mm": dict(zip(cube.state_names, map(lambda v: round(float(v), 2), model.meta["state_p95"]))),
        "experts": list(EXPERTS),
        "missing_optional_fields": cube.missing_optional,
        "message": (f"HYDRA v3 daily rainfall replay. Trained at 0.25 degree grid level on {label} atmospheric "
                    "predictors (CAPE, moisture, pressure, wind, cloud, radiation) plus rainfall history, then "
                    "aggregated to the state. Every +24-hour value used only data available on its issue day; "
                    "observed rainfall is retained for comparison only."),
        "states": per_state,
        "metrics": {"state_scale": metrics_state, "cell_scale": metrics_cell},
        "gate_dynamics": R.gate_dynamics(lead1),
        "grid_cell_count": {name: int(len(cube.state_cells[name])) for name in cube.state_names},
        "nearest_grid_fallback": {name: bool(cube.fallback.get(name)) for name in cube.state_names},
    }
    if not train_gate:
        payload["status"] = "dry_run"
        payload["message"] = "PIPELINE DRY RUN ONLY: synthetic weather and an untrained gate. Not a forecast."
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="Folder (searched recursively) holding ERA5 daily-mean zips/NetCDF for one or more years")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--synthetic-dry-run", action="store_true",
                        help="Exercise the pipeline on synthetic weather (never overwrites the published replay)")
    args = parser.parse_args()
    if args.synthetic_dry_run:
        cube = synthetic_cube(start="2024-01-01", days=730)
        try:
            import torch  # noqa: F401
            train_gate = True
        except ImportError:
            train_gate = False
        output = args.output.with_name("hydra_rolling_rainfall_replay.dry_run.json")
        payload = build(cube, output, args.artifacts.with_name("hydra_rain_v3_dry_run"),
                        date(2025, 7, 1), date(2025, 12, 29), train_gate, "synthetic")
    else:
        cube = load_era5(args.source_dir)
        output = args.output
        payload = build(cube, output, args.artifacts, args.start, args.end)
    print(f"Published {payload['coverage']['days']} HYDRA v3 replay days for {len(payload['states'])} states to {output}")


if __name__ == "__main__":
    main()

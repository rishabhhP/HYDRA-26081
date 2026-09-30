"""Train and publish a HYDRA neural-gating state blend from ERA5 daily means.

The supplied CDS archives preserve daily means, so this job trains a new HYDRA
state model with the fields that are actually present.  It uses the repository's
``GatingBlender``: a neural MLP gate over climatology, persistence, recent-three
day, and anomaly-persistence experts.  It does not load or alter the blocked
checkpoint whose feature contract includes unavailable daily extrema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
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

from backend import data as D
from nwpblend import config as C
from nwpblend.features import CONTEXT_COLS
from nwpblend.models.gating import GatingBlender


SOURCE_DIR = ROOT / "data" / "raw" / "era5_2025_full_source"
RUNTIME = ROOT / "runtime"
OUTPUT = RUNTIME / "hydra_daily_mean_state_blend.json"
ARTIFACTS = RUNTIME / "hydra_daily_mean_state_blend"
FILES = {
    "t2m": "2m_temperature_0_daily-mean.nc", "d2m": "2m_dewpoint_temperature_0_daily-mean.nc",
    "u10": "10m_u_component_of_wind_stream-oper_daily-mean.nc", "v10": "10m_v_component_of_wind_0_daily-mean.nc",
    "msl": "mean_sea_level_pressure_0_daily-mean.nc", "tp": "total_precipitation_0_daily-mean.nc",
    "ssrd": "surface_solar_radiation_downwards_0_daily-mean.nc", "tcc": "total_cloud_cover_0_daily-mean.nc",
    "cape": "convective_available_potential_energy_0_daily-mean.nc", "blh": "boundary_layer_height_0_daily-mean.nc",
}
TARGETS = {"tp_mm": "mm/day", "t2m_C_mean": "°C", "wind_speed_mean": "m/s"}
EXPERTS = ("climatology", "persistence", "recent3", "anom_persistence")


def _month(path: Path) -> int | None:
    aliases = {"jan": 1, "feb": 2, "march": 3, "april": 4, "may": 5, "june": 6,
               "july": 7, "august": 8, "sep": 9, "october": 10, "nov": 11, "dec": 12}
    return next((month for token, month in aliases.items() if token in path.name.casefold()), None)


def _archives(source: Path) -> dict[int, Path]:
    expected = set(FILES.values())
    selected: dict[int, Path] = {}
    for path in source.glob("*.zip*"):
        month = _month(path)
        if month is None:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = {Path(item.filename).name for item in archive.infolist() if not item.is_dir()}
            if expected.issubset(names):
                previous = selected.get(month)
                if previous is None or path.name > previous.name:
                    selected[month] = path
        except zipfile.BadZipFile:
            continue
    missing = sorted(set(range(1, 13)) - set(selected))
    if missing:
        raise FileNotFoundError(f"Complete daily-mean archives are missing for month(s): {missing}")
    return selected


def _state_indices(latitude: np.ndarray, longitude: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    lat2, lon2 = np.meshgrid(latitude, longitude, indexing="ij")
    flat_lat, flat_lon = lat2.ravel(), lon2.ravel()
    assignments = {feature["properties"]["name"]: [] for feature in D.state_features()}
    for index, (lat, lon) in enumerate(zip(flat_lat, flat_lon)):
        state = D.state_at(float(lat), float(lon))
        if state:
            assignments[state].append(index)
    fallback: dict[str, bool] = {}
    for feature in D.state_features():
        name = feature["properties"]["name"]
        if assignments[name]:
            fallback[name] = False
            continue
        center = D.geometry_center(feature["geometry"])
        nearest = int(np.argmin((flat_lat - center["latitude"]) ** 2 + (flat_lon - center["longitude"]) ** 2))
        assignments[name] = [nearest]
        fallback[name] = True
    return {name: np.asarray(cells, dtype=np.int32) for name, cells in assignments.items()}, fallback


def _read_field(archive_path: Path, field: str, indices: dict[str, np.ndarray] | None):
    import xarray as xr

    filename = FILES[field]
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="hydra-daily-mean-") as directory:
        extracted = Path(directory) / filename
        with archive.open(filename) as source, extracted.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        with xr.open_dataset(extracted) as dataset:
            variable = next(iter(dataset.data_vars))
            values = dataset[variable].sel(latitude=slice(37.0, 7.0), longitude=slice(68.0, 97.0))
            if values.shape[1:] != (121, 117):
                raise ValueError(f"{archive_path.name}/{filename} does not match the HYDRA India grid: {values.shape}")
            if indices is None:
                indices, fallback = _state_indices(values.latitude.values.astype(float), values.longitude.values.astype(float))
            else:
                fallback = None
            dates = [str(timestamp.date()) for timestamp in pd.DatetimeIndex(values.valid_time.values)]
            flattened = values.values.astype(np.float64).reshape(len(dates), -1)
    means = {state: np.nanmean(flattened[:, cells], axis=1) for state, cells in indices.items()}
    return dates, means, indices, fallback


def _humidity(temperature: np.ndarray, dewpoint: np.ndarray) -> np.ndarray:
    return np.clip(100 * np.exp(17.625 * dewpoint / (243.04 + dewpoint) - 17.625 * temperature / (243.04 + temperature)), 0, 100)


def _season_id(day: date) -> int:
    return C.SEASONS.index(C.SEASON_OF_MONTH[day.month])


def _regime_id(rainfall: np.ndarray, index: int, season_id: int) -> int:
    trailing = float(np.mean(rainfall[max(0, index - 2):index + 1]))
    if season_id == C.SEASONS.index("monsoon") and trailing >= C.MONSOON_TP3_MM:
        return C.REGIMES.index("monsoon")
    if trailing < C.DRY_TP7_MM:
        return C.REGIMES.index("dry_spell")
    return C.REGIMES.index("normal")


def _row(state: str, arrays: dict[str, np.ndarray], days: list[date], index: int, lead: int, target_name: str) -> dict:
    target = arrays[target_name]
    historic = target[max(0, index - 30):index + 1]
    climatology = float(np.mean(historic))
    persistence = float(target[index])
    recent3 = float(np.mean(target[index - 2:index + 1]))
    prior7 = float(np.mean(target[index - 6:index + 1]))
    anomaly = climatology + (persistence - prior7)
    if target_name == "tp_mm":
        anomaly = max(0.0, anomaly)
    day = days[index]
    season_id = _season_id(day)
    base = {column: 0.0 for column in CONTEXT_COLS}
    center = D.geometry_center(D.state_feature(state)["geometry"])
    base.update({
        "latitude": center["latitude"], "longitude": center["longitude"], "is_ocean": 0, "elev_proxy_m": 0,
        "coastal": 0, "region_id": sorted(feature["properties"]["name"] for feature in D.state_features()).index(state),
        "season_id": season_id, "lead_days": lead, "lead_bucket_id": 1 if lead == 1 else 2,
        "regime_id": _regime_id(arrays["tp_mm"], index, season_id),
        "now_tp_mm": arrays["tp_mm"][index], "now_t2m_C_mean": arrays["t2m_C_mean"][index],
        "now_wind_speed_mean": arrays["wind_speed_mean"][index], "now_msl_hPa_mean": arrays["msl_hPa_mean"][index],
        "now_msl_hPa_min": arrays["msl_hPa_mean"][index], "now_tcc_mean": arrays["tcc_mean"][index],
        "now_cape_max": arrays["cape_max"][index], "now_rh_pct_mean": arrays["rh_pct_mean"][index],
        "now_blh_mean": arrays["blh_mean"][index], "now_ssrd_MJ": arrays["ssrd_MJ"][index],
        "now_msl_anom": arrays["msl_hPa_mean"][index] - np.mean(arrays["msl_hPa_mean"][max(0, index - 30):index + 1]),
        "now_tp3": np.mean(arrays["tp_mm"][index - 2:index + 1]), "now_tp7": np.mean(arrays["tp_mm"][index - 6:index + 1]),
        "now_target_tendency": persistence - float(target[index - 1]), "now_anomaly": persistence - climatology,
        "exp_climatology": climatology, "exp_persistence": persistence, "exp_recent3": recent3,
        "exp_anom_persistence": anomaly, "expert_spread": np.std([climatology, persistence, recent3, anomaly]),
        "expert_range": np.ptp([climatology, persistence, recent3, anomaly]),
    })
    base["state"] = state
    base["issue_date"] = str(day)
    base["target"] = float(target[index + lead]) if index + lead < len(days) else np.nan
    return base


def _frames(series: dict[str, dict[str, np.ndarray]], days: list[date], target: str):
    rows = []
    for state, arrays in series.items():
        for index in range(7, len(days) - 2):
            for lead in (1, 2):
                rows.append(_row(state, arrays, days, index, lead, target))
    frame = pd.DataFrame(rows)
    ordered = sorted(frame.issue_date.unique())
    train_end, validation_end = ordered[int(len(ordered) * .70)], ordered[int(len(ordered) * .85)]
    return (frame[frame.issue_date <= train_end].copy().reset_index(drop=True),
            frame[(frame.issue_date > train_end) & (frame.issue_date <= validation_end)].copy().reset_index(drop=True),
            frame[frame.issue_date > validation_end].copy().reset_index(drop=True))


def _error(actual: np.ndarray, prediction: np.ndarray) -> dict:
    residual = actual - prediction
    return {"mae": round(float(np.mean(np.abs(residual))), 4), "rmse": round(float(np.sqrt(np.mean(residual ** 2))), 4),
            "abs_error_p90": round(float(np.quantile(np.abs(residual), .90)), 4), "samples": int(len(actual))}


def _build_series(archives: dict[int, Path]):
    raw: dict[str, dict[str, dict[str, float]]] = {}
    indices = None
    fallback: dict[str, bool] | None = None
    for field in FILES:
        print(f"Loading {field} from 2025 ERA5 archives", flush=True)
        for month, archive in archives.items():
            days, means, indices, new_fallback = _read_field(archive, field, indices)
            fallback = fallback or new_fallback
            for row, day in enumerate(days):
                for state, values in means.items():
                    raw.setdefault(state, {}).setdefault(day, {})[field] = float(values[row])
    dates = sorted(date.fromisoformat(day) for day, values in next(iter(raw.values())).items() if len(values) == len(FILES))
    series = {}
    for state, by_day in raw.items():
        if any(len(by_day.get(str(day), {})) != len(FILES) for day in dates):
            raise ValueError(f"Incomplete daily mean fields for {state}")
        t2m = np.asarray([by_day[str(day)]["t2m"] - 273.15 for day in dates])
        d2m = np.asarray([by_day[str(day)]["d2m"] - 273.15 for day in dates])
        series[state] = {
            "tp_mm": np.clip(np.asarray([by_day[str(day)]["tp"] * 24000 for day in dates]), 0, None),
            "t2m_C_mean": t2m,
            "wind_speed_mean": np.asarray([np.hypot(by_day[str(day)]["u10"], by_day[str(day)]["v10"]) for day in dates]),
            "msl_hPa_mean": np.asarray([by_day[str(day)]["msl"] / 100 for day in dates]),
            "tcc_mean": np.asarray([by_day[str(day)]["tcc"] for day in dates]),
            "cape_max": np.asarray([by_day[str(day)]["cape"] for day in dates]),
            "rh_pct_mean": _humidity(t2m, d2m),
            "blh_mean": np.asarray([by_day[str(day)]["blh"] for day in dates]),
            "ssrd_MJ": np.asarray([by_day[str(day)]["ssrd"] * 24 / 1e6 for day in dates]),
        }
    return series, dates, indices, fallback or {}


def train(source: Path, output: Path, artifacts: Path) -> dict:
    archives = _archives(source)
    series, days, indices, fallback = _build_series(archives)
    models, metrics = {}, {}
    for target in TARGETS:
        print(f"Training neural gate for {target}", flush=True)
        train_frame, validation, test = _frames(series, days, target)
        static = GatingBlender(static=True).fit(train_frame, validation)
        gate = GatingBlender().fit(train_frame, validation, anchor_init=static.static_logits())
        artifacts_target = artifacts / target
        static.save(artifacts_target, "gating_static")
        gate.save(artifacts_target, "gating")
        prediction, _ = gate.predict(test, return_weights=True)
        metrics[target] = {}
        for lead in (1, 2):
            subset = test.lead_days == lead
            metrics[target][str(lead)] = {
                "global": _error(test.loc[subset, "target"].to_numpy(), prediction[subset]),
                "states": {state: _error(group.target.to_numpy(), prediction[group.index.to_numpy()])
                           for state, group in test.loc[subset].groupby("state")},
            }
        models[target] = gate
    issue_index = len(days) - 1
    states = {}
    for state, arrays in series.items():
        outlooks = []
        for lead in (1, 2):
            forecast = {}
            for target, unit in TARGETS.items():
                frame = pd.DataFrame([_row(state, arrays, days, issue_index, lead, target)]).drop(columns=["state", "issue_date"])
                value, weights = models[target].predict(frame, return_weights=True)
                expert_values = [float(frame.iloc[0][f"exp_{expert}"]) for expert in EXPERTS]
                error = metrics[target][str(lead)]["states"].get(state, metrics[target][str(lead)]["global"])
                margin = error["abs_error_p90"]
                lower = float(value[0] - margin)
                if target == "tp_mm":
                    lower = max(0.0, lower)
                forecast[target] = {
                    "value": round(float(value[0]), 3), "minimum": round(min(expert_values), 3), "maximum": round(max(expert_values), 3),
                    "unit": unit, "spread": round(float(np.std(expert_values)), 3),
                    "interval80": [round(lower, 3), round(float(value[0] + margin), 3)],
                    "experts": [{"name": expert, "weight": round(float(weights[0, index]), 6), "value": round(expert_values[index], 3)} for index, expert in enumerate(EXPERTS)],
                    "backtest_error": error,
                }
            outlooks.append({"lead_hours": lead * 24, "valid_date": str(days[-1] + timedelta(days=lead)), "regime": None,
                             "grid_cell_count": int(len(indices[state])), "forecast": forecast, "events": []})
        states[state] = {"outlooks": outlooks, "nearest_grid_fallback": fallback.get(state, False)}
    payload = {
        "status": "available", "kind": "hydra_daily_mean_neural_gating", "issue_date": str(days[-1]),
        "source_archives": {str(month): path.name for month, path in archives.items()}, "states": states,
        "message": "HYDRA daily-mean adaptive blend retrained on supplied 2025 ERA5 state series. Neural gating learns a convex allocation across climatology, persistence, recent-three-day, and anomaly-persistence experts. Disagreement is expert spread; the 80% band uses held-out rolling backtest residuals.",
        "training_metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    payload = train(args.source_dir, args.output, args.artifacts)
    print(f"Published trained HYDRA daily-mean blend for {len(payload['states'])} states to {args.output}")


if __name__ == "__main__":
    main()

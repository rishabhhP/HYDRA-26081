"""Every data source the HYDRA prototype ships, behind one query interface.

Offline sources (always available):
  imd       IMD state-wise daily rainfall with normals, departures and categories (+ weekly, monthly, cumulative)
  replay    ERA5 state-mean daily rainfall with HYDRA +24h predictions and experts (rolling replay)
  hydra     Published HYDRA daily-mean state forecast cycle (rain, temperature, wind; +24h/+48h)
  ecmwf     ECMWF IFS 0.25 degree analysis snapshot, aggregated to states (temperature, humidity, wind, gusts,
            CAPE, cloud, pressure, water vapour, dew point, radiation)
  station   Meteostat single-station 2025 daily export
  bench     nwpblend benchmark skill tables
Live sources (need internet on the server) are in liveclient.py.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import lexicon as L

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "nwpblend"
RUNTIME = ROOT / "runtime"
CACHE = Path(__file__).resolve().parent / "cache"

IMD_REGIONS = {"COUNTRY : INDIA": "India", "REGION : CENTRAL INDIA": "Central India",
               "REGION : NORTH WEST INDIA": "North West India", "REGION : SOUTH PENINSULA": "South Peninsula",
               "REGION : EAST & NORTH EAST INDIA": "East and North East India"}
IMD_CATEGORY_TEXT = {"large_excess": "large excess (≥ +60% of normal)", "excess": "excess (+20% to +59%)",
                     "normal": "normal (−19% to +19%)", "deficient": "deficient (−20% to −59%)",
                     "large_deficient": "large deficient (−60% to −99%)", "no_rain": "no rain (−100%)",
                     "no_data": "no data"}


def _canon_imd(name: str) -> str:
    if name in IMD_REGIONS:
        return name
    n = re.sub(r"\s*\(UT\)\s*", "", name).replace("&", "and").strip().lower()
    for state in L.STATES:
        if re.sub(r"[^a-z]", "", state.lower()) == re.sub(r"[^a-z]", "", n):
            return state
    return name.title()


# ------------------------------------------------------------------ IMD
@lru_cache(maxsize=1)
def imd() -> pd.DataFrame:
    path = REPO / "rainfall_statewise_daily_imd_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"cumulative_departue_pct": "cumulative_departure_pct"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["place"] = df["state"].map(_canon_imd)
    return df


def imd_coverage() -> tuple[date, date] | None:
    df = imd()
    return (df["date"].min(), df["date"].max()) if not df.empty else None


def imd_series(place: str, start: date, end: date) -> pd.DataFrame:
    df = imd()
    if df.empty:
        return df
    return df[(df["place"] == place) & (df["date"] >= start) & (df["date"] <= end)].sort_values("date")


def imd_states_on(start: date, end: date) -> pd.DataFrame:
    df = imd()
    if df.empty:
        return df
    return df[(df["date"] >= start) & (df["date"] <= end) & df["place"].isin(L.STATES)]


# ------------------------------------------------------------------ ERA5 / HYDRA replay
@lru_cache(maxsize=1)
def replay_payload() -> dict:
    path = RUNTIME / "hydra_rolling_rainfall_replay.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@lru_cache(maxsize=1)
def replay() -> pd.DataFrame:
    payload = replay_payload()
    rows = []
    for state, records in (payload.get("states") or {}).items():
        for r in records:
            row = {"place": state, "date": date.fromisoformat(r["valid_date"]), "issue_date": r["issue_date"],
                   "actual_mm": r["actual_mm"], "hydra_mm": r["hydra_mm"],
                   "lo80": r["interval80"][0], "hi80": r["interval80"][1], "spread_mm": r.get("spread_mm"),
                   "p_rain": r.get("p_rain"), "local_peak_mm": r.get("local_peak_mm"),
                   "actual_local_max_mm": r.get("actual_local_max_mm")}
            for e in r.get("experts", []):
                row[f"exp_{e['name']}"] = e["value"]
                row[f"w_{e['name']}"] = e["weight"]
            rows.append(row)
    return pd.DataFrame(rows)


def replay_coverage() -> tuple[date, date] | None:
    df = replay()
    return (df["date"].min(), df["date"].max()) if not df.empty else None


def replay_series(place: str, start: date, end: date) -> pd.DataFrame:
    df = replay()
    if df.empty:
        return df
    return df[(df["place"] == place) & (df["date"] >= start) & (df["date"] <= end)].sort_values("date")


def replay_meta() -> dict:
    p = replay_payload()
    return {k: p.get(k) for k in ("model_version", "training_cutoff", "calibration", "interval", "lead_hours",
                                  "history_days", "coverage", "experts")}


def replay_experts() -> list[str]:
    df = replay()
    return [c[2:] for c in df.columns if c.startswith("w_")]


# ------------------------------------------------------------------ HYDRA published state cycle
@lru_cache(maxsize=1)
def hydra_cycle() -> dict:
    path = RUNTIME / "hydra_daily_mean_state_blend.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def hydra_state(place: str) -> dict | None:
    return (hydra_cycle().get("states") or {}).get(place)


def hydra_valid_dates() -> list[str]:
    any_state = next(iter((hydra_cycle().get("states") or {}).values()), None)
    return [o["valid_date"] for o in any_state["outlooks"]] if any_state else []


# ------------------------------------------------------------------ ECMWF analysis snapshot by state
ECMWF_FIELDS = {"temperature": ("t2m_C", "°C"), "humidity": ("rh_pct", "%"), "wind": ("wind_speed", "m/s"),
                "wind_gust": ("wind_gust_3h", "m/s"), "cape": ("mucape", "J/kg"), "thunderstorm": ("mucape", "J/kg"),
                "cloud": ("tcc", "fraction"), "pressure": ("msl_hPa", "hPa"), "water_vapour": ("tcwv", "kg/m²"),
                "dew_point": ("d2m_C", "°C"), "solar": ("ssrd_MJ", "MJ/m²"), "heat_stress": ("skt_C", "°C")}


@lru_cache(maxsize=1)
def ecmwf_grid() -> pd.DataFrame:
    path = REPO / "forecast_india_2026-09-24_0h_clean.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def ecmwf_states() -> pd.DataFrame:
    """Per-state mean / max / min of every snapshot field (cached to disk after the first build)."""
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / "ecmwf_state_summary.csv"
    grid = ecmwf_grid()
    if grid.empty:
        return pd.DataFrame()
    if cached.exists():
        return pd.read_csv(cached)
    from backend import data as D
    grid = grid.copy()
    grid["place"] = [D.state_at(float(a), float(b)) for a, b in zip(grid["latitude"], grid["longitude"])]
    inside = grid[grid["place"].notna()]
    fields = sorted({f for f, _ in ECMWF_FIELDS.values()})
    agg = inside.groupby("place")[fields].agg(["mean", "max", "min"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["cells"] = inside.groupby("place").size()
    agg = agg.reset_index()
    agg["valid_time"] = str(grid["valid_time"].iloc[0])
    agg.to_csv(cached, index=False)
    return agg


def ecmwf_date() -> date | None:
    g = ecmwf_grid()
    return date.fromisoformat(str(g["valid_time"].iloc[0])[:10]) if not g.empty else None


def ecmwf_point(lat: float, lon: float) -> dict | None:
    g = ecmwf_grid()
    if g.empty:
        return None
    i = int(((g["latitude"] - lat) ** 2 + (g["longitude"] - lon) ** 2).idxmin())
    return g.loc[i].to_dict()


# ------------------------------------------------------------------ station + benchmarks
@lru_cache(maxsize=1)
def station() -> pd.DataFrame:
    path = REPO / "export_meteostat_2025_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@lru_cache(maxsize=1)
def benchmark() -> pd.DataFrame:
    path = REPO / "model_skill_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def benchmark_season() -> pd.DataFrame:
    path = REPO / "reports" / "results" / "blocked" / "pm_season.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def validation() -> dict:
    path = RUNTIME / "hydra_rainfall_validation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------ coverage registry
def coverage() -> list[dict]:
    out = []
    c = imd_coverage()
    if c:
        out.append({"source": "IMD state-wise daily rainfall", "key": "imd", "variables": ["rainfall"],
                    "start": c[0], "end": c[1], "scope": "36 states/UTs, 4 regions and all-India",
                    "detail": "observed daily rainfall, normal, departure %, category; weekly, monthly and season-cumulative totals"})
    c = replay_coverage()
    if c:
        out.append({"source": "ERA5 state-mean rainfall + HYDRA replay", "key": "replay", "variables": ["rainfall"],
                    "start": c[0], "end": c[1], "scope": "36 states/UTs",
                    "detail": "reanalysis daily state-average rainfall with HYDRA's +24h forecast, expert values, weights and interval"})
    dates = hydra_valid_dates()
    if dates:
        out.append({"source": "HYDRA published state forecast cycle", "key": "hydra", "variables": ["rainfall", "temperature", "wind"],
                    "start": date.fromisoformat(dates[0]), "end": date.fromisoformat(dates[-1]), "scope": "36 states/UTs",
                    "detail": f"issued {hydra_cycle().get('issue_date')}; +24h and +48h with expert weights and backtest error"})
    d = ecmwf_date()
    if d:
        out.append({"source": "ECMWF IFS analysis snapshot", "key": "ecmwf",
                    "variables": sorted(ECMWF_FIELDS), "start": d, "end": d, "scope": "0.25° grid over India, summarised per state",
                    "detail": "zero-hour analysis; zero-hour precipitation is not daily rainfall"})
    s = station()
    if not s.empty:
        out.append({"source": "Meteostat station export", "key": "station", "variables": ["temperature", "rainfall", "wind", "pressure"],
                    "start": s["date"].min(), "end": s["date"].max(), "scope": "one station (not identified in the export)",
                    "detail": "daily mean/min/max temperature, precipitation, wind, pressure"})
    out.append({"source": "Open-Meteo forecast and historical archive (live)", "key": "live",
                "variables": ["rainfall", "temperature", "humidity", "wind", "wind_gust", "cloud", "pressure", "solar", "cape",
                              "thunderstorm", "heat_stress", "soil_moisture", "dew_point"],
                "start": date(1940, 1, 1), "end": None, "scope": "any Indian state (representative point), city or coordinate",
                "detail": "needs internet on the HYDRA server; up to 16 days ahead"})
    return out


def nice(v, digits=1):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:,.{digits}f}"

"""Resolve (place, variable, period) to the best available numbers across every source.

Priority for observed rainfall: IMD (official state values with normals) > ERA5 replay > Open-Meteo archive.
Other variables: ECMWF snapshot on its date > HYDRA cycle on its dates (forecast) > Open-Meteo.
Every result carries source, coverage and a state-mean flag so answers can explain what the number is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import datahub as H
from . import liveclient as LC
from .slots import Place
from .timeparse import TimeSpec, today

UNITS = {"rainfall": "mm", "temperature": "°C", "humidity": "%", "wind": "km/h", "wind_gust": "km/h", "pressure": "hPa",
         "cloud": "%", "cape": "J/kg", "thunderstorm": "J/kg", "solar": "MJ/m²", "dew_point": "°C", "heat_stress": "°C",
         "soil_moisture": "m³/m³", "water_vapour": "kg/m²"}


@dataclass
class Series:
    place: str
    variable: str
    source: str
    source_key: str
    unit: str
    dates: list = field(default_factory=list)
    values: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)        # normals, departures, categories, hydra values, ...
    state_mean: bool = True
    note: str = ""
    status: str = "available"
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "available" and len(self.values) > 0

    def arr(self) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in self.values], float)

    def stat(self, how: str) -> float | None:
        a = self.arr()
        a = a[np.isfinite(a)]
        if not a.size:
            return None
        return float({"total": a.sum(), "sum": a.sum(), "mean": a.mean(), "max": a.max(), "min": a.min()}.get(how, a.mean()))


def _overlap(ts: TimeSpec, cov: tuple[date, date] | None) -> tuple[date, date] | None:
    if not cov or not ts.start:
        return None
    a, b = max(ts.start, cov[0]), min(ts.end, cov[1])
    return (a, b) if a <= b else None


def state_of(place: Place) -> str | None:
    return place.state if place.kind in ("state", "city") else None


def point_of(place: Place) -> tuple[str, float, float]:
    if place.kind == "city" and place.lat is not None:
        return (place.name, place.lat, place.lon)
    return LC.state_point(place.state or place.name)


# ------------------------------------------------------------------ rainfall
def imd_series(place: Place, ts: TimeSpec) -> Series | None:
    key = "COUNTRY : INDIA" if place.kind == "india" else place.name if place.kind == "region" else state_of(place)
    win = _overlap(ts, H.imd_coverage())
    if not key or not win:
        return None
    df = H.imd_series(key, *win)
    if df.empty:
        return None
    label = H.IMD_REGIONS.get(key, key)
    s = Series(label, "rainfall", "IMD state-wise daily rainfall (observed)", "imd", "mm", list(df["date"]),
               list(df["daily_actual_mm"]), state_mean=True)
    s.extra = {"normal": list(df["daily_normal_mm"]), "departure": list(df["daily_departure_pct"]),
               "category": list(df["daily_category"]), "rows": df}
    if place.kind == "city":
        s.note = f"IMD publishes state values; {place.name} is in {place.state}, so this is the {place.state} state average."
    if (win[0], win[1]) != (ts.start, ts.end):
        s.note += f" IMD data cover {win[0]} to {win[1]} of the requested {ts.start} to {ts.end}."
    return s


def replay_series(place: Place, ts: TimeSpec) -> Series | None:
    key = state_of(place)
    win = _overlap(ts, H.replay_coverage())
    if not key or not win:
        return None
    df = H.replay_series(key, *win)
    if df.empty:
        return None
    s = Series(key, "rainfall", "ERA5 reanalysis state-mean rainfall (HYDRA replay)", "replay", "mm", list(df["date"]),
               list(df["actual_mm"]), state_mean=True)
    s.extra = {"rows": df}
    if place.kind == "city":
        s.note = f"ERA5 values here are state averages; {place.name} is in {key}."
    if (win[0], win[1]) != (ts.start, ts.end):
        s.note += f" The replay covers {win[0]} to {win[1]} of the requested period."
    return s


def live_series(place: Place, var: str, ts: TimeSpec) -> Series:
    point = point_of(place)
    start, end = ts.start or today(), ts.end or today()
    result = LC.daily([point], [var], start, end)
    unit = UNITS.get(var, "")
    if result.get("status") != "available":
        return Series(point[0], var, "Open-Meteo", "live", unit, status="unavailable", message=result.get("message", ""))
    p = result["points"][0]
    agg = p["aggregate"]
    primary = next(iter(agg))
    dates = [date.fromisoformat(str(d)[:10]) for d in p["dates"]]
    s = Series(point[0], var, result["provider"], "live", agg[primary]["unit"], dates, p["series"][primary], state_mean=False)
    s.extra = {"aggregate": agg, "series": p["series"], "point": point}
    where = (f"the city point {place.name}" if place.kind == "city" else f"a representative point near the centre of {place.name}")
    s.note = f"Open-Meteo values at {where} ({point[1]:.2f}° N, {point[2]:.2f}° E), not a state-wide average."
    return s


def ecmwf_value(place: Place, var: str) -> dict | None:
    field_unit = H.ECMWF_FIELDS.get(var)
    if not field_unit:
        return None
    fld, unit = field_unit
    if place.kind == "city":
        row = H.ecmwf_point(place.lat, place.lon)
        if not row:
            return None
        return {"value": row[fld], "unit": unit, "kind": "grid point", "date": H.ecmwf_date(), "row": row}
    table = H.ecmwf_states()
    if table.empty:
        return None
    if place.kind == "india":
        return {"value": float(table[f"{fld}_mean"].mean()), "unit": unit, "kind": "mean of state means", "date": H.ecmwf_date(),
                "max": float(table[f"{fld}_max"].max()), "min": float(table[f"{fld}_min"].min())}
    row = table[table["place"] == place.state]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"value": float(r[f"{fld}_mean"]), "max": float(r[f"{fld}_max"]), "min": float(r[f"{fld}_min"]), "unit": unit,
            "kind": "state mean over ECMWF grid cells", "cells": int(r["cells"]), "date": H.ecmwf_date()}


def series(place: Place, var: str, ts: TimeSpec, prefer: list[str] | None = None) -> Series:
    """Best daily series for any place/variable/period."""
    prefer = prefer or []
    if var == "rainfall":
        candidates = []
        if "era5" in prefer or "hydra" in prefer:
            candidates = [replay_series, imd_series]
        else:
            candidates = [imd_series, replay_series]
        for fn in candidates:
            s = fn(place, ts)
            if s and s.ok:
                return s
    if var in ("temperature", "rainfall", "wind") and place.kind != "region":
        s = station_series(place, var, ts) if "meteostat" in prefer else None
        if s and s.ok:
            return s
    if place.kind in ("india", "region"):
        return Series(place.name, var, "", "none", UNITS.get(var, ""), status="unavailable",
                      message="All-India and regional values are available for IMD rainfall only; ask for a state or a ranking instead.")
    return live_series(place, var, ts)


def station_series(place: Place, var: str, ts: TimeSpec) -> Series | None:
    df = H.station()
    win = _overlap(ts, (df["date"].min(), df["date"].max()) if not df.empty else None)
    if not win:
        return None
    d = df[(df["date"] >= win[0]) & (df["date"] <= win[1])]
    col = {"temperature": "temp_avg_C", "rainfall": "precip_mm", "wind": "wind_speed_kmh"}[var]
    unit = {"temperature": "°C", "rainfall": "mm", "wind": "km/h"}[var]
    s = Series("Meteostat station", var, "Meteostat station export (station not identified)", "station", unit,
               list(d["date"]), list(d[col]), state_mean=False)
    s.note = "Single unidentified station; not specific to the place you asked about."
    return s


def hydra_forecast(state: str, valid: date) -> dict | None:
    st = H.hydra_state(state)
    if not st:
        return None
    for outlook in st["outlooks"]:
        if outlook["valid_date"] == valid.isoformat():
            return outlook
    return None

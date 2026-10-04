"""Live and historical weather for any Indian place via Open-Meteo (needs internet on the server).

Forecast API: today up to 16 days ahead and the last ~90 days.  Archive API: 1940 up to ~5 days ago.
Every failure returns {"status": "unavailable", ...}; callers then fall back to offline sources.
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Any

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = float(os.getenv("WEATHERGPT_LIVE_TIMEOUT", "12"))
_CACHE: dict[str, tuple[float, Any]] = {}
CACHE_SECONDS = 1800

# variable -> (daily field, aggregate over days, label, unit)
DAILY = {
    "rainfall": [("precipitation_sum", "sum", "rainfall", "mm"), ("precipitation_probability_max", "max", "maximum rain probability", "%")],
    "temperature": [("temperature_2m_max", "max", "maximum temperature", "°C"), ("temperature_2m_min", "min", "minimum temperature", "°C"),
                    ("temperature_2m_mean", "mean", "mean temperature", "°C")],
    "humidity": [("relative_humidity_2m_mean", "mean", "mean relative humidity", "%")],
    "wind": [("wind_speed_10m_max", "max", "maximum wind speed", "km/h")],
    "wind_gust": [("wind_gusts_10m_max", "max", "peak wind gust", "km/h")],
    "cloud": [("cloud_cover_mean", "mean", "mean cloud cover", "%")],
    "pressure": [("pressure_msl_mean", "mean", "mean sea-level pressure", "hPa")],
    "solar": [("shortwave_radiation_sum", "sum", "solar radiation", "MJ/m²")],
    "dew_point": [("dew_point_2m_mean", "mean", "mean dew point", "°C")],
    "cape": [("cape_max", "max", "maximum CAPE", "J/kg")],
    "thunderstorm": [("cape_max", "max", "maximum CAPE", "J/kg"), ("precipitation_probability_max", "max", "maximum rain probability", "%")],
    "heat_stress": [("apparent_temperature_max", "max", "maximum feels-like temperature", "°C")],
    "soil_moisture": [("soil_moisture_0_to_7cm_mean", "mean", "soil moisture (0-7 cm)", "m³/m³")],
    "water_vapour": [("dew_point_2m_mean", "mean", "mean dew point", "°C")],
}
FORECAST_ONLY = {"precipitation_probability_max", "cape_max"}


def _get(url: str, params: dict) -> Any:
    import httpx

    key = url + str(sorted(params.items()))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    _CACHE[key] = (time.time(), payload)
    return payload


def _agg(values: list[float], how: str) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"sum": sum, "max": max, "min": min}.get(how, lambda v: sum(v) / len(v))(values)


def _fetch(url: str, points, fields, start: date, end: date) -> list[dict]:
    params = {"latitude": ",".join(f"{p[1]:.3f}" for p in points), "longitude": ",".join(f"{p[2]:.3f}" for p in points),
              "daily": ",".join(f[0] for f in fields), "start_date": start.isoformat(), "end_date": end.isoformat(),
              "timezone": "Asia/Kolkata"}
    try:
        payload = _get(url, params)
    except Exception:
        core = [f for f in fields if f[0] in ("precipitation_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max")]
        if not core or core == fields:
            raise
        params["daily"] = ",".join(f[0] for f in core)
        payload = _get(url, params)
    return payload if isinstance(payload, list) else [payload]


def daily(points: list[tuple[str, float, float]], variables: list[str], start: date, end: date) -> dict:
    """Daily values for one or more points; long histories come from the archive, recent days from the forecast API."""
    from .timeparse import today
    ref = today()
    fields = []
    for var in variables or ["rainfall", "temperature"]:
        for spec in DAILY.get(var, []):
            if spec not in fields:
                fields.append(spec)
    if end > ref + timedelta(days=15):
        return {"status": "unavailable", "message": "Open-Meteo forecasts reach 16 days ahead; that date is beyond the forecast horizon."}
    archive_end = min(end, ref - timedelta(days=6))
    parts = []
    try:
        if start < ref - timedelta(days=80):
            arch_fields = [f for f in fields if f[0] not in FORECAST_ONLY] or [("precipitation_sum", "sum", "rainfall", "mm")]
            parts.append(("Open-Meteo Historical Weather API (ERA5-based)", arch_fields, _fetch(ARCHIVE_URL, points, arch_fields, start, archive_end)))
            if end > archive_end:
                parts.append(("Open-Meteo Forecast API", arch_fields, _fetch(FORECAST_URL, points, arch_fields, archive_end + timedelta(days=1), end)))
        else:
            parts.append(("Open-Meteo Forecast API", fields, _fetch(FORECAST_URL, points, fields, start, end)))
    except Exception as exc:
        return {"status": "unavailable", "message": f"The live weather provider could not be reached ({type(exc).__name__})."}
    used_fields = parts[0][1]
    out = []
    for k, point in enumerate(points):
        dates, series = [], {fld[0]: [] for fld in used_fields}
        for _, flds, reports in parts:
            d = (reports[k] if k < len(reports) else {}).get("daily") or {}
            dates += d.get("time", [])
            for fld in used_fields:
                series[fld[0]] += d.get(fld[0], [None] * len(d.get("time", [])))
        series = {name: vals for name, vals in series.items()}
        out.append({"name": point[0], "latitude": point[1], "longitude": point[2], "dates": dates, "series": series,
                    "aggregate": {f[0]: {"value": _agg(series[f[0]], f[1]), "how": f[1], "label": f[2], "unit": f[3]} for f in used_fields}})
    return {"status": "available", "points": out, "provider": " + ".join(dict.fromkeys(p[0] for p in parts)),
            "start": start.isoformat(), "end": end.isoformat()}


def current(name: str, lat: float, lon: float, state: str | None = None) -> dict:
    """Current conditions via the prototype's own live adapter (IMD via IndianAPI first, Open-Meteo fallback)."""
    try:
        from backend import live_weather
        report = live_weather.current_report(state or name, lat, lon)
    except Exception as exc:
        return {"status": "unavailable", "message": f"Live weather adapter failed ({type(exc).__name__})."}
    return report


def state_point(state: str) -> tuple[str, float, float]:
    from backend import data as D
    try:
        from backend.live_weather import STATE_CITIES
    except Exception:
        STATE_CITIES = {}
    feature = D.state_feature(state)
    center = D.geometry_center(feature["geometry"]) if feature else {"latitude": 22.0, "longitude": 79.0}
    return (state, float(center["latitude"]), float(center["longitude"]))

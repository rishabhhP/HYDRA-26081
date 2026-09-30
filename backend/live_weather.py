"""Live Indian weather reports from IndianAPI and Open-Meteo.

IndianAPI supplies IMD-backed city reports when it is configured. Open-Meteo
fills fields missing from those reports and provides a clearly labelled model
fallback when IndianAPI is unavailable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timedelta, timezone
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import httpx


BASE_URL = "https://weather.indianapi.in"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CACHE_MINUTES = max(5, int(os.getenv("INDIAN_WEATHER_CACHE_MINUTES", "30")))
ROOT = Path(__file__).resolve().parents[1]
DAILY_HISTORY_PATH = ROOT / "runtime" / "live_weather_daily.csv"
OPEN_METEO_CURRENT = ",".join((
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain",
    "weather_code", "cloud_cover", "surface_pressure", "wind_speed_10m",
    "wind_direction_10m",
))
OPEN_METEO_HOURLY = ",".join((
    "temperature_2m", "relative_humidity_2m", "precipitation", "precipitation_probability", "cape",
    "weather_code", "wind_speed_10m", "wind_gusts_10m", "wet_bulb_temperature_2m", "soil_moisture_0_to_1cm",
))

# The map uses distributed representative cities, not a continuous observation
# field. Open-Meteo requests them in one batch request.
STATIONS = (
    ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
    ("Shimla", "Himachal Pradesh", 31.1048, 77.1734),
    ("New Delhi", "Delhi", 28.6139, 77.2090),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Guwahati", "Assam", 26.1445, 91.7362),
    ("Patna", "Bihar", 25.5941, 85.1376),
    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Bhubaneswar", "Odisha", 20.2961, 85.8245),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Bengaluru", "Karnataka", 12.9716, 77.5946),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
)

STATE_CITIES = {
    "Andaman and Nicobar Islands":"Port Blair", "Andhra Pradesh":"Visakhapatnam", "Arunachal Pradesh":"Itanagar", "Assam":"Guwahati", "Bihar":"Patna", "Chandigarh":"Chandigarh", "Chhattisgarh":"Raipur", "Dadra and Nagar Haveli and Daman and Diu":"Daman", "Delhi":"New Delhi", "Goa":"Panaji", "Gujarat":"Ahmedabad", "Haryana":"Chandigarh", "Himachal Pradesh":"Shimla", "Jammu and Kashmir":"Jammu", "Jharkhand":"Ranchi", "Karnataka":"Bengaluru", "Kerala":"Thiruvananthapuram", "Ladakh":"Leh", "Lakshadweep":"Kavaratti", "Madhya Pradesh":"Bhopal", "Maharashtra":"Mumbai", "Manipur":"Imphal", "Meghalaya":"Shillong", "Mizoram":"Aizawl", "Nagaland":"Kohima", "Odisha":"Bhubaneswar", "Puducherry":"Puducherry", "Punjab":"Chandigarh", "Rajasthan":"Jaipur", "Sikkim":"Gangtok", "Tamil Nadu":"Chennai", "Telangana":"Hyderabad", "Tripura":"Agartala", "Uttar Pradesh":"Lucknow", "Uttarakhand":"Dehradun", "West Bengal":"Kolkata",
}

# Values exposed on HYDRA's live map layers. They remain city samples, not a
# continuous India-wide field, and retain the provider metadata in responses.
LIVE_LAYER_FIELDS = {
    "live_weather": ("temperature_max_C", "current temperature", "°C"),
    "open_meteo_wind_speed": ("wind_speed_kmh", "current 10 m wind speed", "km/h"),
    "open_meteo_rainfall": ("open_meteo_rainfall_mm", "current rainfall", "mm"),
    "open_meteo_rainfall_24h": ("open_meteo_rainfall_24h_mm", "next 24-hour rainfall", "mm"),
    "open_meteo_humidity": ("open_meteo_humidity_percent", "relative humidity", "%"),
    "open_meteo_lightning": ("open_meteo_lightning_watch_percent", "next 24-hour lightning watch", "%"),
    "open_meteo_thunderstorm": ("open_meteo_cape_max_Jkg", "next 24-hour CAPE", "J/kg"),
    "open_meteo_wind_gust": ("open_meteo_wind_gust_max_kmh", "next 24-hour peak wind gust", "km/h"),
    "open_meteo_heat_stress": ("open_meteo_wet_bulb_max_C", "next 24-hour peak wet-bulb temperature", "°C"),
    "open_meteo_soil_moisture": ("open_meteo_soil_moisture_0_to_1cm", "surface soil moisture (0–1 cm)", "m³/m³"),
}

_lock = Lock()
_cache: dict[str, Any] | None = None
_cached_at: datetime | None = None
_current_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_geocode_cache: dict[str, dict[str, Any] | None] = {}


def _record_daily_samples(stations: list[dict[str, Any]], source: str, observed_at: datetime) -> None:
    """Upsert one durable live-layer sample per UTC day and provider location."""
    if not stations:
        return
    DAILY_HISTORY_PATH.parent.mkdir(exist_ok=True)
    fields = ("date", "observed_at", "city", "state", "latitude", "longitude", "source", *[
        field for field, _, _ in LIVE_LAYER_FIELDS.values()
    ])
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if DAILY_HISTORY_PATH.exists():
        with DAILY_HISTORY_PATH.open(encoding="utf-8", newline="") as stream:
            existing = {(row["date"], row["city"]): row for row in csv.DictReader(stream)}
    day = observed_at.date().isoformat()
    for station in stations:
        row = {field:"" for field in fields}
        row.update({"date":day, "observed_at":observed_at.isoformat(), "city":str(station["city"]),
                    "state":str(station["state"]), "latitude":str(station["latitude"]),
                    "longitude":str(station["longitude"]), "source":source})
        for field, _, _ in LIVE_LAYER_FIELDS.values():
            value = station.get(field)
            row[field] = "" if value is None else str(value)
        existing[(day, row["city"])] = row
    temporary = DAILY_HISTORY_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing[key] for key in sorted(existing))
    os.replace(temporary, DAILY_HISTORY_PATH)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wmo_description(code: Any) -> str | None:
    descriptions = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Rime fog",51:"Light drizzle",53:"Moderate drizzle",55:"Dense drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",80:"Rain showers",81:"Moderate showers",82:"Violent showers",95:"Thunderstorm",96:"Thunderstorm with hail",97:"Heavy thunderstorm",99:"Heavy thunderstorm with hail"}
    return descriptions.get(int(code)) if isinstance(code, int | float) else None


def _series(hourly: dict[str, Any], name: str, start: int = 0, end: int | None = None) -> list[float]:
    return [value for value in (_number(item) for item in hourly.get(name, [])[start:end]) if value is not None]


def _outlook_window(hourly: dict[str, Any], start: int, end: int, lead_hours: int) -> dict[str, float | int | None]:
    """Summarise a defined future hourly window without inventing an NWP blend."""
    temperature = _series(hourly, "temperature_2m", start, end)
    rainfall = _series(hourly, "precipitation", start, end)
    humidity = _series(hourly, "relative_humidity_2m", start, end)
    probability = _series(hourly, "precipitation_probability", start, end)
    gusts = _series(hourly, "wind_gusts_10m", start, end)
    return {
        "lead_hours": lead_hours,
        "temperature_mean_C": round(sum(temperature) / len(temperature), 1) if temperature else None,
        "rainfall_total_mm": round(sum(rainfall), 2) if rainfall else None,
        "humidity_mean_percent": round(sum(humidity) / len(humidity), 1) if humidity else None,
        "precipitation_probability_max_percent": max(probability) if probability else None,
        "wind_gust_max_kmh": max(gusts) if gusts else None,
    }


def _forecast_summary(hourly: dict[str, Any]) -> dict[str, float | None]:
    rainfall = _series(hourly, "precipitation", 0, 24)
    probability = _series(hourly, "precipitation_probability", 0, 24)
    cape = _series(hourly, "cape", 0, 24)
    weather_codes = _series(hourly, "weather_code", 0, 24)
    gusts = _series(hourly, "wind_gusts_10m", 0, 24)
    wet_bulb = _series(hourly, "wet_bulb_temperature_2m", 0, 24)
    soil_moisture = _series(hourly, "soil_moisture_0_to_1cm", 0, 24)
    lightning_scores = []
    for index in range(max(len(cape), len(probability), len(weather_codes))):
        instability = cape[index] if index < len(cape) else 0.0
        rain_chance = probability[index] if index < len(probability) else 0.0
        code = weather_codes[index] if index < len(weather_codes) else 0.0
        # Open-Meteo's generic India endpoint does not return a lightning-strike
        # field. This is a transparent watch indicator from provider CAPE,
        # precipitation probability, and WMO thunderstorm codes.
        score = 100.0 if code >= 95 else min(90.0, instability / 30.0 + rain_chance * 0.45)
        lightning_scores.append(score)
    return {
        "open_meteo_rainfall_24h_mm": round(sum(rainfall), 2) if rainfall else None,
        "open_meteo_precipitation_probability_max": max(probability) if probability else None,
        "open_meteo_cape_max_Jkg": max(cape) if cape else None,
        "open_meteo_lightning_watch_percent": round(max(lightning_scores), 1) if lightning_scores else None,
        "open_meteo_wind_gust_max_kmh": max(gusts) if gusts else None,
        "open_meteo_wet_bulb_max_C": max(wet_bulb) if wet_bulb else None,
        "open_meteo_soil_moisture_0_to_1cm": soil_moisture[0] if soil_moisture else None,
    }


def _station_record(city: str, state: str, latitude: float, longitude: float, *, provider: str, current: dict[str, Any], hourly: dict[str, Any]) -> dict[str, Any]:
    temperature = _number(current.get("temperature_2m"))
    rainfall = _number(current.get("rain"))
    rainfall = rainfall if rainfall is not None else _number(current.get("precipitation"))
    humidity = _number(current.get("relative_humidity_2m"))
    return {
        "city": city, "state": state, "latitude": latitude, "longitude": longitude,
        "temperature_max_C": temperature, "temperature_min_C": temperature,
        "rainfall_mm": rainfall, "open_meteo_rainfall_mm": rainfall,
        "humidity_percent": humidity, "open_meteo_humidity_percent": humidity,
        "description": _wmo_description(current.get("weather_code")),
        "forecast_date": current.get("time"), "wind_speed_kmh": _number(current.get("wind_speed_10m")),
        "wind_direction_deg": _number(current.get("wind_direction_10m")),
        "cloud_cover_percent": _number(current.get("cloud_cover")),
        "pressure_hpa": _number(current.get("surface_pressure")), "provider": provider,
        **_forecast_summary(hourly), "outlook_24h": _outlook_window(hourly, 0, 24, 24),
        "outlook_48h": _outlook_window(hourly, 24, 48, 48),
    }


def fetch_open_meteo_stations(stations: tuple[tuple[str, str, float, float], ...] | list[tuple[str, str, float, float]]) -> list[dict[str, Any]]:
    """Fetch current model values for many locations in one Open-Meteo request."""
    if not stations:
        return []
    params = {
        "latitude": ",".join(str(item[2]) for item in stations),
        "longitude": ",".join(str(item[3]) for item in stations),
        "current": OPEN_METEO_CURRENT, "hourly": OPEN_METEO_HOURLY, "forecast_hours": 48,
        "timezone": "GMT", "wind_speed_unit": "kmh",
    }
    with httpx.Client(timeout=12.0) as client:
        response = client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    reports = payload if isinstance(payload, list) else [payload]
    return [_station_record(city, state, latitude, longitude, provider="Open-Meteo weather model", current=report.get("current", {}), hourly=report.get("hourly", {})) for (city, state, latitude, longitude), report in zip(stations, reports)]


def fetch_station(client: httpx.Client, station: tuple[str, str, float, float]) -> dict[str, Any] | None:
    city, state, latitude, longitude = station
    response = client.get("/india/weather", params={"city": city})
    response.raise_for_status()
    payload = response.json()
    weather = payload.get("weather", {})
    current = weather.get("current", {})
    temperature = current.get("temperature", {})
    humidity = current.get("humidity", {})
    humidity_values = [value for value in (_number(humidity.get(period)) for period in ("morning", "evening")) if value is not None]
    forecast = weather.get("forecast", [])
    first_forecast = forecast[0] if forecast else {}
    return {
        "city": payload.get("city") or city, "state": state, "latitude": latitude, "longitude": longitude,
        "temperature_max_C": _number(temperature.get("max", {}).get("value")),
        "temperature_min_C": _number(temperature.get("min", {}).get("value")),
        "rainfall_mm": _number(current.get("rainfall")),
        "humidity_percent": round(sum(humidity_values) / len(humidity_values), 1) if humidity_values else None,
        "description": first_forecast.get("description"), "forecast_date": first_forecast.get("date"),
        "wind_speed_kmh": None, "wind_direction_deg": None, "cloud_cover_percent": None,
        "pressure_hpa": None, "provider": "IndianAPI / IMD city report",
    }


def _merge_station(imd: dict[str, Any], model: dict[str, Any] | None) -> dict[str, Any]:
    if not model:
        return imd
    merged = {**imd}
    for key in ("open_meteo_rainfall_mm", "open_meteo_humidity_percent", "open_meteo_rainfall_24h_mm", "open_meteo_precipitation_probability_max", "open_meteo_cape_max_Jkg", "open_meteo_lightning_watch_percent", "open_meteo_wind_gust_max_kmh", "open_meteo_wet_bulb_max_C", "open_meteo_soil_moisture_0_to_1cm", "outlook_24h", "outlook_48h"):
        merged[key] = model.get(key)
    added = False
    for key in ("rainfall_mm", "humidity_percent", "wind_speed_kmh", "wind_direction_deg", "cloud_cover_percent", "pressure_hpa", "description", "forecast_date"):
        if merged.get(key) is None and model.get(key) is not None:
            merged[key] = model[key]
            added = True
    if merged.get("temperature_max_C") is None and model.get("temperature_max_C") is not None:
        merged["temperature_max_C"] = model["temperature_max_C"]
        merged["temperature_min_C"] = model["temperature_min_C"]
        added = True
    if added:
        merged["provider"] = "IndianAPI / IMD city report + Open-Meteo enrichment"
    return merged


def _imd_reports(api_key: str | None, stations: tuple[tuple[str, str, float, float], ...] | list[tuple[str, str, float, float]]) -> list[dict[str, Any]]:
    if not api_key:
        return []
    reports: list[dict[str, Any]] = []
    with httpx.Client(base_url=BASE_URL, headers={"x-api-key": api_key, "accept": "application/json"}, timeout=12.0) as client:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch_station, client, station) for station in stations]
            for future in as_completed(futures):
                try:
                    report = future.result()
                    if report:
                        reports.append(report)
                except httpx.HTTPError:
                    continue
    return reports


def _fetch() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    imd = _imd_reports(os.getenv("INDIAN_WEATHER_API_KEY"), STATIONS)
    try:
        model = fetch_open_meteo_stations(STATIONS)
    except httpx.HTTPError:
        model = []
    model_by_city = {item["city"]: item for item in model}
    imd_by_city = {item["city"]: item for item in imd}
    stations = [_merge_station(report, model_by_city.get(city)) for city, report in imd_by_city.items()]
    stations.extend(report for city, report in model_by_city.items() if city not in imd_by_city)
    if not stations:
        return {"status":"unavailable", "message":"Live Indian weather providers did not return a report.", "stations":[], "source":"IndianAPI / IMD city reports and Open-Meteo weather model", "updated_at":now.isoformat()}
    source = "IndianAPI / IMD city reports + Open-Meteo enrichment" if imd else "Open-Meteo current weather model"
    _record_daily_samples(stations, source, now)
    return {"status":"available", "message":"Current weather locations. IMD reports are enriched with Open-Meteo when fields are absent; Open-Meteo-only markers are model values, not station observations.", "stations":sorted(stations, key=lambda item:item["city"]), "source":source, "updated_at":now.isoformat()}


def station_reports() -> dict[str, Any]:
    global _cache, _cached_at
    now = datetime.now(timezone.utc)
    with _lock:
        if _cache is not None and _cached_at is not None and now - _cached_at < timedelta(minutes=CACHE_MINUTES):
            return {**_cache, "cached":True, "cache_minutes":CACHE_MINUTES}
        _cache = _fetch()
        _cached_at = now
        return {**_cache, "cached":False, "cache_minutes":CACHE_MINUTES}


def rank_layer(layer: str, state: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Rank available live-map samples for a WeatherGPT layer question."""
    specification = LIVE_LAYER_FIELDS.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"This live map layer has no numeric query value.", "items":[]}
    field, label, unit = specification
    reports = station_reports()
    if reports["status"] != "available":
        return {"status":"unavailable", "message":reports["message"], "items":[]}
    items = []
    for station in reports["stations"]:
        if state and station.get("state", "").casefold() != state.casefold():
            continue
        value = _number(station.get(field))
        if value is not None:
            items.append({"city":station["city"], "state":station["state"], "latitude":station["latitude"],
                          "longitude":station["longitude"], "value":value, "provider":station.get("provider")})
    items.sort(key=lambda item:item["value"], reverse=True)
    return {"status":"available", "items":items[:limit], "sample_count":len(items), "label":label, "unit":unit,
            "source":reports["source"], "updated_at":reports.get("updated_at"), "cached":reports.get("cached")}


def rank_state_layer(layer: str, limit: int = 5) -> dict[str, Any]:
    """Rank one Open-Meteo representative coordinate for every India state/UT.

    Results are representative-point comparisons. They are not spatially
    averaged statewide observations and are labelled accordingly in WeatherGPT.
    """
    specification = LIVE_LAYER_FIELDS.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"This live map layer has no numeric query value.", "items":[]}
    from . import data as D
    representatives: list[tuple[str, str, float, float]] = []
    for feature in D.state_features():
        state = feature["properties"]["name"]
        center = D.geometry_center(feature["geometry"])
        representatives.append((state, state, float(center["latitude"]), float(center["longitude"])))
    try:
        reports = fetch_open_meteo_stations(tuple(representatives))
    except httpx.HTTPError:
        return {"status":"unavailable", "message":"Open-Meteo did not return state representative values.", "items":[]}
    field, label, unit = specification
    items = []
    for report in reports:
        value = _number(report.get(field))
        if value is not None:
            items.append({"state": report["state"], "city": "state representative point",
                          "latitude": report["latitude"], "longitude": report["longitude"],
                          "value": value, "provider": report.get("provider")})
    items.sort(key=lambda item: item["value"], reverse=True)
    return {"status":"available", "items":items[:limit], "sample_count":len(items), "label":label,
            "unit":unit, "source":"Open-Meteo weather model", "updated_at":datetime.now(timezone.utc).isoformat(),
            "coverage":"one representative coordinate per India state/UT"}


def rank_state_layer_for_day(layer: str, target_date: date, limit: int = 5) -> dict[str, Any]:
    """Rank a completed UTC day using Open-Meteo's past hourly model series."""
    specification = LIVE_LAYER_FIELDS.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"This live map layer has no numeric query value.", "items":[]}
    from . import data as D
    representatives: list[tuple[str, str, float, float]] = []
    for feature in D.state_features():
        state = feature["properties"]["name"]
        center = D.geometry_center(feature["geometry"])
        representatives.append((state, state, float(center["latitude"]), float(center["longitude"])))
    params = {
        "latitude": ",".join(str(item[2]) for item in representatives),
        "longitude": ",".join(str(item[3]) for item in representatives),
        "hourly": OPEN_METEO_HOURLY, "past_days": 2, "forecast_days": 1,
        "timezone": "GMT", "wind_speed_unit": "kmh",
    }
    try:
        with httpx.Client(timeout=14.0) as client:
            response = client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError:
        return {"status":"unavailable", "message":"Open-Meteo did not return the requested past-day values.", "items":[]}
    reports = payload if isinstance(payload, list) else [payload]
    field, _, unit = specification
    field_map = {
        "temperature_max_C": ("temperature_2m", max, "daily maximum temperature"),
        "wind_speed_kmh": ("wind_speed_10m", max, "daily maximum 10 m wind speed"),
        "open_meteo_rainfall_mm": ("precipitation", sum, "daily rainfall"),
        "open_meteo_humidity_percent": ("relative_humidity_2m", lambda values: sum(values) / len(values), "daily mean relative humidity"),
        "open_meteo_cape_max_Jkg": ("cape", max, "daily maximum CAPE"),
        "open_meteo_wind_gust_max_kmh": ("wind_gusts_10m", max, "daily maximum wind gust"),
        "open_meteo_wet_bulb_max_C": ("wet_bulb_temperature_2m", max, "daily maximum wet-bulb temperature"),
        "open_meteo_soil_moisture_0_to_1cm": ("soil_moisture_0_to_1cm", lambda values: sum(values) / len(values), "daily mean surface soil moisture"),
    }
    if field == "open_meteo_lightning_watch_percent":
        source_field, aggregate, label = None, None, "daily maximum lightning watch"
    else:
        source_field, aggregate, label = field_map.get(field, (None, None, None))
    if source_field is None and field != "open_meteo_lightning_watch_percent":
        return {"status":"unavailable", "message":"This live layer has no past-day aggregation.", "items":[]}
    items = []
    date_text = target_date.isoformat()
    for station, report in zip(representatives, reports):
        hourly = report.get("hourly") or {}
        positions = [index for index, value in enumerate(hourly.get("time") or []) if str(value).startswith(date_text)]
        if not positions:
            continue
        if field == "open_meteo_lightning_watch_percent":
            summary = _forecast_summary({
                key: [((hourly.get(key) or [None] * (max(positions) + 1))[index]) for index in positions]
                for key in ("cape", "precipitation_probability", "weather_code")
            })
            value = _number(summary.get(field))
        else:
            values = [_number((hourly.get(source_field) or [None] * (max(positions) + 1))[index]) for index in positions]
            values = [value for value in values if value is not None]
            value = aggregate(values) if values else None
        if value is not None:
            items.append({"state":station[1], "city":"state representative point", "latitude":station[2],
                          "longitude":station[3], "value":float(value), "provider":"Open-Meteo weather model"})
    items.sort(key=lambda item:item["value"], reverse=True)
    if not items:
        return {"status":"unavailable", "message":f"Open-Meteo returned no {target_date.isoformat()} values for this layer.", "items":[]}
    return {"status":"available", "items":items[:limit], "sample_count":len(items), "label":label,
            "unit":unit, "source":"Open-Meteo past-day weather model series", "date":date_text,
            "coverage":"one representative coordinate per India state/UT"}


def _historical_window_layer(
    layer: str,
    stations: tuple[tuple[str, str, float, float], ...] | list[tuple[str, str, float, float]],
    start_date: date,
    end_date: date,
    limit: int = 5,
) -> dict[str, Any]:
    """Aggregate exact historical dates from Open-Meteo's archive API."""
    daily_specifications = {
        "live_weather": ("temperature_2m_max", max, "maximum daily temperature", "°C"),
        "open_meteo_rainfall": ("precipitation_sum", sum, "accumulated rainfall", "mm"),
    }
    specification = daily_specifications.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"Exact date windows are currently available for temperature and rainfall.", "items":[]}
    variable, aggregate, label, unit = specification
    params = {
        "latitude": ",".join(str(item[2]) for item in stations),
        "longitude": ",".join(str(item[3]) for item in stations),
        "daily": variable, "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "timezone": "GMT", "wind_speed_unit": "kmh",
    }
    try:
        with httpx.Client(timeout=18.0) as client:
            response = client.get(OPEN_METEO_ARCHIVE_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError:
        return {"status":"unavailable", "message":"Open-Meteo did not return the requested historical date window.", "items":[]}
    reports = payload if isinstance(payload, list) else [payload]
    items = []
    for station, report in zip(stations, reports):
        values = [_number(value) for value in (report.get("daily") or {}).get(variable, [])]
        values = [value for value in values if value is not None]
        if values:
            items.append({"city":station[0], "state":station[1], "latitude":station[2], "longitude":station[3],
                          "value":float(aggregate(values)), "sample_days":len(values), "provider":"Open-Meteo Historical Weather API"})
    items.sort(key=lambda item:item["value"], reverse=True)
    if not items:
        return {"status":"unavailable", "message":f"Open-Meteo returned no usable values from {start_date} to {end_date}.", "items":[]}
    return {
        "status":"available", "items":items[:limit], "sample_count":len(items), "label":label, "unit":unit,
        "window_start":start_date.isoformat(), "window_end":end_date.isoformat(),
        "source":"Open-Meteo Historical Weather API", "calculation":"maximum daily value" if layer == "live_weather" else "sum of daily precipitation",
    }


def rank_state_layer_for_window(layer: str, start_date: date, end_date: date, limit: int = 5) -> dict[str, Any]:
    """Compare an exact historical temperature/rainfall window across state/UT points."""
    from . import data as D
    representatives = []
    for feature in D.state_features():
        state = feature["properties"]["name"]
        center = D.geometry_center(feature["geometry"])
        representatives.append((state, state, float(center["latitude"]), float(center["longitude"])))
    result = _historical_window_layer(layer, representatives, start_date, end_date, limit)
    if result["status"] == "available":
        result["coverage"] = "one representative coordinate per India state/UT"
    return result


def rank_location_layer_for_window(layer: str, start_date: date, end_date: date, state: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Compare the actual configured live map locations over a date window."""
    stations = tuple(item for item in STATIONS if not state or item[1].casefold() == state.casefold())
    return _historical_window_layer(layer, stations, start_date, end_date, limit)


def layer_value_for_window(layer: str, latitude: float, longitude: float, state: str | None, start_date: date, end_date: date) -> dict[str, Any]:
    """Get a calculated historical value for one parsed location/date window."""
    station = (state or "Selected location", state or "Selected location", latitude, longitude)
    result = _historical_window_layer(layer, (station,), start_date, end_date, limit=1)
    if result["status"] != "available":
        return result
    item = result["items"][0]
    return {**result, "value":item["value"], "observation":item}


def historical_rank_layer(layer: str, window_days: int, state: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Rank retained daily live samples; never fill missing dates with estimates."""
    specification = LIVE_LAYER_FIELDS.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"This live layer has no numeric timeline value.", "items":[]}
    if not DAILY_HISTORY_PATH.exists():
        return {"status":"unavailable", "message":"No daily live-layer history has been recorded yet. Enable a live layer and let HYDRA collect daily samples.", "items":[]}
    with DAILY_HISTORY_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = sorted({row["date"] for row in rows})
    if len(dates) < window_days:
        return {"status":"unavailable", "message":f"Only {len(dates)} recorded live-data day(s) are available; {window_days} are required for this timeline question.", "items":[]}
    selected_dates = set(dates[-window_days:])
    field, label, unit = specification
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        if row["date"] not in selected_dates or (state and row["state"].casefold() != state.casefold()):
            continue
        value = _number(row.get(field))
        if value is not None:
            grouped.setdefault((row["city"], row["state"], row["latitude"], row["longitude"]), []).append(value)
    items = [{"city":city, "state":area, "latitude":float(latitude), "longitude":float(longitude),
              "value":sum(values)/len(values), "sample_days":len(values)}
             for (city, area, latitude, longitude), values in grouped.items()]
    items.sort(key=lambda item:item["value"], reverse=True)
    return {"status":"available", "items":items[:limit], "sample_count":len(items), "label":label, "unit":unit,
            "window_days":window_days, "window_start":dates[-window_days], "window_end":dates[-1],
            "source":"Retained daily IndianAPI / Open-Meteo layer samples"}


def layer_value(layer: str, state: str | None, latitude: float, longitude: float, location_name: str | None = None) -> dict[str, Any]:
    """Get the live-map metric at the selected or parsed location."""
    specification = LIVE_LAYER_FIELDS.get(layer)
    if not specification:
        return {"status":"unavailable", "message":"This live map layer has no numeric query value."}
    field, label, unit = specification
    report = current_report(state or location_name, latitude, longitude)
    observation = report.get("observation")
    if report["status"] != "available" or not observation:
        return {"status":"unavailable", "message":report["message"]}
    value = _number(observation.get(field))
    if value is None:
        return {"status":"unavailable", "message":f"{label.title()} is unavailable from the current provider response."}
    return {"status":"available", "label":label, "unit":unit, "value":value, "observation":observation,
            "source":report["source"], "updated_at":report.get("updated_at"), "cached":report.get("cached")}


def clear_cache() -> None:
    global _cache, _cached_at
    with _lock:
        _cache = None
        _cached_at = None
        _current_cache.clear()
        _geocode_cache.clear()


def resolve_indian_location(query: str) -> dict[str, Any] | None:
    """Resolve an unmatched city only when Open-Meteo confirms it is in India."""
    key = query.casefold().strip()
    if not key:
        return None
    with _lock:
        if key in _geocode_cache:
            return _geocode_cache[key]
    with httpx.Client(timeout=8.0) as client:
        response = client.get(OPEN_METEO_GEOCODING_URL, params={"name":query, "count":5, "language":"en", "format":"json"})
        response.raise_for_status()
        candidates = response.json().get("results", [])
    match = next((item for item in candidates if item.get("country_code") == "IN" and isinstance(item.get("latitude"), (int, float)) and isinstance(item.get("longitude"), (int, float))), None)
    result = None if match is None else {"name":match.get("name") or query, "state":match.get("admin1"),
                                         "latitude":float(match["latitude"]), "longitude":float(match["longitude"])}
    with _lock:
        _geocode_cache[key] = result
    return result


def current_report(state: str | None, latitude: float, longitude: float) -> dict[str, Any]:
    """Current selected-state report with IMD first and Open-Meteo fallback."""
    state_name = state or "Selected location"
    city = STATE_CITIES.get(state_name, state_name)
    now = datetime.now(timezone.utc)
    cache_key = f"{state_name}:{latitude:.3f}:{longitude:.3f}"
    with _lock:
        cached = _current_cache.get(cache_key)
        if cached and now - cached[0] < timedelta(minutes=CACHE_MINUTES):
            return {**cached[1], "cached":True, "cache_minutes":CACHE_MINUTES}
    station = (city, state_name, latitude, longitude)
    imd_reports = _imd_reports(os.getenv("INDIAN_WEATHER_API_KEY"), (station,))
    try:
        model_reports = fetch_open_meteo_stations((station,))
    except httpx.HTTPError:
        model_reports = []
    observation = _merge_station(imd_reports[0], model_reports[0] if model_reports else None) if imd_reports else (model_reports[0] if model_reports else None)
    if not observation:
        return {"status":"unavailable", "message":"Live Indian weather providers did not return a current report.", "source":"IndianAPI / IMD city reports and Open-Meteo weather model"}
    payload = {"status":"available", "message":"Current location weather. IMD values are used when supplied; remaining fields may be Open-Meteo model values.", "source":observation["provider"], "updated_at":now.isoformat(), "observation":observation}
    with _lock:
        _current_cache[cache_key] = (now, payload)
    return {**payload, "cached":False, "cache_minutes":CACHE_MINUTES}


def state_outlook(state: str, latitude: float, longitude: float) -> dict[str, Any]:
    """Return the provider outlook used by the state AI Insights panel."""
    report = current_report(state, latitude, longitude)
    if report['status'] != 'available':
        return report
    observation = report['observation']
    outlook_24h = observation.get('outlook_24h')
    outlook_48h = observation.get('outlook_48h')
    if not isinstance(outlook_24h, dict) or not isinstance(outlook_48h, dict):
        return {'status':'unavailable', 'message':'The live provider did not return a complete +24h/+48h outlook.', 'state':state}
    return {
        'status':'available', 'state':state, 'city':observation['city'], 'latitude':observation['latitude'], 'longitude':observation['longitude'],
        'outlooks':[outlook_24h, outlook_48h], 'source':report['source'], 'updated_at':report.get('updated_at'),
        'message':'Open-Meteo provider outlook at the state representative location. +24h covers the next 24 hours; +48h covers hours 24-48.',
    }

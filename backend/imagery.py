"""Cached public imagery metadata and proxy helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import threading
import httpx

ROOT = Path(__file__).resolve().parents[1]
RAINVIEWER_URL = "https://api.rainviewer.com/public/weather-maps.json"
IMD_URL = "https://mausam.imd.gov.in/Satellite/rswmo_ir1.jpg"
_lock = threading.Lock()
_radar: tuple[datetime, dict] | None = None
_satellite: tuple[datetime, bytes, dict[str, str]] | None = None


def radar() -> dict:
    global _radar
    now = datetime.now(timezone.utc)
    with _lock:
        if _radar and now - _radar[0] < timedelta(minutes=5):
            return {**_radar[1], "cached": True}
    response = httpx.get(RAINVIEWER_URL, timeout=12)
    response.raise_for_status()
    payload = response.json()
    result = {"host": payload.get("host", "https://tilecache.rainviewer.com"), "radar": payload.get("radar", {}).get("past", []), "generated_at": now.isoformat()}
    with _lock:
        _radar = (now, result)
    return {**result, "cached": False}


def satellite() -> tuple[bytes, dict[str, str]]:
    global _satellite
    now = datetime.now(timezone.utc)
    with _lock:
        if _satellite and now - _satellite[0] < timedelta(minutes=15):
            return _satellite[1], _satellite[2]
    response = httpx.get(IMD_URL, timeout=20)
    response.raise_for_status()
    headers = {"Content-Type": response.headers.get("content-type", "image/jpeg"), "X-Hydra-Fetched-At": now.isoformat(), "X-Hydra-Source-Last-Modified": response.headers.get("last-modified", "unavailable")}
    with _lock:
        _satellite = (now, response.content, headers)
    return response.content, headers


def satellite_bounds() -> dict:
    return json.loads((ROOT / "backend" / "config" / "imd_satellite.json").read_text())

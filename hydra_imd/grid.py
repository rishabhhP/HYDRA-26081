"""Read IMD gridded rainfall into (dates, lat, lon, values) numpy arrays.

Uses imdlib when installed (handles archive and real-time files). Without imdlib, archive .grd files
(GrADS binary: float32, 135 lon x 129 lat per day, 66.5-100 E, 6.5-38.5 N, missing = -999) are read
directly, and the byte order / axis layout is verified with a land-sea check rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LAT = np.arange(6.5, 38.5 + 1e-9, 0.25)     # 129
LON = np.arange(66.5, 100.0 + 1e-9, 0.25)   # 135
MISSING = -999.0
LAND_POINT = (20.0, 78.0)                   # central India: always has data
SEA_POINT = (15.0, 68.0)                    # Arabian Sea: never has data


@dataclass
class ImdRain:
    dates: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    rain: np.ndarray                         # (days, lat, lon), NaN outside India / missing

    def window(self, start, end) -> "ImdRain":
        m = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return ImdRain(self.dates[m], self.lat, self.lon, self.rain[m])


def _index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def _orientation_ok(day: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> bool:
    land = day[_index(lat, LAND_POINT[0]), _index(lon, LAND_POINT[1])]
    sea = day[_index(lat, SEA_POINT[0]), _index(lon, SEA_POINT[1])]
    return np.isfinite(land) and not np.isfinite(sea)


def read_grd(path: Path, year: int) -> ImdRain:
    raw = np.fromfile(path, dtype="<f4")
    per_day = len(LAT) * len(LON)
    if raw.size % per_day:
        raise ValueError(f"{path.name}: size {raw.size} is not a whole number of {len(LAT)}x{len(LON)} days")
    n = raw.size // per_day
    raw = np.where(raw <= MISSING + 1, np.nan, raw)
    candidates = [raw.reshape(n, len(LAT), len(LON)),                              # GrADS: lon varies fastest
                  raw.reshape(n, len(LON), len(LAT)).transpose(0, 2, 1)]            # alternative layout
    probe = next((k for k in range(n) if np.isfinite(candidates[0][k]).any()), 0)
    for cube in candidates:
        if _orientation_ok(cube[probe], LAT, LON):
            dates = pd.date_range(f"{year}-01-01", periods=n, freq="D")
            return ImdRain(dates, LAT, LON, cube.astype(np.float32))
    raise ValueError(f"{path.name}: could not verify the grid layout (land/sea check failed)")


def _from_xarray(ds) -> ImdRain:
    var = "rain" if "rain" in ds else next(iter(ds.data_vars))
    da = ds[var].transpose("time", "lat", "lon")
    vals = da.values.astype(np.float32)
    vals = np.where(vals < 0, np.nan, vals)
    lat, lon = da["lat"].values, da["lon"].values
    if lat[0] > lat[-1]:
        lat, vals = lat[::-1], vals[:, ::-1, :]
    return ImdRain(pd.DatetimeIndex(da["time"].values).normalize(), lat, lon, vals)


def open_dir(directory: Path, years: list[int] | None = None, realtime: tuple[str, str] | None = None) -> ImdRain:
    """Open every archive year found (and optionally a real-time window) and stitch them in date order."""
    directory = Path(directory)
    parts: list[ImdRain] = []
    files = sorted(set(directory.rglob("*.grd")))
    found = {}
    for f in files:
        m = re.search(r"(19|20)\d\d", f.stem)
        if m and "real" not in str(f).lower():
            found.setdefault(int(m.group()), f)
    for year in sorted(found):
        if years and year not in years:
            continue
        try:
            import imdlib as imd
            parts.append(_from_xarray(imd.open_data("rain", year, year, "yearwise", str(directory)).get_xarray()))
        except ImportError:
            parts.append(read_grd(found[year], year))
    for nc in sorted(directory.rglob("*.nc")):
        import xarray as xr
        with xr.open_dataset(nc) as ds:
            parts.append(_from_xarray(ds.load()))
    if realtime:
        import imdlib as imd
        parts.append(_from_xarray(imd.open_real_data("rain", realtime[0], realtime[1], str(directory)).get_xarray()))
    if not parts:
        raise FileNotFoundError(f"No IMD rainfall files under {directory}")
    lat, lon = parts[0].lat, parts[0].lon
    for p in parts:
        if p.rain.shape[1:] != (len(lat), len(lon)):
            raise ValueError("IMD files have different grids; keep 0.25 degree rainfall only")
    dates = pd.DatetimeIndex(np.concatenate([p.dates.values for p in parts]))
    rain = np.concatenate([p.rain for p in parts])
    order = np.argsort(dates.values, kind="stable")
    dates, rain = dates[order], rain[order]
    keep = ~dates.duplicated(keep="last")
    return ImdRain(dates[keep], lat, lon, rain[keep])

"""Ingestion: raw files -> tidy tables.

* ``build_era5_daily``  rebuilds the daily ERA5 truth archive from the raw 6-hourly
  NetCDFs (the pre-processed ``era5_india_2025_daily_processed_csv.gz`` referenced in the
  brief is not present in the workspace; this reproduces its schema from source).
* loaders for the IMD statewise table, the Meteostat station and the ECMWF 0 h snapshot.
* ``grib_inventory``   pure-python GRIB1/2 header scan (no ecCodes wheel exists for py3.14).
"""
from __future__ import annotations

import collections
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RD_OVER_G = 287.05 / 9.80665


def season_of(month) -> np.ndarray:
    return np.vectorize(C.SEASON_OF_MONTH.get)(np.asarray(month))


def _rh_from_t_td(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) via Magnus formula."""
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))


def build_era5_daily(out_path: Path = C.ERA5_DAILY_PARQUET) -> pd.DataFrame:
    """Aggregate the 6-hourly ERA5 samples (00/06/12/18 UTC) to UTC calendar days.

    tp in the CDS file is the 1-hour accumulation ending at each sample time, so the
    daily total is estimated as mean(4 hourly samples) * 24 (in mm).
    """
    import xarray as xr

    acc = xr.open_dataset(C.ERA5_ACCUM_NC)
    ins = xr.open_dataset(C.ERA5_INSTANT_NC)
    times = pd.DatetimeIndex(ins.valid_time.values)
    if len(times) % 4 or not (times.hour.values.reshape(-1, 4) == [0, 6, 12, 18]).all():
        raise ValueError("expected exactly 4 samples/day at 00/06/12/18 UTC")
    nd = len(times) // 4
    lat = ins.latitude.values.astype(np.float32)
    lon = ins.longitude.values.astype(np.float32)
    ny, nx = len(lat), len(lon)

    def load(ds, v):
        return ds[v].values.astype(np.float32).reshape(nd, 4, ny, nx)

    out: dict[str, np.ndarray] = {}
    tp = load(acc, "tp")
    out["tp_mm"] = np.clip(tp.mean(1) * 24.0 * 1000.0, 0, None)
    del tp
    ssrd = load(acc, "ssrd")
    out["ssrd_MJ"] = ssrd.mean(1) * 24.0 / 1e6
    del ssrd

    t2m = load(ins, "t2m") - 273.15
    out["t2m_C_mean"], out["t2m_C_min"], out["t2m_C_max"] = t2m.mean(1), t2m.min(1), t2m.max(1)
    d2m = load(ins, "d2m") - 273.15
    out["d2m_C_mean"] = d2m.mean(1)
    out["rh_pct_mean"] = np.clip(_rh_from_t_td(t2m, d2m), 0, 100).mean(1)
    t2m_annual_k = t2m.mean(axis=(0, 1)) + 273.15
    del t2m, d2m

    u, v = load(ins, "u10"), load(ins, "v10")
    ws = np.sqrt(u * u + v * v)
    del u, v
    out["wind_speed_mean"], out["wind_speed_min"], out["wind_speed_max"] = ws.mean(1), ws.min(1), ws.max(1)
    del ws

    msl = load(ins, "msl") / 100.0
    out["msl_hPa_mean"], out["msl_hPa_min"] = msl.mean(1), msl.min(1)
    sp = load(ins, "sp") / 100.0
    out["sp_hPa_mean"] = sp.mean(1)
    # static elevation proxy from the hypsometric equation on annual means (no orography field downloaded)
    elev = RD_OVER_G * t2m_annual_k * np.log(msl.mean(axis=(0, 1)) / sp.mean(axis=(0, 1)))
    del msl, sp
    out["tcc_mean"] = load(ins, "tcc").mean(1)
    out["cape_max"] = load(ins, "cape").max(1)
    out["blh_mean"] = load(ins, "blh").mean(1)
    sst = load(ins, "sst")
    ocean_frac = np.isfinite(sst).mean(axis=(0, 1))
    del sst

    dates = pd.date_range(times[0].normalize(), periods=nd, freq="D")
    n_cells = ny * nx
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    df = pd.DataFrame({
        "date": np.repeat(dates.values, n_cells),
        "latitude": np.tile(lat2.ravel(), nd),
        "longitude": np.tile(lon2.ravel(), nd),
    })
    for k, arr in out.items():
        df[k] = arr.reshape(nd * n_cells).astype(np.float32)
    df["is_ocean"] = np.tile((ocean_frac > 0.5).ravel(), nd).astype(np.int8)
    df["elev_proxy_m"] = np.tile(np.clip(elev, 0, None).ravel(), nd).astype(np.float32)
    df.loc[df.is_ocean == 1, "elev_proxy_m"] = 0.0
    d = pd.DatetimeIndex(df["date"])
    df["day_of_year"] = d.dayofyear.astype(np.int16)
    df["month"] = d.month.astype(np.int8)
    df["season"] = pd.Categorical(season_of(df["month"].values), categories=C.SEASONS)
    df = add_extreme_flags(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def add_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["heavy_rain_day_flag"] = (df["tp_mm"] >= C.HEAVY_RAIN_MM).astype(np.int8)
    df["heatwave_day_flag"] = ((df["t2m_C_max"] >= C.HEATWAVE_TMAX_C) & (df["is_ocean"] == 0)).astype(np.int8)
    df["high_wind_day_flag"] = (df["wind_speed_max"] >= C.HIGH_WIND_MAX_MS).astype(np.int8)
    return df


def load_era5_daily(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(C.ERA5_DAILY_PARQUET, columns=columns)


def build_era5_precipitation_only_history(
    source_path: Path,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Stage a supplied daily-mean ERA5 precipitation file on HYDRA's India grid.

    This deliberately writes a *partial* history.  It is useful for observed
    precipitation analysis and for checking the temporal/grid coverage of a
    download, but it must not be passed to :class:`BlendingForecaster`: the
    trained feature contract also requires temperature, wind, pressure, cloud,
    CAPE, humidity, boundary-layer height, and radiation fields.

    The CDS daily-statistics product stores total precipitation in metres.  Its
    daily mean at 6-hour sampling has the same aggregation convention used by
    ``build_era5_daily``: mean sampled accumulation x 24 converts to mm/day.
    """
    import xarray as xr

    source_path = Path(source_path)
    out_path = out_path or (C.PROCESSED_DIR / "era5_india_2025_precipitation_only.parquet")
    with xr.open_dataset(source_path) as ds:
        if "tp" not in ds or "valid_time" not in ds.coords:
            raise ValueError("expected a CDS daily-statistics NetCDF with tp and valid_time")
        rain = ds["tp"].sel(latitude=slice(37.0, 7.0), longitude=slice(68.0, 97.0))
        dates = pd.DatetimeIndex(rain["valid_time"].values)
        expected_shape = (len(dates), 121, 117)
        if rain.shape != expected_shape:
            raise ValueError(f"expected HYDRA India grid shape {expected_shape}; received {rain.shape}")
        values = np.clip(rain.values.astype(np.float32) * 24_000.0, 0.0, None)
        latitudes = rain.latitude.values.astype(np.float32)
        longitudes = rain.longitude.values.astype(np.float32)

    cells = np.arange(121 * 117, dtype=np.int32)
    lat2, lon2 = np.meshgrid(latitudes, longitudes, indexing="ij")
    result = pd.DataFrame({
        "date": np.repeat(dates.values, len(cells)),
        "cell": np.tile(cells, len(dates)),
        "latitude": np.tile(lat2.ravel(), len(dates)),
        "longitude": np.tile(lon2.ravel(), len(dates)),
        "tp_mm": values.reshape(-1),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    manifest = {
        "source": source_path.name,
        "coverage": {"start": str(dates.min().date()), "end": str(dates.max().date()), "days": len(dates)},
        "grid": {"latitude": [7.0, 37.0], "longitude": [68.0, 97.0], "cells": len(cells)},
        "variable": "tp_mm",
        "conversion": "CDS daily mean total precipitation at 6-hour sampling x 24,000 (m to mm/day)",
        "status": "partial_history_not_valid_for_full_hydra_inference",
        "missing_feature_fields": ["t2m", "u10", "v10", "d2m", "msl", "sp", "tcc", "cape", "blh", "ssrd", "sst"],
    }
    out_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def load_imd_statewise() -> pd.DataFrame:
    df = pd.read_csv(C.IMD_STATEWISE_CSV, parse_dates=["date"])
    return df


def load_meteostat() -> pd.DataFrame:
    """Station daily 2025. NB: the clean file's ``wpgt`` column holds pressure-like values
    (~1010) for 129 rows - a column shift in earlier cleaning - so it is dropped here."""
    df = pd.read_csv(C.METEOSTAT_CSV, parse_dates=["date"])
    return df.drop(columns=["wpgt"], errors="ignore")


def load_ecmwf_snapshot() -> pd.DataFrame:
    df = pd.read_csv(C.ECMWF_FC_CSV, parse_dates=["valid_time", "init_time"])
    return df


def grib_inventory(path: Path, max_messages: int | None = None) -> dict:
    """Header-only scan of a GRIB file: message count, time range, parameter keys."""
    inv: collections.Counter = collections.Counter()
    times = []
    n = 0
    with open(path, "rb") as f:
        pos = 0
        while True:
            f.seek(pos)
            h = f.read(16)
            if len(h) < 8 or h[:4] != b"GRIB":
                break
            ed = h[7]
            if ed == 1:
                length = struct.unpack(">I", b"\0" + h[4:7])[0]
                f.seek(pos + 8)
                pds = f.read(28)
                year = (pds[24] - 1) * 100 + pds[12]
                key = f"grib1:table{pds[3]}:param{pds[8]}:level{pds[9]}"
                times.append((year, pds[13], pds[14], pds[15]))
            else:
                length = struct.unpack(">Q", h[8:16])[0]
                f.seek(pos + 16)
                body = f.read(min(length - 16, 4096))
                p, key = 0, f"grib2:disc{h[6]}"
                while p + 5 <= len(body):
                    sl, sn = struct.unpack(">I", body[p:p + 4])[0], body[p + 4]
                    if sn == 1:
                        times.append((struct.unpack(">H", body[p + 12:p + 14])[0], *body[p + 14:p + 17]))
                    if sn == 4:
                        key += f":cat{body[p + 9]}:num{body[p + 10]}:lev{body[p + 22]}"
                        break
                    p += sl
            inv[key] += 1
            n += 1
            pos += length
            if max_messages and n >= max_messages:
                break
    return {"messages": n, "first_time": min(times) if times else None,
            "last_time": max(times) if times else None, "params": dict(inv)}

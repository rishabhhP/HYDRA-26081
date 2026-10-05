"""Aggregate IMD grid cells to the prototype's 36 states/UTs: area mean and wettest cell per day."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import ImdRain

CACHE = Path(__file__).resolve().parent / "cache"


def cell_states(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """Flat cell indices per state (cell centre inside the boundary). Cached per grid."""
    CACHE.mkdir(exist_ok=True)
    key = CACHE / f"cells_{len(lat)}x{len(lon)}_{lat[0]:.2f}_{lon[0]:.2f}.json"
    if key.exists():
        return {k: np.array(v, int) for k, v in json.loads(key.read_text()).items()}
    from backend import data as D
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    out: dict[str, list[int]] = {}
    for i, (y, x) in enumerate(zip(lat2.ravel(), lon2.ravel())):
        st = D.state_at(float(y), float(x))
        if st:
            out.setdefault(st, []).append(i)
    key.write_text(json.dumps(out))
    return {k: np.array(v, int) for k, v in out.items()}


def state_daily(imd: ImdRain, min_valid_frac: float = 0.5) -> pd.DataFrame:
    """One row per state and day: mean over valid IMD cells, wettest cell, number of valid cells.

    Days where fewer than `min_valid_frac` of a state's cells have data are dropped rather than guessed.
    Small island UTs (Lakshadweep, Andaman and Nicobar) are usually outside IMD's mainland grid and are omitted.
    """
    cells = cell_states(imd.lat, imd.lon)
    flat = imd.rain.reshape(len(imd.dates), -1)
    frames = []
    for state, idx in sorted(cells.items()):
        block = flat[:, idx]
        valid = np.isfinite(block).sum(1)
        ok = valid >= max(1, int(np.ceil(min_valid_frac * len(idx))))
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=1)
            local = np.nanmax(np.where(np.isfinite(block), block, -np.inf), axis=1)
        frames.append(pd.DataFrame({"state": state, "date": imd.dates[ok], "imd_mm": mean[ok],
                                    "imd_local_max_mm": local[ok], "imd_cells": valid[ok]}))
    return pd.concat(frames, ignore_index=True)

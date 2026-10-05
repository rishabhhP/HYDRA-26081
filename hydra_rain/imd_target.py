"""Train and verify HYDRA v3 on IMD gauge-based rainfall instead of ERA5 rainfall (v3.4).

The ERA5-vs-IMD truth check showed ERA5 itself explains most of HYDRA's heavy-rain shortfall: a model trained on
ERA5 precipitation learns ERA5's smoothed peaks. This module swaps the rainfall field of the training cube
(`tp_mm`, which drives the target, persistence, recent-rain, climatology and neighbourhood-rain features) for
IMD 0.25 degree gridded rainfall wherever IMD has data. Atmospheric predictors stay ERA5.

Grid: IMD (6.5-38.5 N, 66.5-100 E) and HYDRA's ERA5 box (7-37 N, 68-97 E) are both on 0.25 degree multiples,
so every ERA5 cell maps to exactly one IMD cell; no interpolation.

Day convention: IMD days run 08:30-08:30 IST, ERA5 days 00-24 UTC. Rather than assume how IMD labels its day,
`alignment_check` measures the IMD date shift (-1, 0, +1) that best matches ERA5 and `--imd-day-shift auto` uses it.

Cells IMD does not cover (sea, Lakshadweep, Andaman and Nicobar) keep ERA5 rainfall; the report says how much of
each state's training data is IMD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def map_indices(era_lat: np.ndarray, era_lon: np.ndarray, imd_lat: np.ndarray, imd_lon: np.ndarray):
    """Index of the IMD cell for each ERA5 row/column, or -1 where the ERA5 cell is outside the IMD grid."""
    def idx(target, source):
        out = np.full(len(target), -1, int)
        for i, v in enumerate(target):
            j = int(np.argmin(np.abs(source - v)))
            if abs(source[j] - v) < 1e-6:
                out[i] = j
        return out
    iy, ix = idx(era_lat, imd_lat), idx(era_lon, imd_lon)
    if (iy < 0).all() or (ix < 0).all():
        raise ValueError("ERA5 and IMD grids share no cells; both must be 0.25 degree grids on the same lattice")
    return iy, ix


def imd_on_era5_grid(cube, imd, shift_days: int = 0) -> np.ndarray:
    """(days, H, W) IMD rainfall for each ERA5 cube day; NaN where IMD has no value. Value for ERA5 day d is IMD[d + shift]."""
    iy, ix = map_indices(cube.lat, cube.lon, imd.lat, imd.lon)
    H, W = cube.shape
    out = np.full((cube.n_days, H, W), np.nan, np.float32)
    pos = {pd.Timestamp(d): k for k, d in enumerate(imd.dates)}
    rows_ok, cols_ok = np.flatnonzero(iy >= 0), np.flatnonzero(ix >= 0)
    for k, d in enumerate(cube.dates):
        j = pos.get(pd.Timestamp(d) + pd.Timedelta(days=shift_days))
        if j is None:
            continue
        out[k][np.ix_(rows_ok, cols_ok)] = imd.rain[j][np.ix_(iy[rows_ok], ix[cols_ok])]
    return out


def _state_series(cube, field: np.ndarray) -> np.ndarray:
    flat = field.reshape(field.shape[0], -1)
    cols = []
    for name in cube.state_names:
        cells = cube.state_cells[name]
        block = flat[:, cells]
        with np.errstate(invalid="ignore"):
            cols.append(np.where(np.isfinite(block).mean(1) >= 0.5, np.nanmean(np.where(np.isfinite(block), block, np.nan), 1), np.nan))
    return np.column_stack(cols)


def alignment_check(cube, imd, shifts=(-1, 0, 1)) -> dict:
    """Correlation of daily state-mean rainfall, ERA5 vs IMD, for each candidate IMD date shift."""
    era = _state_series(cube, cube.fields["tp_era5_mm"] if "tp_era5_mm" in cube.fields else cube.fields["tp_mm"])
    out = {}
    import warnings
    for s in shifts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            other = _state_series(cube, imd_on_era5_grid(cube, imd, s))
        ok = np.isfinite(era) & np.isfinite(other)
        out[s] = {"correlation": float(np.corrcoef(era[ok], other[ok])[0, 1]) if ok.sum() > 30 else float("nan"), "pairs": int(ok.sum())}
    best = max((s for s in out if np.isfinite(out[s]["correlation"])), key=lambda s: out[s]["correlation"], default=0)
    return {"by_shift": out, "best_shift": best}


def apply_imd_target(cube, imd, shift_days: int | str = "auto") -> dict:
    """Replace cube.fields['tp_mm'] with IMD where available (ERA5 elsewhere). Keeps ERA5 as 'tp_era5_mm'. Returns a report."""
    if "tp_era5_mm" not in cube.fields:
        cube.fields["tp_era5_mm"] = cube.fields["tp_mm"].copy()
    alignment = None
    if shift_days == "auto":
        alignment = alignment_check(cube, imd)
        shift_days = alignment["best_shift"]
    grid = imd_on_era5_grid(cube, imd, int(shift_days))
    mask = np.isfinite(grid)
    era = cube.fields["tp_era5_mm"]
    cube.fields["tp_mm"] = np.where(mask, grid, era).astype(np.float32)
    cube.fields["tp_is_imd"] = mask.astype(np.float32)
    flat = mask.reshape(cube.n_days, -1)
    share = {name: float(flat[:, cube.state_cells[name]].mean()) for name in cube.state_names}
    days_with = int(flat.any(1).sum())
    return {"source": "imd", "imd_day_shift": int(shift_days), "alignment": alignment,
            "days_with_imd": days_with, "days_total": int(cube.n_days),
            "imd_share_by_state": {k: round(v, 3) for k, v in share.items()},
            "imd_share_model_cells": float(flat[:, cube.model_cells(1)].mean()),
            "note": "Rainfall target, rainfall-history and neighbourhood-rain features use IMD where available, ERA5 elsewhere; "
                    "atmospheric predictors are ERA5."}

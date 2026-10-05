"""Leakage-safe grid-cell features for HYDRA rainfall v3.

Row = (issue day i, grid cell n, lead L); target = rainfall on day i + L at cell n.
Every feature uses truth from days <= i only. Climatology is fitted on training days
only and never uses truth within +-CLIM_EXCLUDE_DAYS of the date it describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter

from . import config as C
from .grid import GridCube

NOW_FIELDS = ("t2m_C", "d2m_C", "dpd_C", "rh_pct", "cape", "msl_hPa", "u10", "v10", "wind", "tcc",
              "ssrd_MJ", "blh")
OPTIONAL_NOW = ("tcwv", "gust", "sp_hPa")
CAT_COLS = ("season_id", "regime_id", "state_id", "lead_days")
ID_COLS = ("issue_idx", "cell", "lead_days", "state_id")
NEIGHBOUR_FIELDS = ("tp_mm", "cape", "tcwv")


def numeric_columns(cube: GridCube) -> list[str]:
    """All numeric predictors, minus any listed in HYDRA_EXCLUDE_FEATURES (comma-separated; used by the ablation runner)."""
    import os
    excluded = {c.strip() for c in os.getenv("HYDRA_EXCLUDE_FEATURES", "").split(",") if c.strip()}
    return [c for c in _numeric_columns(cube) if c not in excluded]


def _numeric_columns(cube: GridCube) -> list[str]:
    cols = ["latitude", "longitude"]
    cols += [f"now_{f}" for f in NOW_FIELDS]
    cols += ["now_log_cape", "msl_tend", "msl_anom30", "t2m_tend", "cape_tend"]
    cols += [f"now_{f}" for f in OPTIONAL_NOW]
    cols += ["tcwv_tend", "tcwv_anom30", "moist_flux_u", "moist_flux_v"]
    # v3.1: moisture-flux convergence (moisture piling up ahead of convection) and an instability x moisture signal
    cols += ["moist_flux_conv", "moist_flux_conv_nb5", "moist_flux_conv_nb9", "cape_x_rh", "cape_x_rh_nb5"]
    cols += ["tp0", "tp1", "tp2", "tp3", "tp7", "tpmax7", "wet7", "tp_anom", "clim_issue", "clim_target"]
    for size in C.NEIGHBOURHOODS:
        cols += [f"nb{size}_tp_mean", f"nb{size}_tp_max", f"nb{size}_cape_mean", f"nb{size}_tcwv_mean",
                 f"nb{size}_wet_frac"]
    return cols


def season_id(day: np.datetime64) -> int:
    return C.SEASON_OF_MONTH[int(str(day)[5:7])]


def regime_from(tp3: np.ndarray, tp7: np.ndarray, season: int) -> np.ndarray:
    regime = np.full(np.shape(tp3), C.REGIMES.index("normal"), np.int8)
    regime[np.asarray(tp3) >= C.WET_TP3_MM] = C.REGIMES.index("wet_spell")
    if season == C.SEASONS.index("monsoon"):
        regime[np.asarray(tp3) >= C.MONSOON_TP3_MM] = C.REGIMES.index("monsoon_active")
    regime[np.asarray(tp7) < C.DRY_TP7_MM] = C.REGIMES.index("dry_spell")
    return regime


# ------------------------------------------------------------------ climatology
def climatology(cube: GridCube, train_mask: np.ndarray) -> np.ndarray:
    """(D, H*W) day-of-year rainfall climatology using training days only.

    Circular day-of-year windows (so several years pool naturally), widened when fewer than
    CLIM_MIN_COUNT training days are available, always excluding training truth within
    +-CLIM_EXCLUDE_DAYS calendar days of the described date.
    """
    tp = cube.flat("tp_mm").astype(np.float64)
    D, N = tp.shape
    doy = (cube.dates - cube.dates.astype("datetime64[Y]")).astype(int)  # 0..365
    train = np.asarray(train_mask, bool)
    s_doy = np.zeros((366, N))
    n_doy = np.zeros(366)
    np.add.at(s_doy, doy[train], tp[train])
    np.add.at(n_doy, doy[train], 1)
    clim = np.full((D, N), np.nan, np.float32)
    todo = np.ones(D, bool)
    for half in C.CLIM_WINDOWS:
        # circular window sums over day-of-year
        cs = np.concatenate([s_doy[-half:], s_doy, s_doy[:half]])
        cn = np.concatenate([n_doy[-half:], n_doy, n_doy[:half]])
        cs = np.vstack([np.zeros((1, N)), np.cumsum(cs, 0)])
        cn = np.concatenate([[0.0], np.cumsum(cn)])
        for d in np.flatnonzero(todo):
            k = doy[d]
            total, count = cs[k + 2 * half + 1] - cs[k], cn[k + 2 * half + 1] - cn[k]
            near = np.flatnonzero(train[max(0, d - C.CLIM_EXCLUDE_DAYS):d + C.CLIM_EXCLUDE_DAYS + 1]) \
                + max(0, d - C.CLIM_EXCLUDE_DAYS)
            if near.size:
                total = total - tp[near].sum(0)
                count = count - near.size
            if count >= C.CLIM_MIN_COUNT:
                clim[d] = total / count
                todo[d] = False
        if not todo.any():
            break
    if todo.any():
        clim[todo] = (s_doy.sum(0) / max(n_doy.sum(), 1))[None, :]
    return clim


# ------------------------------------------------------------------ per-day feature planes
class FeatureBuilder:
    """Computes full-grid feature planes per issue day and samples them at requested cells."""

    def __init__(self, cube: GridCube, clim: np.ndarray):
        self.cube, self.clim = cube, clim
        self.columns = numeric_columns(cube)
        H, W = cube.shape
        lat2, lon2 = np.meshgrid(cube.lat, cube.lon, indexing="ij")
        self.lat, self.lon = lat2.ravel().astype(np.float32), lon2.ravel().astype(np.float32)
        self.has = {name: name in cube.fields for name in OPTIONAL_NOW}
        self.shape = (H, W)

    def _f(self, name: str, i: int) -> np.ndarray:
        if name not in self.cube.fields:
            return np.full(self.shape, np.nan, np.float32)
        return self.cube.fields[name][max(i, 0)]

    def _window(self, name: str, i: int, days: int) -> np.ndarray:
        return self.cube.fields[name][max(0, i - days + 1):i + 1]

    def plane(self, i: int) -> dict[str, np.ndarray]:
        """All issue-day features on the full grid, flattened to (H*W,)."""
        tp = self._window("tp_mm", i, 7)
        out = {
            "tp0": tp[-1], "tp1": self._f("tp_mm", i - 1), "tp2": self._f("tp_mm", i - 2),
            "tp3": tp[-3:].mean(0), "tp7": tp.mean(0), "tpmax7": tp.max(0),
            "wet7": (tp >= C.WET_MM).sum(0).astype(np.float32),
        }
        for name in NOW_FIELDS + OPTIONAL_NOW:
            out[f"now_{name}"] = self._f(name, i)
        out["now_log_cape"] = np.log1p(out["now_cape"])
        out["msl_tend"] = out["now_msl_hPa"] - self._f("msl_hPa", i - 1)
        out["msl_anom30"] = out["now_msl_hPa"] - self._window("msl_hPa", i, 30).mean(0)
        out["t2m_tend"] = out["now_t2m_C"] - self._f("t2m_C", i - 1)
        out["cape_tend"] = out["now_cape"] - self._f("cape", i - 1)
        if self.has["tcwv"]:
            out["tcwv_tend"] = out["now_tcwv"] - self._f("tcwv", i - 1)
            out["tcwv_anom30"] = out["now_tcwv"] - self._window("tcwv", i, 30).mean(0)
            out["moist_flux_u"] = out["now_tcwv"] * out["now_u10"]
            out["moist_flux_v"] = out["now_tcwv"] * out["now_v10"]
        else:
            for name in ("tcwv_tend", "tcwv_anom30", "moist_flux_u", "moist_flux_v"):
                out[name] = np.full(self.shape, np.nan, np.float32)
        if self.has["tcwv"]:
            # convergence of the column moisture flux, -div(q V), per 0.25 deg cell; north-to-south latitude order flips d/dy
            dqu_dx = np.gradient(out["moist_flux_u"], axis=1)
            dqv_dy = -np.gradient(out["moist_flux_v"], axis=0)
            conv = -(dqu_dx + dqv_dy)
        else:
            conv = np.full(self.shape, np.nan, np.float32)
        out["moist_flux_conv"] = conv
        out["cape_x_rh"] = out["now_cape"] * out["now_rh_pct"] / 100.0
        for size in (5, 9):
            out[f"moist_flux_conv_nb{size}"] = uniform_filter(np.nan_to_num(conv), size, mode="nearest") if self.has["tcwv"] else conv
        out["cape_x_rh_nb5"] = uniform_filter(out["cape_x_rh"], 5, mode="nearest")
        wet = (out["tp0"] >= C.WET_MM).astype(np.float32)
        for size in C.NEIGHBOURHOODS:
            out[f"nb{size}_tp_mean"] = uniform_filter(out["tp0"], size, mode="nearest")
            out[f"nb{size}_tp_max"] = maximum_filter(out["tp0"], size, mode="nearest")
            out[f"nb{size}_cape_mean"] = uniform_filter(out["now_cape"], size, mode="nearest")
            moisture = out["now_tcwv"] if self.has["tcwv"] else out["now_rh_pct"]
            out[f"nb{size}_tcwv_mean"] = uniform_filter(moisture, size, mode="nearest")
            out[f"nb{size}_wet_frac"] = uniform_filter(wet, size, mode="nearest")
        return {k: np.asarray(v, np.float32).ravel() for k, v in out.items()}

    def rows(self, issue_indices, cells: np.ndarray, leads=C.LEADS, with_target: bool = True) -> pd.DataFrame:
        cube, frames = self.cube, []
        cells = np.asarray(cells, np.int64)
        tp_flat = cube.flat("tp_mm")
        for i in issue_indices:
            plane = self.plane(i)
            season = season_id(cube.dates[i])
            base = {name: plane[name][cells] for name in plane}
            base["clim_issue"] = self.clim[i, cells]
            base["tp_anom"] = base["tp0"] - base["clim_issue"]
            regime = regime_from(base["tp3"], base["tp7"], season)
            for lead in leads:
                t = i + lead
                if with_target and t >= cube.n_days:
                    continue
                frame = dict(base)
                frame["latitude"], frame["longitude"] = self.lat[cells], self.lon[cells]
                frame["clim_target"] = self.clim[min(t, cube.n_days - 1), cells]
                frame["season_id"] = np.full(cells.size, season, np.int8)
                frame["regime_id"] = regime
                frame["state_id"] = cube.cell_state[cells].astype(np.int16)
                frame["lead_days"] = np.full(cells.size, lead, np.int8)
                frame["issue_idx"] = np.full(cells.size, i, np.int32)
                frame["cell"] = cells
                frame["target"] = tp_flat[t, cells] if t < cube.n_days else np.full(cells.size, np.nan, np.float32)
                frames.append(pd.DataFrame(frame))
        if not frames:
            return pd.DataFrame(columns=self.columns + list(CAT_COLS) + ["issue_idx", "cell", "target"])
        return pd.concat(frames, ignore_index=True)


def state_p95(cube: GridCube, train_mask: np.ndarray) -> np.ndarray:
    """Per-state training-period wet-day 95th percentile of cell rainfall (mm/day); index = state_id."""
    tp = cube.flat("tp_mm")[np.asarray(train_mask, bool)]
    out = np.full(len(cube.state_names), np.nan, np.float32)
    for sid, name in enumerate(cube.state_names):
        cells = cube.state_cells[name]
        values = tp[:, cells].ravel()
        wet = values[values >= C.WET_MM]
        out[sid] = np.quantile(wet, C.STATE_PERCENTILE) if wet.size >= 20 else C.HEAVY_THRESHOLDS_MM[-1]
    # keep the relative head meaningful and between the fixed thresholds' scale
    return np.clip(out, 10.0, 150.0)


def heavy_labels(target: np.ndarray, state_ids: np.ndarray, p95: np.ndarray) -> dict[str, np.ndarray]:
    labels = {f"{t:g}": (target >= t).astype(np.float32) for t in C.HEAVY_THRESHOLDS_MM}
    thresholds = p95[np.clip(state_ids, 0, len(p95) - 1)]
    labels["state_p95"] = (target >= thresholds).astype(np.float32)
    return labels

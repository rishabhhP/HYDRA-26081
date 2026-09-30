"""Feature engineering on (day x cell) cubes.

Timing convention (the leakage contract, enforced by tests/test_features.py):
    a row is (cell, target day d, lead h). The forecast is issued at the end of day
    i = d - h, so EVERY feature may only use truth from days <= i, plus statistics
    fitted on TRAIN days only (climatology, region x season skill, MSLP normals).

Experts (the sources the gating network blends):
    climatology      train-only smoothed day-of-year mean (+-30 d window, excluding +-5 d)
    persistence      truth on issue day i
    recent3          mean truth over i-2..i
    anom_persistence climatology(d) + [truth(i) - climatology(i)]   (clipped >= 0 for rain)
The real ECMWF forecast has no time overlap with any truth in the workspace (single
2026-09-24 0 h analysis vs. 2025 ERA5), so it cannot be a *trained* expert; see inference.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .ingest import season_of

STATE_VARS = ["tp_mm", "t2m_C_mean", "t2m_C_max", "wind_speed_mean", "wind_speed_max",
              "msl_hPa_mean", "msl_hPa_min", "tcc_mean", "cape_max", "rh_pct_mean",
              "blh_mean", "ssrd_MJ"]
CUBE_VARS = sorted(set(STATE_VARS) | set(C.TARGETS))
STATIC_COLS = ["latitude", "longitude", "is_ocean", "elev_proxy_m", "coastal", "region_id"]
LEAD_BUCKETS = {0: "0-6h", 1: "6-24h", 2: "24-72h", 3: "24-72h"}

# ---- feature schema (documented in README). Categorical columns are integer-coded.
CAT_FEATURES = ["season_id", "regime_id", "region_id", "lead_bucket_id"]


def expert_cols(target: str) -> list[str]:
    return [f"exp_{e}" for e in C.EXPERTS]


SKILL_COLS = ([f"mae7_{e}" for e in C.EXPERTS] + [f"rmse30_{e}" for e in C.EXPERTS]
              + [f"regmae_{e}" for e in C.EXPERTS])
STATE_COLS = [f"now_{v}" for v in STATE_VARS] + ["now_msl_anom", "now_tp3", "now_tp7",
                                                  "now_target_tendency", "now_anomaly"]
CONTEXT_COLS = (["latitude", "longitude", "is_ocean", "elev_proxy_m", "coastal", "region_id",
                 "season_id",
                 "lead_days", "lead_bucket_id", "regime_id"]
                + SKILL_COLS + STATE_COLS + ["expert_spread", "expert_range"])
# calendar columns (month, day_of_year, doy_sin/cos) stay in the tables but are NOT model inputs:
# with ~158 train days they let models memorise training days (val ablation: LightGBM t2m RMSE
# 1.69 with them vs 1.51 without; persistence 1.59).
CALENDAR_COLS = ["month", "day_of_year", "doy_sin", "doy_cos"]
MODEL_FEATURES = CONTEXT_COLS + [f"exp_{e}" for e in C.EXPERTS]
LABEL_COLS = ["target"] + C.FLAGS
ID_COLS = ["date", "cell", "lead_days", "split"]


# ------------------------------------------------------------------ helpers
def _roll_nanmean(x: np.ndarray, win: int, min_periods: int = 1) -> np.ndarray:
    """Trailing rolling mean over axis 0 (window ends at and includes row t), NaN-aware."""
    valid = np.isfinite(x)
    xs = np.where(valid, x, 0.0).astype(np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), xs]), axis=0)
    cn = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), valid]), axis=0)
    t = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(t - win, 0)
    s = cs[t] - cs[lo]
    n = cn[t] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n < min_periods] = np.nan
    return out.astype(np.float32)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """out[t] = x[t-k] (NaN-padded)."""
    out = np.full_like(x, np.nan, dtype=np.float32)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def windowed_train_climatology(y: np.ndarray, train_mask: np.ndarray,
                               half_window: int = C.CLIM_HALF_WINDOW,
                               exclude: int = C.CLIM_EXCLUDE,
                               min_count: int = C.CLIM_MIN_COUNT) -> np.ndarray:
    """clim[d, n] = mean of y over TRAIN days d' with exclude < |d'-d| <= half_window.

    The window is widened (x2, x3, x6, whole year) for days with < min_count train days in
    range (only happens in the time-forward split). Uses cumulative sums: O(D*N).
    """
    D = y.shape[0]
    m = train_mask.astype(np.float64)
    cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y * m[:, None], axis=0)])
    cn = np.concatenate([[0.0], np.cumsum(m)])

    def win_sum(a: np.ndarray, b: np.ndarray):
        a, b = np.clip(a, 0, D), np.clip(b, 0, D)
        return cs[b] - cs[a], cn[b] - cn[a]

    d = np.arange(D)
    clim = np.full(y.shape, np.nan)
    todo = np.ones(D, bool)
    for w in [half_window, 2 * half_window, 3 * half_window, 6 * half_window, D]:
        s1, n1 = win_sum(d - w, d - exclude)          # [d-w, d-exclude)
        s2, n2 = win_sum(d + exclude + 1, d + w + 1)  # (d+exclude, d+w]
        n = n1 + n2
        ok = todo & (n >= min_count)
        clim[ok] = (s1[ok] + s2[ok]) / n[ok][:, None]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():
        clim[todo] = (cs[-1] / max(cn[-1], 1))[None, :]
    return clim.astype(np.float32)


# ------------------------------------------------------------------ fitted state
@dataclass
class FittedStats:
    """Everything estimated from TRAIN days only. Saved as an artifact for inference."""
    clim_by_doy: dict[str, np.ndarray]     # target -> (366, N) indexed by day_of_year
    msl_train_mean: np.ndarray             # (N,)
    region_skill: pd.DataFrame             # region_id, season, lead_days, target, expert -> mae
    static: pd.DataFrame                   # per cell static columns (index = cell)
    meta: dict = field(default_factory=dict)


def load_cubes(df: pd.DataFrame | None = None) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, pd.DataFrame]:
    """Daily archive -> cubes of shape (D, N_all_cells) + per-cell static table."""
    from .ingest import load_era5_daily
    cols = ["date", "latitude", "longitude", "is_ocean", "elev_proxy_m"] + CUBE_VARS + C.FLAGS
    if df is None:
        df = load_era5_daily(columns=cols)
    df = df.sort_values(["date", "latitude", "longitude"], ascending=[True, False, True])
    dates = pd.DatetimeIndex(df["date"].unique())
    D = len(dates)
    N = len(df) // D
    cubes = {v: df[v].to_numpy(np.float32).reshape(D, N) for v in CUBE_VARS + C.FLAGS}
    first = df.iloc[:N]
    static = pd.DataFrame({
        "latitude": first["latitude"].to_numpy(np.float32),
        "longitude": first["longitude"].to_numpy(np.float32),
        "is_ocean": first["is_ocean"].to_numpy(np.int8),
        "elev_proxy_m": first["elev_proxy_m"].to_numpy(np.float32),
    })
    static.index.name = "cell"
    static = add_static(static)
    return cubes, dates, static


def add_static(static: pd.DataFrame) -> pd.DataFrame:
    lats = np.sort(static.latitude.unique())[::-1]
    lons = np.sort(static.longitude.unique())
    ny, nx = len(lats), len(lons)
    ocean = static.is_ocean.to_numpy().reshape(ny, nx)
    pad = np.pad(ocean, 1, mode="edge")
    nb = np.stack([pad[i:i + ny, j:j + nx] for i in range(3) for j in range(3)])
    static["coastal"] = ((nb.min(0) == 0) & (nb.max(0) == 1)).ravel().astype(np.int8)
    r = np.floor((static.latitude - 7.0) / C.REGION_DEG).clip(0, None).astype(int)
    c = np.floor((static.longitude - 68.0) / C.REGION_DEG).clip(0, None).astype(int)
    static["region_id"] = (r * 100 + c).astype(np.int16)
    return static


def region_name(region_id: int) -> str:
    r, c = divmod(int(region_id), 100)
    la, lo = 7 + r * C.REGION_DEG, 68 + c * C.REGION_DEG
    return f"{la:.0f}-{la + C.REGION_DEG:.0f}N/{lo:.0f}-{lo + C.REGION_DEG:.0f}E"


def subgrid_cells(static: pd.DataFrame, stride: int = C.GRID_STRIDE) -> np.ndarray:
    lats = np.sort(static.latitude.unique())[::-1]
    lons = np.sort(static.longitude.unique())
    keep_lat = set(lats[::stride].tolist())
    keep_lon = set(lons[::stride].tolist())
    m = static.latitude.isin(keep_lat) & static.longitude.isin(keep_lon)
    return static.index[m].to_numpy()


def holdout_cells(static: pd.DataFrame, n: int = C.SPATIAL_HOLDOUT_CELLS, seed: int = C.SEED) -> np.ndarray:
    sub = set(subgrid_cells(static).tolist())
    rest = np.array([c for c in static.index if c not in sub])
    return np.sort(np.random.default_rng(seed).choice(rest, size=min(n, len(rest)), replace=False))


# ------------------------------------------------------------------ builder
class FeatureBuilder:
    def __init__(self, cubes: dict[str, np.ndarray], dates: pd.DatetimeIndex, static: pd.DataFrame):
        self.cubes, self.dates, self.static = cubes, pd.DatetimeIndex(dates), static
        self.D, self.N = next(iter(cubes.values())).shape
        self.doy = self.dates.dayofyear.to_numpy()
        self.month = self.dates.month.to_numpy()
        self.season = season_of(self.month)
        self.season_id = np.array([C.SEASONS.index(s) for s in self.season], dtype=np.int8)

    # ---------------- fitting (train days only)
    def fit(self, split: pd.Series) -> FittedStats:
        train = (split.reindex(self.dates).values == "train")
        clim_by_doy = {}
        for t in C.TARGETS:
            clim = windowed_train_climatology(self.cubes[t], train)
            if t == "tp_mm":
                clim = np.clip(clim, 0, None)
            tab = np.full((367, self.N), np.nan, np.float32)
            tab[self.doy] = clim
            tab[366] = tab[365] if np.isnan(tab[366]).all() else tab[366]
            clim_by_doy[t] = tab
        msl_mean = self.cubes["msl_hPa_mean"][train].mean(0).astype(np.float32)
        stats = FittedStats(clim_by_doy, msl_mean, pd.DataFrame(), self.static,
                            meta={"n_train_days": int(train.sum())})
        stats.region_skill = self._region_skill(stats, train)
        return stats

    def _region_skill(self, stats: FittedStats, train: np.ndarray) -> pd.DataFrame:
        rows = []
        reg = self.static["region_id"].to_numpy()
        for t in C.TARGETS:
            y = self.cubes[t]
            for h in C.LEADS:
                ex = self.experts(t, h, stats)
                for e, arr in ex.items():
                    err = np.abs(arr - y)
                    for s_id, s in enumerate(C.SEASONS):
                        dm = train & (self.season_id == s_id)
                        if not dm.any():
                            continue
                        e_s = np.nanmean(err[dm], axis=0)  # per cell
                        g = pd.Series(e_s).groupby(reg).mean()
                        for rid, v in g.items():
                            rows.append((rid, s, h, t, e, float(v)))
        return pd.DataFrame(rows, columns=["region_id", "season", "lead_days", "target", "expert", "mae"])

    # ---------------- experts
    def clim_cube(self, target: str, stats: FittedStats) -> np.ndarray:
        return stats.clim_by_doy[target][self.doy]

    def experts(self, target: str, h: int, stats: FittedStats) -> dict[str, np.ndarray]:
        """Expert forecasts for every target day d at lead h, shape (D, N)."""
        y = self.cubes[target]
        clim = self.clim_cube(target, stats)
        persist = _shift(y, h)
        recent3 = _shift(_roll_nanmean(y, 3, 3), h)
        anom = clim + (persist - _shift(clim, h))
        if target == "tp_mm":
            anom = np.clip(anom, 0, None)
        return {"climatology": clim, "persistence": persist, "recent3": recent3, "anom_persistence": anom}

    # ---------------- regime at issue day (cube indexed by issue day i)
    def regime_cube(self, stats: FittedStats) -> np.ndarray:
        cb = self.cubes
        tp3 = _roll_nanmean(cb["tp_mm"], 3, 3)
        tp7 = _roll_nanmean(cb["tp_mm"], 7, 7) * 7
        land = (self.static["is_ocean"].to_numpy() == 0)[None, :]
        msl_anom = cb["msl_hPa_min"] - stats.msl_train_mean[None, :]
        monsoon_season = (self.season == "monsoon")[:, None]
        reg = np.full((self.D, self.N), C.REGIMES.index("normal"), np.int8)
        # lowest priority first; later assignments override
        reg[land & (tp7 < C.DRY_TP7_MM)] = C.REGIMES.index("dry_spell")
        reg[(tp3 >= C.MONSOON_TP3_MM) & ~monsoon_season] = C.REGIMES.index("wet_spell")
        reg[(tp3 >= C.MONSOON_TP3_MM) & monsoon_season] = C.REGIMES.index("monsoon")
        reg[land & (cb["t2m_C_max"] >= C.HEATWAVE_TMAX_C)] = C.REGIMES.index("heatwave")
        reg[(msl_anom <= C.CYCLONIC_MSL_ANOM_HPA) & (cb["wind_speed_max"] >= C.CYCLONIC_WIND_MS)] = C.REGIMES.index("cyclonic")
        return reg

    # ---------------- rows
    def table(self, target: str, lead: int, stats: FittedStats, cells: np.ndarray,
              split: pd.Series | None = None, days: np.ndarray | None = None) -> pd.DataFrame:
        """Materialise rows for ``cells`` x valid target days at one lead."""
        h = lead
        y = self.cubes[target]
        ex = self.experts(target, h, stats)
        first_valid = C.WARMUP_DAYS + h  # issue day index >= WARMUP_DAYS
        d_idx = np.arange(first_valid, self.D) if days is None else np.asarray(days)
        d_idx = d_idx[d_idx >= first_valid]
        nd, nc = len(d_idx), len(cells)
        i_idx = d_idx - h

        def take(cube_d: np.ndarray, idx=d_idx) -> np.ndarray:  # cube indexed by day -> flat rows
            return cube_d[np.ix_(idx, cells)].reshape(-1)

        out: dict[str, np.ndarray] = {
            "date": np.repeat(self.dates.values[d_idx], nc),
            "cell": np.tile(cells, nd).astype(np.int32),
            "lead_days": np.full(nd * nc, h, np.int8),
        }
        st = self.static.loc[cells]
        for c in STATIC_COLS:
            out[c] = np.tile(st[c].to_numpy(), nd)
        out["season_id"] = np.repeat(self.season_id[d_idx], nc)
        out["month"] = np.repeat(self.month[d_idx], nc).astype(np.int8)
        doy = self.doy[d_idx]
        out["day_of_year"] = np.repeat(doy, nc).astype(np.int16)
        out["doy_sin"] = np.repeat(np.sin(2 * np.pi * doy / 365.25), nc).astype(np.float32)
        out["doy_cos"] = np.repeat(np.cos(2 * np.pi * doy / 365.25), nc).astype(np.float32)
        out["lead_bucket_id"] = np.full(nd * nc, sorted(set(LEAD_BUCKETS.values())).index(LEAD_BUCKETS[h]), np.int8)
        out["regime_id"] = take(self.regime_cube(stats), i_idx)

        # expert predictions for target day d, and their historical skill known at issue day i
        for e, arr in ex.items():
            out[f"exp_{e}"] = take(arr)
            abs_err = np.abs(arr - y)          # error of the expert on each (past) target day
            sq_err = (arr - y) ** 2
            out[f"mae7_{e}"] = take(_roll_nanmean(abs_err, 7, 3), i_idx)       # target days i-6..i
            out[f"rmse30_{e}"] = take(np.sqrt(_roll_nanmean(sq_err, 30, 7)), i_idx)
        rs = stats.region_skill
        rs = rs[(rs.target == target) & (rs.lead_days == h)]
        season_arr = np.repeat(self.season[d_idx], nc)
        for e in C.EXPERTS:
            lut = rs[rs.expert == e].set_index(["region_id", "season"])["mae"]
            key = pd.MultiIndex.from_arrays([out["region_id"], season_arr])
            out[f"regmae_{e}"] = lut.reindex(key).to_numpy(np.float32)

        # current (issue-day) weather state
        for v in STATE_VARS:
            out[f"now_{v}"] = take(self.cubes[v], i_idx)
        out["now_msl_anom"] = take(self.cubes["msl_hPa_min"] - stats.msl_train_mean[None, :], i_idx)
        out["now_tp3"] = take(_roll_nanmean(self.cubes["tp_mm"], 3, 3), i_idx)
        out["now_tp7"] = take(_roll_nanmean(self.cubes["tp_mm"], 7, 7) * 7, i_idx)
        out["now_target_tendency"] = take(y - _shift(y, 1), i_idx)
        out["now_anomaly"] = take(y - self.clim_cube(target, stats), i_idx)

        E = np.stack([out[f"exp_{e}"] for e in C.EXPERTS], 1)
        out["expert_spread"] = E.std(1).astype(np.float32)
        out["expert_range"] = (E.max(1) - E.min(1)).astype(np.float32)

        # labels on target day d
        out["target"] = take(y)
        for f in C.FLAGS:
            out[f] = take(self.cubes[f]).astype(np.int8)
        df = pd.DataFrame(out)
        if split is not None:
            df["split"] = pd.Categorical(np.repeat(split.reindex(self.dates).values[d_idx], nc))
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        return df


def build_all(split_name: str = "blocked", cellsets: tuple[str, ...] = ("subgrid", "holdout")) -> FittedStats:
    """Build + persist feature tables for every target x lead x cellset under one split."""
    import pickle

    from .splits import SPLITTERS

    cubes, dates, static = load_cubes()
    split = SPLITTERS[split_name](dates)
    fb = FeatureBuilder(cubes, dates, static)
    stats = fb.fit(split)
    outdir = C.FEATURES_DIR / split_name
    outdir.mkdir(parents=True, exist_ok=True)
    split.to_frame().to_csv(outdir / "split_days.csv")
    with open(outdir / "fitted_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    cellmap = {"subgrid": subgrid_cells(static), "holdout": holdout_cells(static)}
    for cs in cellsets:
        for t in C.TARGETS:
            for h in C.LEADS:
                df = fb.table(t, h, stats, cellmap[cs], split)
                df.to_parquet(outdir / f"{t}_lead{h}_{cs}.parquet", index=False)
                print(f"[features] {split_name} {cs} {t} lead{h}: {df.shape}", flush=True)
                del df
    return stats


def load_table(split_name: str, target: str, leads=C.LEADS, cellset: str = "subgrid",
               splits_keep=None, columns=None) -> pd.DataFrame:
    parts = []
    for h in leads:
        df = pd.read_parquet(C.FEATURES_DIR / split_name / f"{target}_lead{h}_{cellset}.parquet", columns=columns)
        if splits_keep is not None:
            df = df[df["split"].isin(splits_keep)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

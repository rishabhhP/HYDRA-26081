"""Grid-cell ERA5 daily-mean cube for HYDRA rainfall v3.

v2 averaged every field to a state mean before modelling, which erased local extremes.
v3 keeps the full 0.25 degree grid, (day, lat, lon) per field, and only aggregates
predictions to states at the very end.

Archive discovery is multi-year: every ``*.zip`` (or loose ``*.nc``) under the source
directory is read and placed by the dates inside the file, so 2023/, 2024/, 2025/
sub-folders or year-tagged filenames all work. Optional predictors (total-column water
vapour, gusts, surface pressure) are used when present and reported when absent.
"""
from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from . import config as C

REQUIRED = {
    "t2m": ("2m_temperature",), "d2m": ("2m_dewpoint_temperature",),
    "u10": ("10m_u_component_of_wind",), "v10": ("10m_v_component_of_wind",),
    "msl": ("mean_sea_level_pressure",), "tp": ("total_precipitation",),
    "ssrd": ("surface_solar_radiation_downwards",), "tcc": ("total_cloud_cover",),
    "cape": ("convective_available_potential_energy",), "blh": ("boundary_layer_height",),
}
OPTIONAL = {
    "tcwv": ("total_column_water_vapour",),
    "gust": ("10m_wind_gust", "instantaneous_10m_wind_gust", "maximum_10m_wind_gust"),
    "sp": ("surface_pressure",),
}
ALL_RAW = {**REQUIRED, **OPTIONAL}


@dataclass
class GridCube:
    dates: np.ndarray                     # datetime64[D], consecutive
    lat: np.ndarray                       # (H,) north -> south
    lon: np.ndarray                       # (W,)
    fields: dict[str, np.ndarray]         # derived physical fields, each (D, H, W) float32
    state_names: list[str]
    cell_state: np.ndarray                # (H*W,) state index or -1 (outside India)
    state_cells: dict[str, np.ndarray]    # flat cell indices per state (nearest-cell fallback for tiny UTs)
    fallback: dict[str, bool] = field(default_factory=dict)
    missing_optional: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lat), len(self.lon)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def day_index(self, day) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(day, "D")))

    def flat(self, name: str) -> np.ndarray:
        values = self.fields[name]
        return values.reshape(values.shape[0], -1)

    def model_cells(self, stride: int = 1) -> np.ndarray:
        """Flat indices of cells that belong to a state (optionally thinned for training)."""
        H, W = self.shape
        rows, cols = np.divmod(np.arange(H * W), W)
        inside = self.cell_state >= 0
        if stride > 1:
            inside &= (rows % stride == 0) & (cols % stride == 0)
        cells = np.flatnonzero(inside)
        extra = np.concatenate([v for v in self.state_cells.values()]) if self.state_cells else np.empty(0, int)
        return np.unique(np.concatenate([cells, extra])) if stride == 1 else cells


# ------------------------------------------------------------------ physical derivations
def humidity(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    return np.clip(100 * np.exp(17.625 * td_c / (243.04 + td_c) - 17.625 * t_c / (243.04 + t_c)), 0, 100)


def derive(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Raw ERA5 daily means -> modelling units. Daily-mean accumulations are per-hour means."""
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    t2m, d2m = raw["t2m"] - 273.15, raw["d2m"] - 273.15
    out = {
        "tp_mm": f32(np.clip(raw["tp"] * 24000.0, 0, None)),
        "t2m_C": f32(t2m), "d2m_C": f32(d2m), "dpd_C": f32(t2m - d2m), "rh_pct": f32(humidity(t2m, d2m)),
        "u10": f32(raw["u10"]), "v10": f32(raw["v10"]), "wind": f32(np.hypot(raw["u10"], raw["v10"])),
        "msl_hPa": f32(raw["msl"] / 100.0), "tcc": f32(raw["tcc"]), "cape": f32(np.clip(raw["cape"], 0, None)),
        "blh": f32(raw["blh"]), "ssrd_MJ": f32(raw["ssrd"] * 24 / 1e6),
    }
    missing = []
    for name in OPTIONAL:
        if name not in raw:
            missing.append(name)
            continue
        value = raw[name] / 100.0 if name == "sp" else raw[name]
        out[{"tcwv": "tcwv", "gust": "gust", "sp": "sp_hPa"}[name]] = f32(value)
    return out, missing


# ------------------------------------------------------------------ archive reading
def _field_of(filename: str) -> str | None:
    stem = Path(filename).name.casefold()
    for name, prefixes in ALL_RAW.items():
        if any(stem.startswith(prefix) for prefix in prefixes):
            return name
    return None


def _read_nc(path: Path):
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        variable = next(iter(dataset.data_vars))
        values = dataset[variable].sel(latitude=slice(C.LAT_NORTH, C.LAT_SOUTH),
                                       longitude=slice(C.LON_WEST, C.LON_EAST))
        if values.shape[-2:] != C.GRID_SHAPE:
            raise ValueError(f"{path.name} does not match the HYDRA India grid: {values.shape}")
        time_name = "valid_time" if "valid_time" in values.coords else "time"
        dates = pd.DatetimeIndex(values[time_name].values).normalize().values.astype("datetime64[D]")
        return (dates, values.values.astype(np.float32), values.latitude.values.astype(float),
                values.longitude.values.astype(float))


def _sources(source: Path):
    """Yield (field, opener) for every ERA5 NetCDF found in zips or loose under ``source``."""
    for path in sorted(source.rglob("*")):
        if path.suffix == ".nc" and path.is_file():
            name = _field_of(path.name)
            if name:
                yield name, path.name, (lambda p=path: p)
        elif ".zip" in path.name and path.is_file():
            try:
                with zipfile.ZipFile(path) as archive:
                    members = [m.filename for m in archive.infolist() if not m.is_dir()]
            except zipfile.BadZipFile:
                continue
            for member in members:
                name = _field_of(member)
                if name:
                    yield name, f"{path.name}/{Path(member).name}", (path, member)


def load_era5(source: Path, years: tuple[int, ...] | None = None, assign_states=None) -> GridCube:
    """Read every daily-mean ERA5 field under ``source`` into one consecutive-day cube."""
    pieces: dict[str, dict[np.datetime64, np.ndarray]] = {}
    lat = lon = None
    with tempfile.TemporaryDirectory(prefix="hydra-v3-") as tmp:
        for name, label, opener in _sources(source):
            if isinstance(opener, tuple):
                archive_path, member = opener
                target = Path(tmp) / Path(member).name
                with zipfile.ZipFile(archive_path) as archive, archive.open(member) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                path = target
            else:
                path = opener()
            dates, values, lat, lon = _read_nc(path)
            for index, day in enumerate(dates):
                if years and int(str(day)[:4]) not in years:
                    continue
                pieces.setdefault(name, {})[day] = values[index]
            print(f"  read {label}: {len(dates)} days", flush=True)
            if path.parent == Path(tmp):
                path.unlink(missing_ok=True)
    missing_required = [name for name in REQUIRED if name not in pieces]
    if missing_required:
        raise FileNotFoundError(f"ERA5 daily-mean fields missing entirely: {missing_required}")
    common = sorted(set.intersection(*(set(pieces[name]) for name in REQUIRED)))
    if not common:
        raise ValueError("No day has every required ERA5 field")
    dates = np.arange(common[0], common[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
    gaps = sorted(set(dates) - set(common))
    if gaps:
        raise ValueError(f"Required ERA5 fields have {len(gaps)} missing day(s), first {gaps[0]}")
    raw = {}
    for name, by_day in pieces.items():
        if name in OPTIONAL and not all(day in by_day for day in dates):
            print(f"  optional field {name} incomplete for the period; not used", flush=True)
            continue
        raw[name] = np.stack([by_day[day] for day in dates]).astype(np.float32)
    fields, missing = derive(raw)
    if assign_states is None:
        assign_states = india_state_assignment
    names, cell_state, state_cells, fallback = assign_states(lat, lon)
    return GridCube(dates, lat, lon, fields, names, cell_state, state_cells, fallback, missing)


def india_state_assignment(lat: np.ndarray, lon: np.ndarray):
    """Map grid cells to the 36 states/UTs using the prototype's boundary file."""
    from backend import data as D

    features = D.state_features()
    names = sorted(feature["properties"]["name"] for feature in features)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    flat_lat, flat_lon = lat2.ravel(), lon2.ravel()
    cell_state = np.full(flat_lat.size, -1, np.int32)
    lookup = {name: i for i, name in enumerate(names)}
    for index, (y, x) in enumerate(zip(flat_lat, flat_lon)):
        state = D.state_at(float(y), float(x))
        if state:
            cell_state[index] = lookup[state]
    state_cells, fallback = {}, {}
    for feature in features:
        name = feature["properties"]["name"]
        cells = np.flatnonzero(cell_state == lookup[name])
        fallback[name] = cells.size == 0
        if cells.size == 0:
            center = D.geometry_center(feature["geometry"])
            cells = np.array([int(np.argmin((flat_lat - center["latitude"]) ** 2
                                            + (flat_lon - center["longitude"]) ** 2))])
        state_cells[name] = cells.astype(np.int64)
    return names, cell_state, state_cells, fallback


# ------------------------------------------------------------------ synthetic cube (tests / smoke runs only)
def synthetic_cube(start: str = "2024-01-01", days: int = 540, shape: tuple[int, int] = (28, 28),
                   seed: int = 0) -> GridCube:
    """ERA5-shaped toy weather with a learnable moisture/CAPE -> next-day convective rain link.

    Never used for any published number: it exists so the full v3 pipeline can be exercised
    without the multi-gigabyte ERA5 archives.
    """
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    H, W = shape
    dates = np.arange(np.datetime64(start, "D"), np.datetime64(start, "D") + days)
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    monsoon = np.clip(np.sin((doy - 150) / 365 * 2 * np.pi * 1.6), 0, None)       # Jun-Sep bump
    lat = np.linspace(30, 30 - 0.25 * (H - 1), H)
    lon = np.linspace(72, 72 + 0.25 * (W - 1), W)
    smooth = lambda a, s: gaussian_filter(a, sigma=(0, s, s))  # noqa: E731
    noise = lambda s=3: smooth(rng.standard_normal((days, H, W)), s) * 3  # noqa: E731
    moisture = 30 + 25 * monsoon[:, None, None] + 8 * noise()
    cape = np.clip(400 + 1800 * monsoon[:, None, None] + 900 * noise(2), 0, None)
    msl = 1008 - 6 * monsoon[:, None, None] + 3 * noise()
    tp = np.zeros((days, H, W))
    trigger = (moisture[:-1] - 45) / 8 + (cape[:-1] - 1500) / 700 - (msl[:-1] - 1004) / 3
    burst = rng.gamma(0.8, 1.0, size=(days - 1, H, W))
    local = smooth(rng.standard_normal((days - 1, H, W)), 1.2) * 2.5
    tp[1:] = np.clip(np.exp(0.9 * trigger + local) * burst, 0, 250)
    tp[1:] += 0.35 * tp[:-1]
    t2m = 300 + 6 * np.cos((doy - 140) / 365 * 2 * np.pi)[:, None, None] + noise()
    d2m = t2m - np.clip(15 - (moisture - 30) / 3, 1, 25)
    raw = {
        "tp": tp / 24000.0, "t2m": t2m, "d2m": d2m, "u10": 2 + noise(), "v10": 1 + noise(),
        "msl": msl * 100, "ssrd": np.clip(2.2e7 - 4e5 * moisture, 1e6, None) / 24,
        "tcc": np.clip(0.2 + moisture / 100 + 0.05 * noise(), 0, 1), "cape": cape,
        "blh": np.clip(900 + 150 * noise(), 50, None), "tcwv": moisture, "gust": 6 + np.abs(noise()),
        "sp": (msl - 20) * 100,
    }
    fields, missing = derive({k: np.asarray(v, np.float32) for k, v in raw.items()})
    names = ["North", "East", "South", "West"]
    rows, cols = np.divmod(np.arange(H * W), W)
    cell_state = np.where(rows < H // 2, np.where(cols < W // 2, 3, 0), np.where(cols < W // 2, 2, 1)).astype(np.int32)
    cell_state[(rows == 0) & (cols == 0)] = -1
    state_cells = {name: np.flatnonzero(cell_state == i) for i, name in enumerate(names)}
    return GridCube(dates, lat, lon, fields, names, cell_state, state_cells, {n: False for n in names}, missing)

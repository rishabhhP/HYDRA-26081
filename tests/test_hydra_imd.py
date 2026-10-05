"""IMD truth pipeline and ERA5 download planning."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydra_imd.grid import LAT, LON, read_grd
from hydra_imd.states import cell_states, state_daily

ROOT = Path(__file__).resolve().parents[1]


def _fake_year(tmp: Path, transpose=False, days=30, seed=0):
    cells = cell_states(LAT, LON)
    land = np.zeros(len(LAT) * len(LON), bool)
    for v in cells.values():
        land[v] = True
    cube = np.full((days, len(LAT), len(LON)), -999.0, np.float32)
    cube.reshape(days, -1)[:, land] = np.random.default_rng(seed).gamma(0.6, 8, (days, land.sum()))
    path = tmp / "2019.grd"
    (cube.transpose(0, 2, 1) if transpose else cube).astype("<f4").tofile(path)
    return path, cube


@pytest.mark.parametrize("transpose", [False, True])
def test_grd_layout_is_verified_not_assumed(tmp_path, transpose):
    path, cube = _fake_year(tmp_path, transpose)
    imd = read_grd(path, 2019)
    assert imd.rain.shape == (30, 129, 135)
    assert np.allclose(np.nan_to_num(imd.rain), np.where(cube < 0, 0, cube))


def test_garbage_file_is_rejected(tmp_path):
    p = tmp_path / "2019.grd"
    np.random.default_rng(0).random(129 * 135 * 3).astype("<f4").tofile(p)   # no ocean mask -> land/sea check fails
    with pytest.raises(ValueError):
        read_grd(p, 2019)


def test_state_means_and_missing_days(tmp_path):
    path, _ = _fake_year(tmp_path)
    imd = read_grd(path, 2019)
    imd.rain[3] = np.nan                                     # a day with no data must be dropped, not filled
    df = state_daily(imd)
    assert df["state"].nunique() == 36 and pd.Timestamp("2019-01-04") not in set(df["date"])
    k = df[(df["state"] == "Kerala") & (df["date"] == "2019-01-01")].iloc[0]
    cells = cell_states(LAT, LON)["Kerala"]
    assert np.isclose(k["imd_mm"], np.nanmean(imd.rain[0].ravel()[cells]))
    assert np.isclose(k["imd_local_max_mm"], np.nanmax(imd.rain[0].ravel()[cells]))


def test_imd_truth_replaces_target_and_lags(tmp_path):
    from hydra_post import data as D
    df, _ = D.load()
    imd = df[["state", "date"]].copy()
    imd["imd_mm"] = df["y"] * 2 + 1
    imd["imd_local_max_mm"] = df["y_local_max"]
    csv = tmp_path / "imd.csv"
    imd.to_csv(csv, index=False)
    t, _ = D.load(truth="imd", imd_csv=csv)
    assert np.allclose(t["y"], t["y_era5"] * 2 + 1)
    g = t[t["state"] == "Kerala"].reset_index(drop=True)
    assert np.allclose(g["obs_lag1"].iloc[1:].to_numpy(), g["y"].iloc[:-1].to_numpy())   # lags follow the new truth


def test_era5_plan_matches_the_v3_loader():
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_era5_parallel import OPTIONAL, REQUIRED, plan
    from hydra_rain.grid import REQUIRED as LOADER_REQUIRED, _field_of
    jobs = plan([2024], REQUIRED + OPTIONAL, "year", "daily", Path("out"))
    names = {_field_of(j["target"].name + ".nc") for j in jobs}
    assert set(LOADER_REQUIRED) <= names and "tcwv" in names
    assert all(j["request"]["area"] == [37, 68, 7, 97] for j in jobs)

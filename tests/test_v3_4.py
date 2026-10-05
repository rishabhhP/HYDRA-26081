"""v3.4: IMD training target, adaptive conformal interval, frozen-candidate harness."""
import copy
import json

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ IMD target
def _synthetic_imd(cube, scale=1.4, shift=0, sea=True):
    from hydra_imd.grid import ImdRain, LAT, LON
    from hydra_rain.imd_target import map_indices
    iy, ix = map_indices(cube.lat, cube.lon, LAT, LON)
    dates = pd.date_range(str(cube.dates[0]), periods=cube.n_days + 2)
    rain = np.full((len(dates), len(LAT), len(LON)), np.nan, np.float32)
    for k in range(cube.n_days):
        block = cube.fields["tp_mm"][k] * scale
        if sea:
            block = block.copy()
            block[:3, :3] = np.nan
        rain[k + shift][np.ix_(iy, ix)] = block
    return ImdRain(dates, LAT, LON, rain)


def test_imd_target_maps_cells_exactly_and_keeps_era5_where_missing():
    from hydra_rain.grid import synthetic_cube
    from hydra_rain.imd_target import apply_imd_target
    cube = synthetic_cube(days=120, shape=(12, 12), seed=2)
    era = cube.fields["tp_mm"].copy()
    rep = apply_imd_target(cube, _synthetic_imd(cube), 0)
    assert np.allclose(cube.fields["tp_mm"][:, 3:, 3:], era[:, 3:, 3:] * 1.4)
    assert np.allclose(cube.fields["tp_mm"][:, :3, :3], era[:, :3, :3])
    assert np.allclose(cube.fields["tp_era5_mm"], era)
    assert 0.9 < rep["imd_share_model_cells"] <= 1.0


@pytest.mark.parametrize("planted", [0, 1])
def test_imd_day_shift_is_measured(planted):
    from hydra_rain.grid import synthetic_cube
    from hydra_rain.imd_target import alignment_check
    cube = synthetic_cube(days=150, shape=(12, 12), seed=3)
    assert alignment_check(cube, _synthetic_imd(cube, shift=planted))["best_shift"] == planted


# ------------------------------------------------------------------ adaptive conformal
def _drift(seed=0):
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(360), 30)
    x = rng.gamma(2, 2, len(t))
    y = rng.gamma(1.0, x * np.where(t < 150, 1.0, 0.35))
    f = pd.DataFrame({"date": pd.Timestamp("2025-01-01") + pd.to_timedelta(t, "D"), "x": x, "y": y, "obs_lag1": x, "season": 0})
    return f[t < 150], f[t >= 180]


def test_aci_holds_80_percent_under_drift_with_balanced_tails():
    from hydra_post.aci import AdaptiveInterval
    tr, te = _drift()
    lo, hi, _ = AdaptiveInterval(["x"]).fit(tr).run(te)
    cov = ((te["y"] >= lo) & (te["y"] <= hi)).mean()
    above, below = (te["y"] > hi).mean(), (te["y"] < lo).mean()
    assert 0.77 <= cov <= 0.83 and abs(above - below) < 0.03


def test_aci_bounds_never_use_the_same_or_later_days():
    from hydra_post.aci import AdaptiveInterval
    tr, te = _drift(1)
    a = AdaptiveInterval(["x"]).fit(tr)
    lo1, hi1, _ = copy.deepcopy(a).run(te)
    later = te.copy()
    cut = later["date"].min() + pd.Timedelta(days=40)
    later.loc[later["date"] >= cut, "y"] *= 10
    lo2, hi2, _ = copy.deepcopy(a).run(later)
    early = (te["date"] <= cut).to_numpy()          # bounds on the cut day itself are issued before it is observed
    assert np.allclose(lo1[early], lo2[early]) and np.allclose(hi1[early], hi2[early])


def test_release_gate_uses_the_declared_final_interval():
    from hydra_post.release import gate
    from pathlib import Path
    ev = json.loads((Path(__file__).resolve().parents[1] / "runtime" / "hydra_v3_1_evaluation.json").read_text())
    ev = copy.deepcopy(ev)
    ev["interval"]["adaptive conformal (ACI)"] = {"coverage": 0.95, "above": 0.03, "below": 0.02, "width": 20}
    ev["interval_final"] = "adaptive conformal (ACI)"
    assert "80% interval coverage within 75-85%" in gate(ev)["failed"]


# ------------------------------------------------------------------ frozen candidates
def test_freeze_detects_edits_and_overlap(tmp_path):
    from hydra_post import candidates as K
    spec = {"name": "c", "chosen_on": {"start": "2025-08-01", "end": "2025-12-31"}, "amount": {"method": "gate"}}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(spec))
    frozen = K.freeze(p)
    K.verify(frozen)
    edited = dict(frozen, amount={"method": "hydra"})
    with pytest.raises(SystemExit):
        K.verify(edited)
    assert K.overlaps(frozen, pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-31"))
    assert not K.overlaps(frozen, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-30"))
    assert K.overlaps({"name": "x"}, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-30"))   # unknown => not independent
    with pytest.raises(SystemExit):
        K.run_holdout(frozen, "2025-10-01", "2025-12-31")


def test_promotion_rule():
    from hydra_post import candidates as K
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=60).repeat(20)
    y = rng.gamma(1, 6, len(days))
    base = pd.DataFrame({"state": np.tile(np.arange(20), 60), "date": days, "y": y, "y_local_max": y * 3,
                         "e_persistence": y + rng.normal(0, 6, len(y))})
    for k in ("state_10", "state_20", "state_64.5", "local_64.5"):
        col, thr = K.EVENTS[k]
        base[f"p_{k}"] = np.clip((base[col] >= thr) * 0.7 + 0.1, 0, 1)
    good = base.assign(amount=np.clip(y + rng.normal(0, 1, len(y)), 0, None), lo=y * 0.4, hi=y * 1.6 + 1)
    good.loc[good.index[::10], "lo"] = good["y"] + 1            # ~10% below
    good.loc[good.index[5::10], "hi"] = good["y"] - 0.5         # ~10% above
    worse = good.assign(amount=np.clip(y + rng.normal(0, 4, len(y)), 0, None))
    assert K.compare(good, worse, independent=True)["verdict"] == "PROMOTE"
    assert K.compare(worse, good, independent=True)["verdict"] == "KEEP PRODUCTION"
    assert K.compare(good, worse, independent=False)["verdict"].startswith("NOT INDEPENDENT")

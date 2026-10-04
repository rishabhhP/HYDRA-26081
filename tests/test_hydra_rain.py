"""HYDRA rainfall v3: leakage, calibration, metrics, gate export parity and an end-to-end smoke run."""
import numpy as np
import pandas as pd
import pytest

from hydra_rain import config as C
from hydra_rain import metrics as M
from hydra_rain.calibration import StratifiedConformal, conformal_quantile, symmetric_margin
from hydra_rain.experts import AUX, EXPERTS
from hydra_rain.features import FeatureBuilder, climatology
from hydra_rain.gate import RainGate
from hydra_rain.grid import synthetic_cube


@pytest.fixture(scope="module")
def cube():
    return synthetic_cube(days=200, shape=(16, 16), seed=3)


def test_features_never_see_the_future(cube):
    clim = climatology(cube, cube.dates <= cube.dates[120])
    cells = cube.model_cells(1)[:40]
    before = FeatureBuilder(cube, clim).rows([100], cells).drop(columns="target")
    cube.fields["tp_mm"][101:] += 50.0        # change only truth after the issue day
    try:
        after = FeatureBuilder(cube, clim).rows([100], cells).drop(columns="target")
    finally:
        cube.fields["tp_mm"][101:] -= 50.0
    pd.testing.assert_frame_equal(before, after)


def test_climatology_uses_training_days_only(cube):
    mask = cube.dates <= cube.dates[90]
    clim = climatology(cube, mask)
    cube.fields["tp_mm"][91:] *= 10
    try:
        np.testing.assert_allclose(climatology(cube, mask), clim)
    finally:
        cube.fields["tp_mm"][91:] /= 10


def test_conformal_80_is_80_not_90():
    rng = np.random.default_rng(0)
    residual_cal, residual_test = rng.standard_normal(4000), rng.standard_normal(20000)
    legacy = np.quantile(np.abs(residual_cal), 0.90)            # what v2 called "80%"
    fixed = symmetric_margin(residual_cal, 0.20)
    assert abs(np.mean(np.abs(residual_test) <= legacy) - 0.90) < 0.02
    assert abs(np.mean(np.abs(residual_test) <= fixed) - 0.80) < 0.02


def test_stratified_cqr_coverage_and_fallback():
    rng = np.random.default_rng(1)
    n = 6000
    strata = pd.DataFrame({"state": rng.integers(0, 3, n), "season": rng.integers(0, 2, n),
                           "lead": rng.integers(1, 3, n), "regime": rng.integers(0, 2, n)})
    scale = 1 + 2 * strata["state"].to_numpy()                  # heteroscedastic by state
    y = rng.standard_normal(n) * scale
    lo, hi = -0.5 * np.ones(n), 0.5 * np.ones(n)               # deliberately too narrow raw band
    cal = StratifiedConformal(min_n=40, nonneg=False).fit(lo[:3000], hi[:3000], y[:3000], strata[:3000])
    l2, h2, used = cal.apply(lo[3000:], hi[3000:], strata[3000:])
    yt = y[3000:]
    inside = (yt >= l2) & (yt <= h2)
    assert abs(inside.mean() - 0.80) < 0.04
    for s in range(3):
        m = strata["state"].to_numpy()[3000:] == s
        assert abs(inside[m].mean() - 0.80) < 0.06
    unseen = strata[:5].assign(state=99)
    _, _, used = cal.apply(lo[:5], hi[:5], unseen)
    assert set(used) <= {"season+lead+regime", "season+lead", "global"}


def test_conformal_quantile_finite_sample():
    scores = np.arange(1, 11, dtype=float)                      # n = 10, alpha 0.2 -> ceil(8.8)/10 -> 9th value
    assert conformal_quantile(scores, 0.20) == 9.0


def test_metrics_contract():
    obs = np.array([0, 0.5, 5, 25, 70, 2, 30])
    pred = np.array([0.2, 2, 4, 12, 30, 1, 28])
    rep = M.full_report(obs, pred, lower=pred - 5, upper=pred + 5, references={"persistence": np.roll(obs, 1)})
    assert rep["continuous"]["samples"] == 7
    assert rep["heavy"]["20"]["events"] == 3 and rep["heavy"]["20"]["hits"] == 2
    assert rep["peaks"]["max_observed"] == 70 and rep["peaks"]["predicted_at_max"] == 30
    assert 0 <= rep["interval"]["coverage"] <= 1
    assert "mae_skill" in rep["skill"]["persistence"]


def test_weight_dynamics_flags_static_gate():
    static = np.tile([0.3, 0.3, 0.2, 0.2], (50, 1)) + 1e-4
    assert M.weight_dynamics(static, list("abcd"))["static_warning"]
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(4) * 0.5, 50)
    assert not M.weight_dynamics(w, list("abcd"))["static_warning"]


def _toy_rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    numeric = ["f1", "f2"]
    rows = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n), "season_id": rng.integers(0, 4, n),
                         "regime_id": rng.integers(0, 4, n), "state_id": rng.integers(0, 3, n),
                         "lead_days": rng.choice(C.LEADS, n)})
    rows["target"] = np.clip(np.exp(rows["f1"]) * rng.gamma(1, 2, n), 0, None).astype(np.float32)
    ex = pd.DataFrame({e: np.clip(rows["target"] + rng.normal(0, 1 + k, n), 0, None) for k, e in enumerate(EXPERTS)})
    ex["hurdle_p_rain"], ex["hurdle_wet_amount"] = rng.uniform(0, 1, n), rng.gamma(2, 3, n)
    return numeric, rows, ex.astype(np.float32)


def test_gate_numpy_forward_and_export_roundtrip(tmp_path):
    numeric, rows, ex = _toy_rows()
    gate = RainGate(numeric, EXPERTS, AUX, n_states=3).random_init()
    out = gate.forward(rows, ex)
    np.testing.assert_allclose(out["weights"].sum(1), 1, atol=1e-9)
    assert out["weights"].shape == (len(rows), len(EXPERTS))
    assert (out["raw_lower"] <= out["blend"] + 1e-9).all() and (out["raw_upper"] >= out["blend"] - 1e-9).all()
    assert set(out["heavy"]) == set(C.HEAVY_KEYS)
    gate.save(tmp_path)
    again = RainGate.load(tmp_path).forward(rows, ex)
    np.testing.assert_allclose(again["blend"], out["blend"])


def test_gate_training_parity_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    numeric, rows, ex = _toy_rows(1200)
    labels = {k: (rows["target"].to_numpy() >= t).astype(np.float32) for k, t in zip(C.HEAVY_KEYS, (5.0, 10.0, 8.0))}
    tr, va = slice(0, 900), slice(900, None)
    gate = RainGate(numeric, EXPERTS, AUX, n_states=3)
    gate.fit(rows.iloc[tr], ex.iloc[tr], rows.iloc[va], ex.iloc[va],
             {k: v[tr] for k, v in labels.items()}, {k: v[va] for k, v in labels.items()})
    out = gate.forward(rows, ex)
    assert np.isfinite(out["blend"]).all()
    gate.save(tmp_path)
    np.testing.assert_allclose(RainGate.load(tmp_path).forward(rows, ex)["weights"], out["weights"], atol=1e-6)
    assert torch is not None


def test_end_to_end_synthetic_pipeline(monkeypatch):
    from hydra_rain.pipeline import HydraRainModel

    monkeypatch.setitem(C.GBM_PARAMS, "max_iter", 40)
    monkeypatch.setattr(C, "EXPERT_OOF_FOLDS", 2)
    cube = synthetic_cube(days=330, shape=(14, 14), seed=5)
    try:
        import torch  # noqa: F401
        train_gate = True
    except ImportError:
        train_gate = False
    cutoff = cube.dates[260]
    model = HydraRainModel().fit(cube, cutoff, train_gate=train_gate)
    cells, states = model.predict(cube, np.arange(262, 320))
    assert {"lower80", "upper80", "p_rain", "p_heavy_20", "local_peak_mm", "actual_local_max_mm"} <= set(states.columns)
    assert (states["lower80"] <= states["upper80"]).all()
    assert set(states["state"]) == set(cube.state_names)
    assert model.meta["splits"]["calibration"][1] <= str(cutoff)

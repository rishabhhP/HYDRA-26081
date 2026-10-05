"""HYDRA v3.1 post-processing: leakage, diagnostic reproduction, calibration and interval behaviour."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydra_post import data as D
from hydra_post import metrics as M
from hydra_post.diagnostic import headline
from hydra_post.models import AsymmetricInterval, NativeCalibration, skill_gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    frame, _ = D.load()
    return frame


def test_reproduces_the_v3_diagnostic(df):
    h = headline(df)
    expected = {"mae": 3.80, "rmse": 7.47, "bias": -0.82, "occurrence_recall_prain": 0.772, "occurrence_precision_prain": 0.965,
                "recall_10": 0.475, "recall_20": 0.244, "csi_20": 0.195, "coverage": 0.899, "above": 0.097, "below": 0.004,
                "peak_obs": 161.9, "peak_pred": 3.8, "top5_ratio": 0.189, "mae_skill_persistence": -0.027, "rmse_skill_persistence": 0.062}
    for k, v in expected.items():
        assert abs(h[k] - v) < 0.006 + 0.01 * abs(v), (k, h[k], v)   # diagnostic values are rounded
    assert h["events_64.5"] == 13


def test_lag_features_use_only_past_days(df):
    g = df[df["state"] == "Kerala"].reset_index(drop=True)
    assert np.allclose(g["obs_lag1"].iloc[1:].to_numpy(), g["y"].iloc[:-1].to_numpy())
    changed = df.copy()
    target = changed["state"].eq("Kerala") & changed["date"].eq(pd.Timestamp("2025-09-15"))
    changed.loc[target, "y"] += 500
    again = D.add_features(changed)
    row = again["state"].eq("Kerala") & again["date"].eq(pd.Timestamp("2025-09-15"))
    for col in ("obs_lag1", "obs_mean3", "obs_mean7", "obs_max7"):
        assert again.loc[row, col].item() == D.add_features(df).loc[row, col].item()


def test_skill_gate_never_uses_the_target_day(df):
    sub = df[df["state"].isin(["Kerala", "Assam"])].reset_index(drop=True)
    p1, w1 = skill_gate(sub, D.experts(sub))
    future = sub.copy()
    future.loc[future["date"] >= pd.Timestamp("2025-10-01"), "y"] *= 3
    p2, w2 = skill_gate(future, D.experts(future))
    before = (sub["date"] <= pd.Timestamp("2025-10-01")).to_numpy()   # weight for 1 Oct uses errors up to 30 Sep only
    assert np.allclose(w1[before], w2[before]) and np.allclose(p1[before], p2[before])


def test_skill_gate_respects_concentration_cap(df):
    _, w = skill_gate(df[df["state"] == "Goa"], D.experts(df), eta=5.0, decay=0.99, cap=0.5)
    assert np.nanmax(w) <= 0.5 + 1e-9
    assert np.allclose(np.nansum(w, 1), 1)


def test_asymmetric_interval_balances_tails():
    rng = np.random.default_rng(0)
    n = 4000
    dates = pd.date_range("2025-01-01", periods=n // 20).repeat(20)
    x = rng.gamma(2, 2, n)
    y = rng.gamma(1.0, x)                                   # right-skewed like rainfall
    frame = pd.DataFrame({"date": dates, "x": x, "y": y, "obs_lag1": x, "season": 0})
    tr, te = frame.iloc[:3000], frame.iloc[3000:]
    lo, hi = AsymmetricInterval(["x"], min_n=40).fit(tr).predict(te)
    iv = M.interval(te["y"], lo, hi)
    assert 0.74 <= iv["coverage"] <= 0.88
    assert abs(iv["above"] - iv["below"]) < 0.06


def test_native_calibration_improves_brier(df):
    tr, te = df[df["date"] < "2025-10-01"], df[df["date"] >= "2025-10-01"]
    p = NativeCalibration().fit(tr).predict(te)["state_20"]["stack"]
    obs = (te["y"] >= 20).to_numpy()
    assert ((p - obs) ** 2).mean() < ((te["pa_20"].to_numpy() - obs) ** 2).mean()


def test_saved_evaluation_shows_the_improvements():
    path = ROOT / "runtime" / "hydra_v3_1_evaluation.json"
    if not path.exists():
        pytest.skip("run python -m hydra_post.evaluate first")
    ev = json.loads(path.read_text())
    a = ev["amount"]
    assert a["FINAL (history-selected)"]["mae"] < a["raw HYDRA v3"]["mae"]
    assert a["FINAL (history-selected)"]["rmse"] < a["raw HYDRA v3"]["rmse"]
    assert ev["confidence_intervals_95"]["final_minus_raw_mae"][1] < 0
    asym = ev["interval"]["asymmetric split conformal"]
    assert abs(asym["above"] - asym["below"]) < 0.03
    assert ev["probability"]["state_20"]["FINAL calibrated"]["bss"] > 0

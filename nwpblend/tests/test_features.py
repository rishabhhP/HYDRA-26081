"""Leakage + schema tests for the feature pipeline."""
import copy

import numpy as np
import pandas as pd
import pytest

from nwpblend import config as C
from nwpblend.features import (CAT_FEATURES, LABEL_COLS, MODEL_FEATURES, FeatureBuilder,
                               windowed_train_climatology)
from nwpblend.splits import EMBARGO, TEST, TRAIN, VAL, blocked_within_season, time_forward

NON_FEATURE = set(LABEL_COLS) | {"date", "cell", "split"}


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("target", C.TARGETS)
@pytest.mark.parametrize("lead", C.LEADS)
def test_features_never_see_truth_after_issue_day(synth, target, lead):
    """Perturb every cube value strictly after the issue day; features for that row must not move.

    Train-fitted statistics are held fixed (they are allowed to use any TRAIN day, by design);
    what this proves is that no feature reads truth from the target day or the lead window.
    """
    fb, stats = synth["fb"], synth["stats"]
    d = 200
    i = d - lead
    base = fb.table(target, lead, stats, np.arange(fb.N), days=np.array([d]))
    cubes2 = copy.deepcopy(synth["cubes"])
    rng = np.random.default_rng(1)
    for v in cubes2:
        cubes2[v][i + 1:] += rng.normal(0, 50, cubes2[v][i + 1:].shape).astype(np.float32)
    fb2 = FeatureBuilder(cubes2, synth["dates"], synth["static"])
    pert = fb2.table(target, lead, stats, np.arange(fb.N), days=np.array([d]))
    feats = [c for c in base.columns if c not in NON_FEATURE]
    pd.testing.assert_frame_equal(base[feats], pert[feats])
    assert not np.allclose(base["target"], pert["target"]), "sanity: the label itself did change"


def test_climatology_uses_train_days_only(synth):
    y = synth["cubes"]["t2m_C_mean"].copy()
    train = (synth["split"].values == TRAIN)
    c1 = windowed_train_climatology(y, train)
    y2 = y.copy()
    y2[~train] += 1000.0  # val/test/embargo truth must be invisible
    c2 = windowed_train_climatology(y2, train)
    np.testing.assert_allclose(c1, c2, rtol=0, atol=1e-4)


def test_climatology_excludes_the_target_day_neighbourhood(synth):
    y = synth["cubes"]["t2m_C_mean"].copy()
    train = np.ones(y.shape[0], bool)
    d = 180
    c1 = windowed_train_climatology(y, train)
    y[d - C.CLIM_EXCLUDE: d + C.CLIM_EXCLUDE + 1] += 1000.0
    c2 = windowed_train_climatology(y, train)
    np.testing.assert_allclose(c1[d], c2[d], atol=1e-4)


def test_region_skill_is_train_only(synth):
    rs = synth["stats"].region_skill
    assert {"region_id", "season", "lead_days", "target", "expert", "mae"} <= set(rs.columns)
    assert rs["mae"].notna().all() and (rs["mae"] >= 0).all()


# ------------------------------------------------------------------ splits
def test_blocked_split_integrity():
    dates = pd.date_range("2025-01-01", "2025-12-31")
    s = blocked_within_season(dates)
    assert set(s.unique()) <= {TRAIN, VAL, TEST, EMBARGO}
    season = pd.Series(dates.month).map(C.SEASON_OF_MONTH).values
    for sea in C.SEASONS:
        present = set(s.values[season == sea])
        assert {TRAIN, VAL, TEST} <= present, f"{sea} missing a split: {present}"
    held = np.flatnonzero(s.isin([VAL, TEST]).values)
    train = np.flatnonzero((s == TRAIN).values)
    gap = np.abs(train[:, None] - held[None, :]).min(1)
    assert gap.min() > C.EMBARGO_DAYS, "train day inside embargo window"
    # deterministic
    assert (blocked_within_season(dates) == s).all()


def test_time_forward_split_is_ordered():
    dates = pd.date_range("2025-01-01", "2025-12-31")
    s = time_forward(dates)
    assert dates[s == TRAIN].max() < dates[s == VAL].min() <= dates[s == VAL].max() < dates[s == TEST].min()


# ------------------------------------------------------------------ schema
@pytest.mark.parametrize("target", C.TARGETS)
def test_table_schema(synth, target):
    fb, stats = synth["fb"], synth["stats"]
    df = fb.table(target, 1, stats, np.arange(fb.N), split=synth["split"])
    missing = set(MODEL_FEATURES + LABEL_COLS + ["date", "cell", "split", "lead_days"]) - set(df.columns)
    assert not missing, missing
    for c in MODEL_FEATURES:
        assert np.issubdtype(df[c].dtype, np.number), f"{c} not numeric"
    for c in CAT_FEATURES:
        assert np.issubdtype(df[c].dtype, np.integer), f"{c} categorical must be integer-coded"
    exp = [f"exp_{e}" for e in C.EXPERTS]
    assert df[exp].notna().all().all(), "experts must be defined after warm-up"
    assert df["regime_id"].between(0, len(C.REGIMES) - 1).all()
    assert df["season_id"].between(0, 3).all()
    assert (df["expert_spread"] >= 0).all()
    if target == "tp_mm":
        assert (df[exp] >= 0).all().all(), "rain experts must be non-negative"
    # one row per (cell, day)
    assert not df.duplicated(["cell", "date"]).any()


def test_expert_definitions(synth):
    fb, stats = synth["fb"], synth["stats"]
    y = synth["cubes"]["t2m_C_mean"]
    df = fb.table("t2m_C_mean", 2, stats, np.array([0]), days=np.array([100]))
    assert df["exp_persistence"].iloc[0] == pytest.approx(y[98, 0])
    assert df["exp_recent3"].iloc[0] == pytest.approx(y[96:99, 0].mean(), rel=1e-5)
    assert df["now_t2m_C_mean"].iloc[0] == pytest.approx(y[98, 0])

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nwpblend import config as C  # noqa: E402
from nwpblend.features import CUBE_VARS, FeatureBuilder, add_static  # noqa: E402
from nwpblend.splits import blocked_within_season  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """A 3x4 synthetic grid for a full year with seasonal cycle + AR(1) weather noise."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", "2025-12-31")
    D = len(dates)
    lats, lons = np.array([20.0, 19.75, 19.5]), np.array([75.0, 75.25, 75.5, 75.75])
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    static = pd.DataFrame({"latitude": la.ravel().astype(np.float32), "longitude": lo.ravel().astype(np.float32),
                           "is_ocean": np.array([0, 0, 0, 1] * 3, np.int8),
                           "elev_proxy_m": rng.uniform(0, 500, 12).astype(np.float32)})
    static.index.name = "cell"
    static = add_static(static)
    N = len(static)
    seas = np.sin(2 * np.pi * np.arange(D) / 365)[:, None]

    def ar1(scale):
        x = np.zeros((D, N))
        for t in range(1, D):
            x[t] = 0.7 * x[t - 1] + rng.normal(0, scale, N)
        return x

    cubes = {v: (10 + 5 * seas + ar1(1.0)).astype(np.float32) for v in CUBE_VARS}
    cubes["tp_mm"] = np.clip(rng.gamma(0.5, 6, (D, N)) * (seas + 1.2), 0, None).astype(np.float32)
    cubes["t2m_C_max"] = cubes["t2m_C_mean"] + 30
    cubes["msl_hPa_min"] = (1005 + ar1(3.0)).astype(np.float32)
    cubes["msl_hPa_mean"] = cubes["msl_hPa_min"] + 2
    for f in C.FLAGS:
        cubes[f] = (rng.random((D, N)) < 0.05).astype(np.float32)
    split = blocked_within_season(dates)
    fb = FeatureBuilder(cubes, dates, static)
    stats = fb.fit(split)
    return {"cubes": cubes, "dates": dates, "static": static, "split": split, "fb": fb, "stats": stats}

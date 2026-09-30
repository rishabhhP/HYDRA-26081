"""Backend hand-off smoke test: saved artifacts load and produce the documented output schema."""
import numpy as np
import pytest

from nwpblend import config as C

ART = C.ARTIFACTS_DIR / "blocked"
pytestmark = pytest.mark.skipif(not (ART / "tp_mm" / "gating.json").exists() or not C.ERA5_DAILY_PARQUET.exists(),
                                reason="artifacts / daily archive not built yet")


def test_forecast_schema():
    from nwpblend.inference import BlendingForecaster
    from nwpblend.ingest import load_era5_daily
    fc = BlendingForecaster(ART)
    hist = load_era5_daily()
    hist = hist[hist.date >= "2025-11-15"]
    out = fc.forecast(hist, "2025-12-28", leads=(1, 2, 3), cells=[100, 7000])
    assert len(out) == 2 * 3
    for t in C.TARGETS:
        for c in [f"pred_{t}", f"lgbm_{t}", f"lo80_{t}", f"hi80_{t}", f"spread_{t}"] + [f"w_{e}_{t}" for e in C.EXPERTS]:
            assert c in out and out[c].notna().all(), c
        w = out[[f"w_{e}_{t}" for e in C.EXPERTS]].sum(1)
        np.testing.assert_allclose(w, 1, atol=1e-4)
        assert (out[f"lo80_{t}"] <= out[f"hi80_{t}"]).all()
    for f in C.FLAGS:
        assert out[f"prob_{f}"].between(0, 1).all()
    assert set(out["regime"]) <= set(C.REGIMES)

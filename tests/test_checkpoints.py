"""Validate supplied real checkpoints; synthetic feature probes are test-only."""
import json
import sys
import numpy as np
import pandas as pd
import pytest
from backend import data as D

sys.path.insert(0,str(D.REPO))


@pytest.mark.parametrize('target',D.MANIFEST['targets'])
def test_checkpoint_load_and_export_parity(target):
    from nwpblend.models.gating import GatingBlender,numpy_forward
    from nwpblend.models.quantile import QuantileModel
    from nwpblend.models.extremes import ExtremeClassifier
    from nwpblend.models.lgbm_season import SeasonalLGBM
    path=D.ART/target
    meta=json.loads((path/'gating.json').read_text())
    arrays=np.load(path/'gating.npz')
    probe=pd.DataFrame({k:[float(arrays['med'][i])] for i,k in enumerate(meta['num_cols'])})
    for k in meta['cat_cols']:probe[k]=meta['vocab'][k][0]
    probe['lead_days']=1
    model=GatingBlender.load(path,'gating')
    weights=model.weights(probe)
    np.testing.assert_allclose(weights,numpy_forward(arrays,meta,probe),atol=1e-6)
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
    assert weights.shape==(1,len(D.MANIFEST['experts']))
    assert SeasonalLGBM.load(path)
    assert QuantileModel.load(path)
    flag={'tp_mm':'heavy_rain_day_flag','t2m_C_mean':'heatwave_day_flag','wind_speed_mean':'high_wind_day_flag'}[target]
    assert ExtremeClassifier.load(path,flag)

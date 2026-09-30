"""Model-level smoke tests: gating is a convex blend and the numpy export matches torch."""
import json

import numpy as np

from nwpblend import config as C
from nwpblend.models.gating import GatingBlender, numpy_forward


def test_gating_convex_and_portable(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "GATING_EPOCHS", 2)
    fb, stats, split = synth["fb"], synth["stats"], synth["split"]
    df = fb.table("t2m_C_mean", 1, stats, np.arange(fb.N), split=split)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    g = GatingBlender().fit(tr, va)
    p, w = g.predict(va, return_weights=True)
    np.testing.assert_allclose(w.sum(1), 1, atol=1e-5)
    E = va[[f"exp_{e}" for e in C.EXPERTS]].to_numpy()
    assert (p >= E.min(1) - 1e-4).all() and (p <= E.max(1) + 1e-4).all()
    g.save(tmp_path, "g")
    npz = dict(np.load(tmp_path / "g.npz"))
    meta = json.loads((tmp_path / "g.json").read_text())
    np.testing.assert_allclose(numpy_forward(npz, meta, va), w, atol=1e-4)
    g2 = GatingBlender.load(tmp_path, "g")
    np.testing.assert_allclose(g2.weights(va), w, atol=1e-6)

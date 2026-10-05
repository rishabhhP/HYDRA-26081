"""Post-processed HYDRA in the prototype: release gate, backend payloads, WeatherGPT answers."""
import copy
import json
import os
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("WEATHERGPT_TODAY", "2026-10-04")
ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "runtime" / "hydra_v3_1_evaluation.json"
needs_eval = pytest.mark.skipif(not EVAL.exists(), reason="run python -m hydra_post.evaluate first")


@needs_eval
def test_gate_validates_current_release_and_catches_regressions():
    from hydra_post.release import FINAL, RAW, gate
    ev = json.loads(EVAL.read_text())
    assert gate(ev)["status"] == "validated"
    worse = copy.deepcopy(ev)
    worse["amount"][FINAL]["mae"] = worse["amount"][RAW]["mae"] + 0.1
    g = gate(worse)
    assert g["status"] == "provisional" and "MAE not worse than raw v3" in g["failed"]
    lopsided = copy.deepcopy(ev)
    final = lopsided.get("interval_final", "asymmetric split conformal")
    lopsided["interval"][final].update(above=0.15, below=0.01)
    assert "Interval tails balanced (within 4 points)" in gate(lopsided)["failed"]
    assert gate(ev, stale=True)["status"] == "provisional"
    assert gate(ev, baseline=worse)["status"] == "validated"          # better than a worse previous release
    better_baseline = copy.deepcopy(ev)
    better_baseline["amount"][FINAL]["mae"] -= 0.5
    assert "MAE not worse than the previous release" in gate(ev, baseline=better_baseline)["failed"]


@needs_eval
def test_state_payload_is_out_of_sample_with_correct_tiers():
    from backend import postprocessed as P
    s = P.state_payload("Maharashtra")
    assert s["status"] == "available" and s["provenance"]["kind"] == "out_of_sample"
    row = next(r for r in s["rows"] if r["valid_date"] == "2025-08-18")
    for event, t in row["alert"].items():
        assert t == P.tier(row["probabilities"][event])
    lo, hi = row["interval80"]
    assert lo <= hi
    for event, tiers in s["tier_summary"].items():
        issued = [t["issued"] for t in tiers]
        assert issued == sorted(issued, reverse=True)                  # watch >= alert >= warning


def test_tier_thresholds():
    from backend.postprocessed import tier
    assert [tier(p) for p in (0.05, 0.2, 0.39, 0.4, 0.6, 0.95, None)] == [None, "watch", "watch", "alert", "warning", "warning", None]


@needs_eval
def test_alerts_sorted_and_unknown_state():
    from backend import postprocessed as P
    a = P.alerts_on("2025-08-18")
    probs = [x["probability"] or 0 for x in a["states"]]
    assert probs == sorted(probs, reverse=True) and all(x["tier"] for x in a["issued"])
    assert P.state_payload("Atlantis")["status"] == "integration_pending"
    assert P.alerts_on("2025-08-18", "bogus")["status"] == "error"


@needs_eval
def test_weathergpt_uses_calibrated_risk():
    from weathergpt import engine
    r = engine.answer("Was there a heavy rain alert in Maharashtra on 18 August 2025?", None, None)
    assert "extremes-calibrated" in r["engine"] and "WARNING" in r["answer"]
    r = engine.answer("which states had heavy rain warnings on 18 Aug 2025", None, None)
    assert "issued a heavy-rain tier" in r["answer"]
    r = engine.answer("were there any heavy rain events in Kerala in August 2025?", None, None)
    assert "ERA5 grid cell" in r["answer"] and "HYDRA v3.1 risk for this period" in r["answer"]
    r = engine.answer("how accurate is HYDRA in Kerala?", None, None)
    assert "v3.1 release gate" in r["answer"]

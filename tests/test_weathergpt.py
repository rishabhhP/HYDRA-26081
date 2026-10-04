"""WeatherGPT v2: parsing, routing, offline answers, follow-ups, live paths (mocked) and the trained model."""
import json
import os
import random
from datetime import date, timedelta

import pytest

os.environ.setdefault("WEATHERGPT_TODAY", "2026-10-04")

from weathergpt import engine, intent, liveclient  # noqa: E402
from weathergpt.slots import extract  # noqa: E402
from weathergpt.timeparse import parse  # noqa: E402


def ask(q, session=None):
    r = engine.answer(q, {"latitude": 19.0, "longitude": 73.0}, session)
    assert r is not None and r["answer"].strip()
    assert "error" not in r["engine"], r["answer"]
    return r


@pytest.mark.parametrize("text,start,end", [
    ("rain on 20 September 2026", "2026-09-20", "2026-09-20"),
    ("between Aug 1 and Aug 31 2025", "2025-08-01", "2025-08-31"),
    ("july to september 2025", "2025-07-01", "2025-09-30"),
    ("last week", "2026-09-21", "2026-09-27"),
    ("kal kitni barish hui", "2026-10-03", "2026-10-03"),
    ("kal barish hogi kya", "2026-10-05", "2026-10-05"),
    ("monsoon 2025", "2025-06-01", "2025-09-30"),
    ("last October", "2025-10-01", "2025-10-31"),
])
def test_time_parsing(text, start, end):
    ts = parse(text, date(2026, 10, 4))
    assert (str(ts.start), str(ts.end)) == (start, end)


def test_places_aliases_typos_cities():
    s = extract("compare keral, MP and Bombay rainfall")
    names = [p.name for p in s.places]
    assert names == ["Kerala", "Madhya Pradesh", "Bombay"]
    assert s.places[2].state == "Maharashtra"
    assert not extract("look up the rain").places        # 'up' is not Uttar Pradesh


@pytest.mark.parametrize("q,intent_name", [
    ("Which state had the highest rainfall last week?", "rank_places"),
    ("how many days did Kerala get more than 20 mm in August 2025?", "threshold_days"),
    ("why was Kerala so wet in August 2025?", "why"),
    ("is Bihar in rainfall deficit?", "anomaly"),
    ("did HYDRA predict the heavy rain in Maharashtra on 18 August 2025?", "model_vs_actual"),
    ("how accurate is HYDRA in Maharashtra?", "model_accuracy"),
    ("which expert dominates in Maharashtra?", "model_weights"),
    ("kal Mumbai mein chhata le jaun kya", "advice"),
    ("what's the weather in London", "out_of_scope"),
])
def test_routing(q, intent_name):
    assert engine.resolve_intent(q, extract(q))[0] == intent_name


@pytest.mark.parametrize("q,needle", [
    ("How much rain did Kerala receive on 20 September 2026?", "Kerala: 5.0 mm"),
    ("total rain in Odisha in August 2025", "325.2 mm"),
    ("top 5 wettest states last week", "Odisha"),
    ("compare rainfall in Kerala and Karnataka in September 2026", "wetter than"),
    ("is Bihar in rainfall deficit?", "below normal"),
    ("how humid was Kerala on 24 Sep 2026", "%"),
    ("which state had the highest CAPE on 24 september 2026", "J/kg"),
    ("did HYDRA predict the heavy rain in Maharashtra on 18 August 2025?", "HYDRA forecast"),
    ("how accurate is HYDRA in Maharashtra?", "MAE"),
    ("what does the 80% interval mean?", "80%"),
    ("what data do you have", "IMD"),
])
def test_offline_answers(q, needle):
    assert needle in ask(q)["answer"]


def test_followups_keep_context():
    sid = f"t{random.random()}"
    assert "Kerala" in ask("How much rain did Kerala get in August 2025?", sid)["answer"]
    assert "Karnataka" in ask("and Karnataka?", sid)["answer"]
    r = ask("which day was wettest there?", sid)
    assert "Karnataka" in r["answer"] and r["weathergpt"]["intent"] == "rank_days"
    r = ask("did HYDRA predict it?", sid)
    assert "17 Aug 2025" in r["answer"]


def test_live_paths_with_mocked_provider(monkeypatch):
    def fake_get(url, params):
        a, b = date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"])
        days = [(a + timedelta(d)).isoformat() for d in range((b - a).days + 1)]
        reports = [{"daily": {"time": days, **{f: [5.0] * len(days) for f in params["daily"].split(",")}}}
                   for _ in params["latitude"].split(",")]
        return reports if len(reports) > 1 else reports[0]
    monkeypatch.setattr(liveclient, "_get", fake_get)
    for q in ("will it rain in Hyderabad tomorrow?", "which state got the most rain yesterday?",
              "hottest day in Delhi last month", "rainfall in Kerala in March 2019", "should I carry an umbrella in Pune tomorrow?"):
        assert ask(q)["answer"]


def test_live_failure_degrades_gracefully(monkeypatch):
    def boom(url, params):
        raise OSError("offline")
    monkeypatch.setattr(liveclient, "_get", boom)
    r = ask("will it rain in Hyderabad tomorrow?")
    assert "can't" in r["answer"] or "unavailable" in r["answer"]


def test_legacy_questions_are_deferred():
    assert engine.answer("show HYDRA weights for grid cell 12000 tomorrow", None, None) is None


def test_trained_model_quality():
    assert intent.MODEL_PATH.exists()
    metrics = json.loads((intent.MODEL_PATH.parent / "intent_metrics.json").read_text())
    assert metrics["acceptance"]["accuracy"] >= 0.9
    from weathergpt.evaluate import run
    assert run("blind.tsv")["system_accuracy"] >= 0.93

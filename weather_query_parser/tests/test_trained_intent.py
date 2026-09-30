"""Regression checks for the optional trained WeatherGPT classifier."""

from weather_query_parser.nlp.parser import parse_query
from weather_query_parser.nlp.trained_intent import status


def test_trained_intent_is_available_and_does_not_replace_deterministic_slots():
    assert status()["status"] == "available"

    parsed = parse_query("Show Hydra head weights for rainfall in Kochi tomorrow")

    # Existing deterministic routing remains the contract used by WeatherGPT.
    assert parsed["weather_variable"] == "rainfall"
    assert parsed["time_range"]["lead_hours"] == 24
    assert parsed["model_evidence_request"] is True

    learned = parsed["trained_intent"]
    assert learned["status"] == "available"
    assert learned["intent"] == "model_explain"
    assert learned["confidence"] >= 0.5
    assert learned["slots"]["model_explain_type"]["value"] is not None

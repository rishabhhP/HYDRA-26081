"""Representative WeatherGPT wording bank.

These are deterministic parser regression cases, not canned chatbot replies.
Each case validates the reusable intent fields that select a grounded source.
"""
import pytest

from weather_query_parser.nlp.parser import parse_query


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Will it rain in Chennai tomorrow?", {"weather_variable": "rainfall", "data_domain": "hydra_model", "lead": 24}),
        ("How warm will grid cell 12000 be in 48 hours?", {"weather_variable": "temperature", "grid_cell": "12000", "lead": 48}),
        ("Where are wind gusts strongest today?", {"weather_variable": "wind_gust", "data_domain": "live_layers", "scope": "spatial_ranking"}),
        ("Is there a lightning watch in Kerala now?", {"weather_variable": "thunderstorm", "data_layer": "open_meteo_lightning", "data_domain": "live_layers"}),
        ("Which state had the highest temperature yesterday?", {"weather_variable": "temperature", "data_domain": "live_layers", "ranking_scope": "states"}),
        ("Which state had most precipitation last week?", {"weather_variable": "rainfall", "data_domain": "live_layers", "ranking_scope": "states"}),
        ("What was rainfall in Kerala on 20 September 2026?", {"weather_variable": "rainfall", "data_domain": "live_layers", "location": "Kerala"}),
        ("Where had the highest rainfall from 2026-09-20 to 2026-09-25?", {"weather_variable": "rainfall", "data_domain": "live_layers", "scope": "spatial_ranking"}),
        ("Show HYDRA weight distribution for grid cell 12000 tomorrow", {"model_evidence_request": True, "grid_cell": "12000", "bucket": "hydra_short_range_forecast"}),
        ("Which expert dominates rainfall at 11.50, 84.50 tomorrow?", {"model_evidence_request": True, "weather_variable": "rainfall", "lead": 24}),
        ("Explain expert spread and the saved interval", {"bucket": "hydra_model_methodology", "topic": "uncertainty"}),
        ("Which month was wettest in India in 2014?", {"weather_variable": "rainfall", "bucket": "historical_rainfall_2014", "operation": "time_ranking"}),
        ("Can radar tell me the rainfall in my city?", {"data_domain": "imagery", "data_layer": "radar"}),
        ("What questions can HYDRA answer?", {"operation": "capability"}),
    ],
)
def test_question_bank_routes_to_a_grounded_intent(question, expected):
    parsed = parse_query(question)
    assert parsed["weather_variable"] == expected.get("weather_variable", parsed["weather_variable"])
    assert parsed["data_domain"] == expected.get("data_domain", parsed["data_domain"])
    assert parsed["data_layer"] == expected.get("data_layer", parsed["data_layer"])
    assert parsed["location"] == expected.get("location", parsed["location"])
    assert parsed["grid_cell"] == expected.get("grid_cell", parsed["grid_cell"])
    assert parsed["query_scope"] == expected.get("scope", parsed["query_scope"])
    assert parsed["ranking_scope"] == expected.get("ranking_scope", parsed["ranking_scope"])
    assert parsed["model_evidence_request"] == expected.get("model_evidence_request", parsed["model_evidence_request"])
    assert parsed["query_bucket"] == expected.get("bucket", parsed["query_bucket"])
    assert parsed["question_operation"] == expected.get("operation", parsed["question_operation"])
    assert parsed["explanation_topic"] == expected.get("topic", parsed["explanation_topic"])
    if "lead" in expected:
        assert parsed["time_range"]["lead_hours"] == expected["lead"]

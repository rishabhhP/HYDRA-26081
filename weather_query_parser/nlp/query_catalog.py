"""Supported WeatherGPT question buckets and deterministic intent helpers.

The catalog is the contract between natural-language parsing and HYDRA's
available data. It keeps query support data-driven instead of adding a one-off
route for every sentence a user types.
"""
from __future__ import annotations

import re


QUERY_CATALOG = (
    {
        "id": "historical_rainfall_2014",
        "label": "2014 historical rainfall observations",
        "source": "RF25_ind2014_rfp25.nc",
        "parameters": ("rainfall", "year", "spatial_scope", "aggregation", "ranking_direction"),
        "question_types": ("location_ranking", "time_ranking", "point_lookup", "aggregate"),
        "response_fields": ("period", "grid_resolution", "location", "state", "coordinates", "rainfall_mm", "source", "limitation"),
        "explanation_logic": "Sums or averages supplied daily rainfall observations over the requested 2014 calendar interval; labels grid cells and does not infer farm suitability.",
        "example_questions": ("Which place had the most rainfall in India in 2014?", "Which month was wettest in 2014?", "What was rainfall at 22.75, 75.75 on 2014-07-01?", "Which state grid cells were driest in 2014?"),
        "limits": "Daily rainfall observations only, on a 0.25° grid for 2014. It does not measure crop or farm suitability.",
    },
    {
        "id": "live_weather_layers",
        "label": "Live weather and retained timeline layers",
        "source": "IndianAPI / Open-Meteo",
        "parameters": ("temperature", "rainfall", "humidity", "wind_gust", "thunderstorm", "heat_stress", "soil_moisture", "location", "timeline_days"),
        "question_types": ("point_lookup", "location_ranking", "time_ranking"),
        "response_fields": ("location", "state", "coordinates", "value", "unit", "provider", "updated_at", "retained_days", "limitation"),
        "explanation_logic": "Reads an IndianAPI/Open-Meteo provider value or an average from only the days HYDRA retained; it never interpolates an India-wide field.",
        "example_questions": ("What is humidity in Kottayam now?", "Where is rainfall highest today?", "Where was humidity highest over 30 days?", "What is the next-24-hour rain outlook in Maharashtra?", "Where are wind gusts strongest?"),
        "limits": "Representative provider locations. Timeline questions use only days HYDRA retained.",
    },
    {
        "id": "hydra_short_range_forecast",
        "label": "HYDRA short-range forecast archive",
        "source": "nwpblend archived forecast output",
        "parameters": ("rainfall", "temperature", "wind", "lead_time", "location", "state", "extreme_event"),
        "question_types": ("point_lookup", "location_ranking", "explanation"),
        "response_fields": ("valid_date", "forecast_value", "unit", "interval80", "expert_weights", "regime", "event_probability", "provenance", "limitation"),
        "explanation_logic": "Uses the saved trained HYDRA output for the selected grid and lead. It reports weights as blend allocation, spread as disagreement, and saved intervals without claiming calibrated confidence.",
        "example_questions": ("What is rainfall in this grid tomorrow?", "Where is temperature highest at 48 hours?", "Where is heavy-rain probability highest?", "Explain this wind forecast.", "What is the difference between the experts?"),
        "limits": "Archive coverage is limited to the supplied forecast cells and dates; it is not a historical climate record.",
    },
    {
        "id": "radar_satellite_imagery",
        "label": "Radar and satellite imagery",
        "source": "Published radar frames / IMD satellite IR image",
        "parameters": ("radar", "satellite", "time"),
        "question_types": ("availability", "explanation"),
        "response_fields": ("layer", "updated_at", "frame_count", "bounds", "limitation"),
        "explanation_logic": "Reports published imagery availability and metadata only; it does not derive a city weather measurement from pixels.",
        "example_questions": ("Is radar available?", "What does the satellite layer show?", "How many radar frames are available?"),
        "limits": "HYDRA does not infer city-level rainfall or lightning values from image pixels.",
    },
    {
        "id": "hydra_model_methodology",
        "label": "HYDRA model methodology and evidence",
        "source": "nwpblend manifest, saved forecast outputs, and checkpoint metadata",
        "parameters": ("architecture", "experts", "weights", "uncertainty", "weather_regime", "event_classifier", "data_coverage", "provenance"),
        "question_types": ("explanation", "capability"),
        "response_fields": ("method", "trained_experts", "weights_or_intervals", "coverage", "provenance", "limitation"),
        "explanation_logic": "Explains only the documented saved-model behavior and evidence. It distinguishes model allocation from causal explanation and archive coverage from fresh inference.",
        "example_questions": ("How does HYDRA make a forecast?", "What do the expert weights mean?", "How is uncertainty shown?", "Which extreme-event classifiers are trained?", "Can HYDRA run fresh inference?", "What data is missing?"),
        "limits": "No causal attribution, operational confidence, or fresh model execution is claimed when the original history or climatology artifacts are absent.",
    },
)


def question_operation(text: str, default: str) -> str:
    """Classify an operation independent of a dataset or weather variable."""
    lowered = text.casefold()
    if re.search(r"\b(?:what|which)\s+can\s+(?:you|hydra)\b|\bwhat\s+(?:weather\s+)?(?:questions?|data|parameters?)\s+can\s+(?:you|hydra)\b|\b(?:available|supported)\s+(?:questions?|data|parameters?)\b", lowered):
        return "capability"
    if re.search(r"\b(?:compare|difference|differ|versus|vs\.?|between)\b|\b\w+\s+or\s+\w+\s+(?:warmer|cooler|wetter|drier|windier|hotter|colder)\b", lowered):
        return "comparison"
    if re.search(r"\b(?:why|explain|how)\b|\bwhat\s+(?:do|does|data|dataset|model|models|experts?|weights?|classifiers?|artifacts?)\b|\bwhich\s+(?:(?:extreme[- ]?event\s+)?classifiers?|models?|experts?)\b|\bcan\s+(?:hydra|the model)\b.*\b(?:fresh|inference)\b", lowered):
        return "explanation"
    if re.search(r"\b(?:month|season|week|day)\b.*\b(?:highest|lowest|most|least|wettest|driest)\b|\b(?:highest|lowest|most|least|wettest|driest)\b.*\b(?:month|season|week|day)\b", lowered):
        return "time_ranking"
    if re.search(r"\b(?:total|average|mean|sum|accumulated)\b", lowered) and default != "spatial_ranking":
        return "aggregate"
    if default == "spatial_ranking":
        return "location_ranking"
    return "point_lookup"


def ranking_direction(text: str) -> str:
    return "ascending" if re.search(r"\b(?:lowest|least|driest|minimum)\b", text, re.I) else "descending"


def explanation_topic(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"\b(?:uncertainty|confidence|interval|spread|disagreement)\b", lowered):
        return "uncertainty"
    if re.search(r"\b(?:weight|expert|gate|gating|allocation|persistence)\b", lowered):
        return "weights"
    if re.search(r"\b(?:heavy.?rain|heatwave|high.?wind|classifier|event)\b", lowered):
        return "events"
    if re.search(r"\b(?:data|dataset|coverage|missing|fresh inference|history|climatology)\b", lowered):
        return "data_coverage"
    if re.search(r"\b(?:provenance|source|checkpoint|artifact)\b", lowered):
        return "provenance"
    if re.search(r"\b(?:architecture|blend|model\s+work|forecast\s+made|how\s+does\s+(?:hydra|the model|the forecast)|how\s+is\s+(?:the forecast|rainfall|temperature|wind).*(?:made|produced|calculated))\b", lowered):
        return "architecture"
    return "forecast_value"


def query_bucket(domain: str, variable: str | None, text: str, operation: str = "point_lookup", topic: str = "forecast_value") -> str | None:
    if operation == "explanation" and topic != "forecast_value":
        return "hydra_model_methodology"
    if domain == "historical_observations" and variable == "rainfall" and re.search(r"\b2014\b", text):
        return "historical_rainfall_2014"
    if domain == "live_layers":
        return "live_weather_layers"
    if domain == "imagery":
        return "radar_satellite_imagery"
    if domain == "hydra_model":
        return "hydra_short_range_forecast"
    return None

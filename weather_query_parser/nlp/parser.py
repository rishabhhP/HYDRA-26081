"""Main deterministic parser for weather questions."""

from __future__ import annotations

import re

from .event_detector import detect_extreme_event
from .location_matcher import extract_location_candidate, match_location
from .models import Coordinates, ParsedQuery
from .query_catalog import explanation_topic, query_bucket, question_operation, ranking_direction
from .time_parser import parse_time_range
from .trained_intent import predict as predict_trained_intent


LAYER_PATTERNS = (
    ("satellite", "satellite", "imagery", (r"\bsatellite\b", r"\bimd\s+ir\b")),
    ("radar", "radar", "imagery", (r"\bradar\b",)),
    ("soil_moisture", "open_meteo_soil_moisture", "live_layers", (r"\bsoil\s+moisture\b", r"\bcrop\s+stress\b")),
    ("heat_stress", "open_meteo_heat_stress", "live_layers", (r"\bheat\s+stress\b", r"\bwet[ -]?bulb\b")),
    ("thunderstorm", "open_meteo_lightning", "live_layers", (r"\blightning\b", r"\blightning watch\b")),
    ("thunderstorm", "open_meteo_thunderstorm", "live_layers", (r"\bthunderstorm\b", r"\bcape\b")),
    ("wind_gust", "open_meteo_wind_gust", "live_layers", (r"\bwind\s+gusts?\b", r"\bgusts?\b")),
    ("humidity", "open_meteo_humidity", "live_layers", (r"\bhumidity\b", r"\bhumid\b")),
    ("rainfall", "open_meteo_rainfall_24h", "live_layers", (r"\bnext\s+24\s*(?:h|hours?)\b", r"\b24[ -]?hour\b", r"\bprecipitation probability\b")),
    ("rainfall", "open_meteo_rainfall", "live_layers", (r"\bcurrent\s+(?:rain|rainfall|precipitation)\b", r"\blive\s+(?:rain|rainfall|precipitation)\b")),
    ("rainfall", "hydra_forecast", "hydra_model", (r"\brainfall\b", r"\brains?\b", r"\bprecipitation\b", r"\bprecip\b", r"\bshowers?\b", r"\bwettest\b", r"\bdriest\b")),
    ("temperature", "hydra_forecast", "hydra_model", (r"\btemperature\b", r"\bheat\b", r"\bhot\b", r"\bcold\b", r"\bhow\s+(?:warm|cool)\b")),
    ("wind", "hydra_forecast", "hydra_model", (r"\bwind(?:s)?\b", r"\bbreeze\b")),
)
COORDINATE_PATTERNS = (
    re.compile(r"\b(?:lat|latitude)\s*[:=]?\s*(-?\d{1,2}(?:\.\d+)?)\s*[,; ]+\s*(?:lon|lng|longitude)\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)\b", re.I),
    re.compile(r"\b(-?\d{1,2}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)\b"),
)


def _coordinates(text: str) -> Coordinates | None:
    for pattern in COORDINATE_PATTERNS:
        if match := pattern.search(text):
            lat, lon = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return Coordinates(lat=lat, lon=lon)
    return None


def _layer(text: str) -> tuple[str | None, str | None, str]:
    matches = [(match.start(), variable, layer, domain) for variable, layer, domain, patterns in LAYER_PATTERNS for pattern in patterns if (match := re.search(pattern, text, re.I))]
    if not matches:
        return None, None, "hydra_model"
    _, variable, layer, domain = min(matches, key=lambda item: item[0])
    # Current and live questions should query the live API for the three
    # shared variables, unless a specific HYDRA forecast is requested.
    if domain == "hydra_model" and re.search(r"\b(today|yesterday|now|current|live)\b", text, re.I) and not re.search(r"\bhydra\b|\bforecast\b", text, re.I):
        live_layer = {
            "rainfall": "open_meteo_rainfall",
            "temperature": "live_weather",
            "wind": "open_meteo_wind_speed",
        }.get(variable, "live_weather")
        return variable, live_layer, "live_layers"
    return variable, layer, domain


def _query_scope(text: str) -> str:
    """Detect requests that need a ranked scan of model forecast cells."""

    ranking_patterns = (
        r"\bwhere\b",
        r"\b(?:which|name|list|show)\b.*\b(?:places?|locations?|areas?|cells?)\b",
        r"\b(?:highest|most|top|best|ideal|perfect|prone)\b",
    )
    return "spatial_ranking" if any(re.search(pattern, text, re.I) for pattern in ranking_patterns) else "selected_location"


def _ranking_scope(text: str) -> str:
    """Tell an India/state comparison apart from a list of map locations."""

    return "states" if re.search(r"\b(?:which|what|where)\s+(?:india(?:n)?\s+)?states?\b|\bstate(?:s)?\s+(?:had|has|with)\b", text, re.I) else "locations"


def _grid_cell(text: str) -> str | None:
    match = re.search(r"\b(?:grid(?:\s+cell)?|cell)\s*(?:id\s*)?(?:#\s*)?(\d+)\b", text, re.I)
    return match.group(1) if match else None


def _model_evidence_request(text: str, location: str | None, coordinates: Coordinates | None, time_range) -> bool:
    asks_for_weights = bool(re.search(
        r"\b(?:weight(?:s|ing)?|(?:head|expert)\s+(?:allocation|share|contribution|reliability|weights?)|(?:dominant|highest|largest)\s+(?:expert|head)|(?:expert|head)\b.*\b(?:dominates?|highest|largest|contributes?|reliable)|\breliable\b.*\b(?:expert|head)\b|model\s+(?:allocation|contribution)|weight\s+distribution|blend\s+distribution)\b",
        text, re.I,
    ))
    has_forecast_scope = bool(location or coordinates or _grid_cell(text) or time_range.kind == "lead_hours" or re.search(r"\b(?:this|selected)\s+(?:grid|cell|state|forecast)\b", text, re.I))
    return asks_for_weights and has_forecast_scope


def _use_case(text: str) -> str:
    categories = {
        "agriculture": (r"\bfarm(?:er|ing|s)?\b", r"\bcrop\b", r"\bsow(?:ing)?\b", r"\bharvest\b", r"\birrigation\b"),
        "aviation": (r"\baviation\b", r"\bflight\b", r"\bairport\b", r"\brunway\b"),
        "marine": (r"\bmarine\b", r"\bshipping\b", r"\bport\b", r"\bvessel\b", r"\bfishing\b"),
        "energy": (r"\benergy\b", r"\bpower\b", r"\bsolar\b", r"\bwind farm\b", r"\bpower grid\b"),
        "urban": (r"\burban\b", r"\bcity planning\b", r"\binfrastructure\b"),
        "emergency": (r"\bemergency\b", r"\bdisaster\b", r"\bevacuat", r"\bresponse\b"),
    }
    matches = [(match.start(), category) for category, patterns in categories.items() for pattern in patterns if (match := re.search(pattern, text, re.I))]
    return min(matches)[1] if matches else "general"


def _analysis_window(text: str) -> tuple[int | None, str | None]:
    patterns = (
        (r"\b(?:one|1)\s+years?(?:\s+of\s+data)?\b|\b(?:last|past|previous)\s+year\b", 365),
        (r"\b(\d+)\s+years?(?:\s+of\s+data)?\b", 365),
        (r"\b(?:one|1)\s+months?(?:\s+of\s+data)?\b|\b(?:last|past|previous)\s+month\b", 30),
        (r"\b(\d+)\s+months?(?:\s+of\s+data)?\b", 30),
        (r"\b(\d+)\s+days?(?:\s+of\s+data)?\b", 1),
    )
    for pattern, multiplier in patterns:
        if match := re.search(pattern, text, re.I):
            count = int(match.group(1)) if match.lastindex else 1
            return count * multiplier, match.group(0)
    return None, None


def parse_query(text: str) -> dict:
    """Extract only explicit weather-query fields; never silently correct input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = " ".join(text.strip().split())
    location, suggestion, city_coordinates = match_location(cleaned)
    coordinates = _coordinates(cleaned) or (Coordinates(**city_coordinates) if city_coordinates else None)
    variable, layer, domain = _layer(cleaned)
    time_range = parse_time_range(cleaned)
    # "Weather" without a named parameter is a request for current
    # conditions.  It must not fall through to the small HYDRA forecast
    # archive, which only contains scored forecast cells.  Keep explicit
    # temperature/rainfall/HYDRA questions on their existing routes.
    weather_summary_request = (
        variable is None
        and bool(re.search(r"\b(?:weather|conditions?)\b", cleaned, re.I))
        and (time_range.kind == "unspecified" or time_range.lead_hours in (0, 24, 48))
    )
    if weather_summary_request:
        layer, domain = "weather_summary", "live_layers"
    # A completed calendar window cannot be a HYDRA lead-time forecast. Route
    # temperature/rainfall questions to an exact historical provider query.
    if variable in {"temperature", "rainfall"} and time_range.kind == "date_range":
        layer = "live_weather" if variable == "temperature" else "open_meteo_rainfall"
        domain = "live_layers"
    # The supplied crop-yield repository provides one complete annual rainfall
    # observation grid. Route only that explicit year to the observation path.
    if variable == "rainfall" and re.search(r"\b2014\b", cleaned):
        layer, domain = "rainfall_observations_2014", "historical_observations"
    analysis_window_days, analysis_window_text = _analysis_window(cleaned)
    scope = _query_scope(cleaned)
    operation = question_operation(cleaned, scope)
    topic = explanation_topic(cleaned)
    parsed = ParsedQuery(
        location=location,
        location_suggestion=suggestion,
        location_candidate=extract_location_candidate(cleaned) if not location and not suggestion else None,
        coordinates=coordinates,
        weather_variable=variable,
        weather_summary_request=weather_summary_request,
        data_layer=layer,
        data_domain=domain,
        time_range=time_range,
        extreme_event_type=detect_extreme_event(cleaned),
        query_scope=scope,
        ranking_scope=_ranking_scope(cleaned),
        use_case=_use_case(cleaned),
        query_bucket=query_bucket(domain, variable, cleaned, operation, topic),
        question_operation=operation,
        ranking_direction=ranking_direction(cleaned),
        explanation_topic=topic,
        model_evidence_request=_model_evidence_request(cleaned, location, coordinates, time_range),
        analysis_window_days=analysis_window_days,
        analysis_window_text=analysis_window_text,
        grid_cell=_grid_cell(cleaned),
    )
    result = parsed.as_response()
    # Generic questions such as “what do expert weights mean?” are
    # methodology. A location/grid/lead-specific weight request must instead
    # reach the saved HYDRA output that contains the actual allocations.
    if result["model_evidence_request"] and result["explanation_topic"] == "weights":
        result["query_bucket"] = "hydra_short_range_forecast"
    if result["model_evidence_request"] and result["grid_cell"]:
        result["query_scope"] = "selected_location"
    # Learned labels are recorded as evidence only. Deterministic extraction
    # remains authoritative for every source and routing decision.
    result["trained_intent"] = predict_trained_intent(cleaned)
    return result

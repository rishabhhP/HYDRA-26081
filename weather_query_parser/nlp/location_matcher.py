"""Indian location matching with non-destructive fuzzy suggestions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

try:  # RapidFuzz is preferred, with a stdlib fallback for minimal deployments.
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - exercised only in dependency-light installs
    fuzz = process = None


# These names come from HYDRA's existing india-states.geojson / STATE_CITIES
# vocabulary. They are intentionally the parser's initial state/UT vocabulary.
INDIAN_STATES_AND_UTS = (
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
    "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
)

# Cities already used by the HYDRA live-weather markers or quick-location menu.
# Coordinates make a recognised city usable by WeatherGPT without a silent lookup.
MAJOR_CITIES: dict[str, tuple[float, float]] = {
    "Srinagar": (34.0837, 74.7973), "Shimla": (31.1048, 77.1734),
    "New Delhi": (28.6139, 77.2090), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Guwahati": (26.1445, 91.7362), "Patna": (25.5941, 85.1376),
    "Kolkata": (22.5726, 88.3639), "Ahmedabad": (23.0225, 72.5714),
    "Bhopal": (23.2599, 77.4126), "Bhubaneswar": (20.2961, 85.8245),
    "Mumbai": (19.0760, 72.8777), "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
    "Thiruvananthapuram": (8.5241, 76.9366), "Kochi": (9.9312, 76.2673),
    "Indore": (22.7196, 75.8577),
}

SUGGESTION_THRESHOLD = 82


def _contains(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<!\\w){re.escape(term)}(?!\\w)", text, re.IGNORECASE))


def _fuzzy_match(query: str, choices: tuple[str, ...]) -> tuple[str | None, float]:
    if fuzz and process:
        match = process.extractOne(query, choices, scorer=fuzz.WRatio)
        return (match[0], float(match[1])) if match else (None, 0.0)
    match = max(choices, key=lambda item: SequenceMatcher(None, query.casefold(), item.casefold()).ratio())
    return match, SequenceMatcher(None, query.casefold(), match.casefold()).ratio() * 100


def _candidate_phrases(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return [" ".join(words[start:start + size]) for size in range(1, 6) for start in range(len(words) - size + 1)]


def match_location(text: str) -> tuple[str | None, str | None, dict[str, float] | None]:
    """Match an exact location, or return only a non-destructive suggestion."""

    choices = tuple(INDIAN_STATES_AND_UTS) + tuple(MAJOR_CITIES)
    for choice in sorted(choices, key=len, reverse=True):
        if _contains(text, choice):
            coords = MAJOR_CITIES.get(choice)
            return choice, None, {"lat": coords[0], "lon": coords[1]} if coords else None

    best_choice: str | None = None
    best_score = 0.0
    for phrase in _candidate_phrases(text):
        choice, score = _fuzzy_match(phrase, choices)
        if score > best_score:
            best_choice, best_score = choice, score
    if best_choice and best_score >= SUGGESTION_THRESHOLD:
        return None, best_choice, None
    return None, None, None


def extract_location_candidate(text: str) -> str | None:
    """Keep an unmatched place phrase for a validated India-only geocoder."""
    match = re.search(r"\b(?:in|at|around|near)\s+([A-Za-z][A-Za-z' -]{1,60}?)(?=\s+(?:today|tomorrow|now|next|this|with|for|during)\b|[?.!,]|$)", text, re.I)
    return " ".join(match.group(1).split()) if match else None

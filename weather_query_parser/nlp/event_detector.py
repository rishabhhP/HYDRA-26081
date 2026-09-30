"""Detection for the three extreme-event classifiers supplied by nwpblend."""

from __future__ import annotations

import re


EVENT_PATTERNS = {
    "heavy_rain": (r"\b(?:very\s+)?heavy\s+(?:rain(?:fall)?s?|precipitation)\b", r"\bextremes+rains?\b"),
    "heatwave": (r"\bheat\s*wave\b", r"\bextreme\s+heat\b"),
    "high_wind": (r"\bhigh\s+(?:wind|winds)\b", r"\bstrong\s+(?:wind|winds)\b"),
}


def detect_extreme_event(text: str) -> str | None:
    matches = [(match.start(), event) for event, patterns in EVENT_PATTERNS.items() for pattern in patterns if (match := re.search(pattern, text, re.I))]
    return min(matches)[1] if matches else None

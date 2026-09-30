"""Time expressions mapped to HYDRA's forecast leads or explicit date ranges."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .models import TimeRange

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _range(start: date, end: date, text: str) -> TimeRange:
    return TimeRange(kind="date_range", start_date=start.isoformat(), end_date=end.isoformat(), text=text)


def _lead(hours: int, text: str) -> TimeRange:
    return TimeRange(kind="lead_hours", lead_hours=hours, text=text)


def _parse_absolute(text: str, today: date) -> TimeRange | None:
    match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if not match:
        return None
    try:
        requested = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None
    days = (requested - today).days
    if days < 0:
        return _range(requested, requested, match.group(0))
    if days == 0:
        return _lead(0, match.group(0))
    if days <= 3:
        return _lead((24, 48, 72)[days - 1], match.group(0))
    return _range(requested, requested, match.group(0))


def _parse_named_date(text: str, today: date) -> TimeRange | None:
    names = "|".join(MONTHS)
    patterns = (
        rf"\b(\d{{1,2}})\s+({names})\.?,?\s+(20\d{{2}})\b",
        rf"\b({names})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b",
    )
    for index, pattern in enumerate(patterns):
        if not (match := re.search(pattern, text, re.I)):
            continue
        day_text, month_text, year_text = match.groups() if index == 0 else (match.group(2), match.group(1), match.group(3))
        try:
            requested = date(int(year_text), MONTHS[month_text.casefold()], int(day_text))
        except ValueError:
            continue
        days = (requested - today).days
        if days < 0:
            return _range(requested, requested, match.group(0))
        if days == 0:
            return _lead(0, match.group(0))
        if days <= 3:
            return _lead((24, 48, 72)[days - 1], match.group(0))
        return _range(requested, requested, match.group(0))
    return None


def _parse_year(text: str) -> TimeRange | None:
    """Keep an explicit calendar year as an interval for historical data."""
    match = re.search(r"\b(20\d{2})\b", text)
    if not match:
        return None
    year = int(match.group(1))
    return _range(date(year, 1, 1), date(year, 12, 31), match.group(0))


def parse_time_range(text: str, today: date | None = None) -> TimeRange:
    """Parse point leads first; preserve true intervals instead of guessing one."""

    today = today or date.today()
    lowered = text.casefold()
    if re.search(r"\b(today|now)\b.*\b(tomorrow)\b|\b(tomorrow)\b.*\b(today|now)\b", lowered):
        return TimeRange(kind="ambiguous", text="today or tomorrow")
    if match := re.search(r"\b(?:from|between)\s+(20\d{2}-\d{1,2}-\d{1,2})\s+(?:to|and)\s+(20\d{2}-\d{1,2}-\d{1,2})\b", lowered):
        try:
            start, end = (date.fromisoformat(value) for value in match.groups())
            if start <= end:
                return _range(start, end, match.group(0))
        except ValueError:
            pass
    if absolute := _parse_absolute(lowered, today):
        return absolute
    if named_date := _parse_named_date(lowered, today):
        return named_date
    if year := _parse_year(lowered):
        return year
    if match := re.search(r"\b(?:at|in|for)\s*(0|24|48|72)\s*(?:h|hours?)\b", lowered):
        return _lead(int(match.group(1)), match.group(0))
    if "day after tomorrow" in lowered:
        return _lead(48, "day after tomorrow")
    if re.search(r"\btomorrow\b", lowered):
        return _lead(24, "tomorrow")
    if re.search(r"\byesterday\b", lowered):
        return _range(today - timedelta(days=1), today - timedelta(days=1), "yesterday")
    if re.search(r"\b(today|now|current)\b", lowered):
        return _lead(0, "today")
    if match := re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(days?|weeks?|months?)\b", lowered):
        count, unit = int(match.group(1)), match.group(2)
        days = count * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)
        return _range(today - timedelta(days=days), today - timedelta(days=1), match.group(0))
    if re.search(r"\b(?:last|previous)\s+week\b", lowered):
        return _range(today - timedelta(days=7), today - timedelta(days=1), "last week")
    if re.search(r"\b(?:last|previous)\s+month\b", lowered):
        return _range(today - timedelta(days=30), today - timedelta(days=1), "last month")
    if match := re.search(r"\b(in|next)\s+(\d+)\s+days?\b", lowered):
        qualifier, days_text = match.groups()
        days = int(days_text)
        if qualifier == "in" and days in (1, 2, 3):
            return _lead(days * 24, match.group(0))
        return _range(today + timedelta(days=1), today + timedelta(days=days), match.group(0))
    if "this weekend" in lowered:
        start = today + timedelta(days=(5 - today.weekday()) % 7)
        return _range(start, start + timedelta(days=1), "this weekend")
    if "next week" in lowered:
        start = today + timedelta(days=7 - today.weekday())
        return _range(start, start + timedelta(days=6), "next week")
    return TimeRange(kind="unspecified")

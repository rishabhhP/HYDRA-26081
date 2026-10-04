"""Date and period extraction for WeatherGPT (English + common Hinglish)."""
from __future__ import annotations

import calendar
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS.update({"sept": 9, "agust": 8, "febuary": 2, "janurary": 1, "octobar": 10, "decembar": 12, "novembar": 11})
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "a": 1, "an": 1, "couple of": 2, "few": 3}
SEASONS = {
    "monsoon": (6, 1, 9, 30), "southwest monsoon": (6, 1, 9, 30), "sw monsoon": (6, 1, 9, 30),
    "post monsoon": (10, 1, 12, 31), "post-monsoon": (10, 1, 12, 31), "northeast monsoon": (10, 1, 12, 31),
    "ne monsoon": (10, 1, 12, 31), "retreating monsoon": (10, 1, 12, 31),
    "pre monsoon": (3, 1, 5, 31), "pre-monsoon": (3, 1, 5, 31), "summer": (3, 1, 5, 31), "winter": (1, 1, 2, 28),
}
FUTURE_WORDS = r"\b(will|going to|gonna|forecast|expected|expect|tomorrow|next|upcoming|coming|hogi|hoga|hongi|predict(?:ed|ion)?\s+for)\b"
PAST_WORDS = r"\b(was|were|did|had|recorded|observed|happened|fell|last|past|previous|yesterday|ago|hui|hua|thi|tha|pichle|pichhle)\b"


def today() -> date:
    override = os.getenv("WEATHERGPT_TODAY")
    return date.fromisoformat(override) if override else date.today()


@dataclass
class TimeSpec:
    kind: str = "none"            # none | date | range | lead
    start: date | None = None
    end: date | None = None
    text: str | None = None
    lead_hours: int | None = None
    relative: bool = False
    granularity: str | None = None  # day | week | month | season | year | custom

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1 if self.start and self.end else 0

    @property
    def is_future(self) -> bool:
        return bool(self.start and self.start > today())

    @property
    def is_present(self) -> bool:
        return bool(self.start and self.start == today() and self.end == today())

    def as_dict(self) -> dict:
        out = asdict(self)
        out["start"] = self.start.isoformat() if self.start else None
        out["end"] = self.end.isoformat() if self.end else None
        return out


def _year_for(month: int, day: int, text: str, ref: date) -> int:
    future = bool(re.search(FUTURE_WORDS, text, re.I)) and not re.search(PAST_WORDS, text, re.I)
    try:
        candidate = date(ref.year, month, min(day, calendar.monthrange(ref.year, month)[1]))
    except ValueError:
        return ref.year
    if future:
        return ref.year if candidate >= ref else ref.year + 1
    return ref.year if candidate <= ref else ref.year - 1


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _num(token: str) -> int:
    token = token.lower().strip()
    return int(token) if token.isdigit() else WORD_NUMBERS.get(token, 1)


def _single_date(text: str, ref: date) -> tuple[date | None, tuple[int, int] | None]:
    """Find one explicit calendar date. Returns (date, span)."""
    patterns = (
        (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "ymd"),
        (r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", "dmy"),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_RE})\.?,?\s*(\d{{4}})?\b", "dMy"),
        (rf"\b({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?\b", "Mdy"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        try:
            if kind == "ymd":
                y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            elif kind == "dmy":
                d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
            elif kind == "dMy":
                d, m = int(match.group(1)), MONTHS[match.group(2).lower()]
                y = int(match.group(3)) if match.group(3) else _year_for(m, d, text, ref)
            else:
                m, d = MONTHS[match.group(1).lower()], int(match.group(2))
                if d > 31:
                    continue
                y = int(match.group(3)) if match.group(3) else _year_for(m, d, text, ref)
            return date(y, m, d), match.span()
        except (ValueError, KeyError):
            continue
    return None, None


def parse(text: str, ref: date | None = None) -> TimeSpec:
    ref = ref or today()
    t = " " + text.lower() + " "
    t = re.sub(r"\s+", " ", t)

    # ---------------- explicit ranges: from X to Y / between X and Y / X - Y
    range_match = re.search(r"\b(?:from|between)\s+(.+?)\s+(?:to|and|till|until|through|-)\s+(.+?)(?=[?.,!]|\s+(?:in|for|at|of|across|over)\s|\s*$)", t)
    if range_match:
        a, _ = _single_date(range_match.group(1), ref)
        b, _ = _single_date(range_match.group(2), ref)
        if a and not b:  # "from 1 to 15 September"
            day_only = re.match(r"\s*(\d{1,2})(?:st|nd|rd|th)?\s*$", range_match.group(1))
            b2, _ = _single_date(range_match.group(2), ref)
            if day_only and b2:
                a, b = date(b2.year, b2.month, int(day_only.group(1))), b2
        if not a:
            day_only = re.match(r"\s*(\d{1,2})(?:st|nd|rd|th)?\s*$", range_match.group(1))
            if day_only and b:
                a = date(b.year, b.month, int(day_only.group(1)))
        if not (a and b):
            ma = re.search(rf"\b({MONTH_RE})\b\s*(\d{{4}})?", range_match.group(1))
            mb = re.search(rf"\b({MONTH_RE})\b\s*(\d{{4}})?", range_match.group(2))
            if ma and mb:
                m1, m2 = MONTHS[ma.group(1)], MONTHS[mb.group(1)]
                y2 = int(mb.group(2)) if mb.group(2) else _year_for(m2, 1, t, ref)
                y1 = int(ma.group(2)) if ma.group(2) else (y2 if m1 <= m2 else y2 - 1)
                a, b = date(y1, m1, 1), _month_end(y2, m2)
        if a and b:
            if not re.search(r"\b\d{4}\b", range_match.group(1)) and re.search(r"\b\d{4}\b", range_match.group(2)):
                try:
                    a = a.replace(year=b.year)
                except ValueError:
                    pass
            if a > b:
                a, b = b, a
            return TimeSpec("range", a, b, range_match.group(0).strip(), granularity="custom")
    month_span = re.search(rf"\b({MONTH_RE})\b\s*(\d{{4}})?\s*(?:to|till|until|through|-|-)\s*({MONTH_RE})\b\s*(\d{{4}})?", t)
    if month_span:
        m1, m2 = MONTHS[month_span.group(1)], MONTHS[month_span.group(3)]
        y2 = int(month_span.group(4)) if month_span.group(4) else _year_for(m2, 1, t, ref)
        y1 = int(month_span.group(2)) if month_span.group(2) else (y2 if m1 <= m2 else y2 - 1)
        return TimeSpec("range", date(y1, m1, 1), _month_end(y2, m2), month_span.group(0).strip(), granularity="custom")
    compact = re.search(rf"\b(\d{{1,2}})\s*(?:-|to|-)\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\s*(\d{{4}})?", t)
    if compact:
        m = MONTHS[compact.group(3)]
        y = int(compact.group(4)) if compact.group(4) else _year_for(m, int(compact.group(1)), t, ref)
        try:
            return TimeSpec("range", date(y, m, int(compact.group(1))), date(y, m, int(compact.group(2))), compact.group(0).strip(), granularity="custom")
        except ValueError:
            pass

    # ---------------- relative days
    rel = (
        (r"\bday before yesterday\b|\bparso (?:hui|hua|tha|thi)\b", -2), (r"\bday after tomorrow\b|\bparso\b", 2),
        (r"\byesterday\b|\byday\b|\bkal\b(?=.*\b(?:hui|hua|thi|tha|was|were|did|padi|giri)\b)", -1),
        (r"\btomorrow\b|\btmrw\b|\btmr\b|\bkal\b", 1),
        (r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b|\baaj\b|\bright now\b|\bcurrently\b|\bnow\b|\babhi\b|\bat the moment\b|\bcurrent\b|\blive\b", 0),
    )
    for pattern, offset in rel:
        if re.search(pattern, t):
            d = ref + timedelta(days=offset)
            text_label = re.search(pattern, t).group(0).strip()
            return TimeSpec("date", d, d, text_label, lead_hours=offset * 24 if offset >= 0 else None, relative=True, granularity="day")

    lead = re.search(r"\+?\s*(24|48|72)\s*(?:h|hr|hrs|hours?)\b", t)
    if lead:
        hours = int(lead.group(1))
        d = ref + timedelta(days=hours // 24)
        return TimeSpec("lead", d, d, lead.group(0).strip(), lead_hours=hours, relative=True, granularity="day")
    nxt = re.search(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|a|few|couple of)\s+(day|days|week|weeks)\b", t)
    if nxt:
        n = _num(nxt.group(1)) * (7 if nxt.group(2).startswith("week") else 1)
        return TimeSpec("range", ref + timedelta(days=1), ref + timedelta(days=n), nxt.group(0).strip(), lead_hours=min(n, 3) * 24,
                        relative=True, granularity="custom")
    if re.search(r"\b(?:this|coming) weekend\b", t):
        sat = ref - timedelta(days=ref.weekday() - 5) if ref.weekday() >= 5 else ref + timedelta(days=5 - ref.weekday())
        return TimeSpec("range", sat, sat + timedelta(days=1), "this weekend", relative=True, granularity="custom")

    # ---------------- last N units
    last = re.search(r"\b(?:last|past|previous|pichle|pichhle|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|few|couple of)?\s*(day|days|din|week|weeks|hafte|fortnight|month|months|mahine|year|years)\b", t)
    if last:
        n = _num(last.group(1)) if last.group(1) else 1
        unit = last.group(2)
        days = n * (7 if unit.startswith(("week", "haft")) else 14 if unit == "fortnight" else 30 if unit.startswith(("month", "mahin")) else 365 if unit.startswith("year") else 1)
        if unit in ("week", "hafte") and n == 1 and re.search(r"\blast week\b|\bpichle hafte\b", t):
            start = ref - timedelta(days=ref.weekday() + 7)
            return TimeSpec("range", start, start + timedelta(days=6), "last week", relative=True, granularity="week")
        if unit in ("month", "mahine") and n == 1 and re.search(r"\blast month\b|\bpichle mahine\b|\bprevious month\b", t):
            first = (ref.replace(day=1) - timedelta(days=1)).replace(day=1)
            return TimeSpec("range", first, _month_end(first.year, first.month), "last month", relative=True, granularity="month")
        return TimeSpec("range", ref - timedelta(days=days), ref - timedelta(days=1), last.group(0).strip(), relative=True,
                        granularity="custom")
    ago = re.search(r"\b(\d+|one|two|three|four|five|six|seven|ten)\s+(day|days|week|weeks)\s+ago\b", t)
    if ago:
        d = ref - timedelta(days=_num(ago.group(1)) * (7 if ago.group(2).startswith("week") else 1))
        return TimeSpec("date", d, d, ago.group(0).strip(), relative=True, granularity="day")
    if re.search(r"\b(this|current|so far this|the) season\b|\bseason (so far|to date)\b|\bso far\b", t):
        m = ref.month
        if m in (6, 7, 8, 9, 10):
            start, end = date(ref.year, 6, 1), min(ref, date(ref.year, 9, 30))
            label = "this monsoon season"
        elif m in (11, 12):
            start, end, label = date(ref.year, 10, 1), ref, "this post-monsoon season"
        elif m in (1, 2):
            start, end, label = date(ref.year, 1, 1), ref, "this winter"
        else:
            start, end, label = date(ref.year, 3, 1), ref, "this pre-monsoon season"
        return TimeSpec("range", start, end, label, relative=True, granularity="season")
    if re.search(r"\bthis week\b", t):
        start = ref - timedelta(days=ref.weekday())
        return TimeSpec("range", start, ref, "this week", relative=True, granularity="week")
    if re.search(r"\bthis month\b", t):
        return TimeSpec("range", ref.replace(day=1), ref, "this month", relative=True, granularity="month")
    if re.search(r"\bthis year\b", t):
        return TimeSpec("range", date(ref.year, 1, 1), ref, "this year", relative=True, granularity="year")
    if re.search(r"\blast year\b", t):
        return TimeSpec("range", date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31), "last year", relative=True, granularity="year")
    since = re.search(r"\bsince\s+(.+?)(?=[?.,!]|$)", t)
    if since:
        a, _ = _single_date(since.group(1), ref)
        if not a:
            mm = re.search(rf"\b({MONTH_RE})\b\s*(\d{{4}})?", since.group(1))
            if mm:
                m = MONTHS[mm.group(1)]
                a = date(int(mm.group(2)) if mm.group(2) else _year_for(m, 1, t, ref), m, 1)
        if a:
            return TimeSpec("range", a, ref, since.group(0).strip(), granularity="custom")

    # ---------------- single explicit date
    d, _ = _single_date(t, ref)
    if d:
        return TimeSpec("date", d, d, d.isoformat(), granularity="day")

    # ---------------- parts of a month
    part = re.search(rf"\b(first|1st|second|2nd|third|3rd|fourth|4th|last|final)\s+week\s+of\s+({MONTH_RE})\s*(\d{{4}})?", t)
    if part:
        m = MONTHS[part.group(2)]
        y = int(part.group(3)) if part.group(3) else _year_for(m, 1, t, ref)
        idx = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}.get(part.group(1), -1)
        end_of_month = _month_end(y, m)
        start = end_of_month - timedelta(days=6) if idx < 0 else date(y, m, 1 + 7 * idx)
        return TimeSpec("range", start, min(start + timedelta(days=6), end_of_month), part.group(0).strip(), granularity="week")
    portion = re.search(rf"\b(early|beginning of|start of|mid|middle of|late|end of)\s*-?\s*({MONTH_RE})\s*(\d{{4}})?", t)
    if portion:
        m = MONTHS[portion.group(2)]
        y = int(portion.group(3)) if portion.group(3) else _year_for(m, 1, t, ref)
        end = _month_end(y, m)
        word = portion.group(1)
        if word in ("early", "beginning of", "start of"):
            a, b = date(y, m, 1), date(y, m, 10)
        elif word in ("mid", "middle of"):
            a, b = date(y, m, 11), date(y, m, 20)
        else:
            a, b = date(y, m, 21), end
        return TimeSpec("range", a, b, portion.group(0).strip(), granularity="custom")

    # ---------------- seasons
    for name in sorted(SEASONS, key=len, reverse=True):
        sm = re.search(rf"\b{re.escape(name)}\b(?:\s+(?:season\s+)?(?:of\s+)?(\d{{4}}))?", t)
        if sm:
            m1, d1, m2, d2 = SEASONS[name]
            y = int(sm.group(1)) if sm.group(1) else (ref.year if date(ref.year, m1, d1) <= ref else ref.year - 1)
            yr = re.search(r"\b(20\d\d|19\d\d)\b", t)
            if not sm.group(1) and yr:
                y = int(yr.group(1))
            return TimeSpec("range", date(y, m1, d1), date(y, m2, d2), sm.group(0).strip(), granularity="season")

    # ---------------- month (+year)
    mm = re.search(rf"\b(?:in|during|for|of|month of|throughout)?\s*({MONTH_RE})\b\.?\s*(\d{{4}})?", t)
    if mm and not re.search(rf"\b(?:may|march)\b", mm.group(1) or "") or (mm and (mm.group(2) or re.search(rf"\b(?:in|during|month of|throughout)\s+(?:may|march)\b", t))):
        m = MONTHS[mm.group(1)]
        y = int(mm.group(2)) if mm.group(2) else _year_for(m, 1, t, ref)
        if not mm.group(2) and date(y, m, 1) > ref and not re.search(FUTURE_WORDS, t):
            y -= 1
        if not mm.group(2) and re.search(rf"\b(?:last|previous|pichle)\s+{re.escape(mm.group(1))}\b", t) and m >= ref.month:
            y = ref.year - 1
        return TimeSpec("range", date(y, m, 1), _month_end(y, m), mm.group(0).strip(), granularity="month")

    year = re.search(r"\b(?:in|during|for|of|year)?\s*(19\d\d|20\d\d)\b", t)
    if year:
        y = int(year.group(1))
        return TimeSpec("range", date(y, 1, 1), min(date(y, 12, 31), max(ref, date(y, 1, 1))) if y == ref.year else date(y, 12, 31),
                        year.group(0).strip(), granularity="year")
    return TimeSpec()

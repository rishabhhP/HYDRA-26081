"""Slot extraction for WeatherGPT: locations, variables, statistics, thresholds, sources, advice topics."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches

from . import lexicon as L
from . import timeparse

STOP_FUZZY = {"rainfall", "weather", "temperature", "highest", "lowest", "forecast", "humidity", "between", "compare",
              "average", "maximum", "minimum", "yesterday", "tomorrow", "monsoon", "september", "october", "november",
              "december", "january", "february", "august", "accuracy", "states", "which", "where", "india", "model",
              "hydra", "experts", "expert", "weights", "explain", "should", "would", "could", "rainy", "about", "there",
              "their", "trend", "total", "today", "please", "predict", "predicted", "prediction", "region", "summer",
              "winter", "pressure", "degrees", "rained", "raining", "humid", "cloudy", "windy", "season", "across",
              "number", "recorded", "observed", "during", "showers", "storm", "storms", "thunder", "normal", "heavy",
              "excess", "deficit", "deficient", "chance", "probability", "interval", "uncertainty", "reliable"}


@dataclass
class Place:
    name: str
    kind: str                     # state | city | region | india
    state: str | None = None
    lat: float | None = None
    lon: float | None = None
    matched: str | None = None
    fuzzy: bool = False


@dataclass
class Slots:
    places: list[Place] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    statistic: str | None = None
    direction: str | None = None          # desc | asc
    top_k: int | None = None
    threshold: dict | None = None         # {op, value, unit}
    sources: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    model_metric: str | None = None
    lead_hours: int | None = None
    grid_cell: str | None = None
    coordinates: tuple[float, float] | None = None
    time: timeparse.TimeSpec = field(default_factory=timeparse.TimeSpec)
    followup: bool = False
    india_scope: bool = False
    state_scope: bool = False             # "which state", "states"
    compare: bool = False
    language: str = "en"

    def as_dict(self) -> dict:
        out = asdict(self)
        out["time"] = self.time.as_dict()
        return out


def normalise(text: str) -> str:
    t = text.replace("&", " and ").replace("’", "'")
    t = re.sub(r"[^\w\s.°+/:,?%-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def hinglish_to_english(text: str) -> tuple[str, bool]:
    words = text.split()
    hits = 0
    out = []
    for w in words:
        key = w.lower().strip("?,.!")
        if key in L.HINGLISH:
            hits += 1
            out.append(L.HINGLISH[key])
        else:
            out.append(w)
    return " ".join(out), hits >= 2


_CITY_KEYS = sorted(L.CITIES, key=len, reverse=True)
_STATE_KEYS = sorted(L.STATES, key=len, reverse=True)
_ALIAS_KEYS = sorted(L.STATE_ALIASES, key=len, reverse=True)
_FUZZY_POOL = {name.lower(): ("state", name) for name in L.STATES}
_FUZZY_POOL.update({name.lower(): ("city", name) for name in L.CITIES})
_FUZZY_POOL.update({alias: ("state", state) for alias, state in L.STATE_ALIASES.items() if len(alias) >= 5})


def _place_for(kind: str, name: str, matched: str, fuzzy=False) -> Place:
    if kind == "city":
        state, lat, lon = L.CITIES[name]
        return Place(name, "city", state, lat, lon, matched, fuzzy)
    return Place(name, "state", name, None, None, matched, fuzzy)


def extract_places(original: str) -> tuple[list[Place], bool]:
    text = " " + normalise(original).lower() + " "
    taken = [False] * len(text)
    found: list[tuple[int, Place]] = []

    def claim(start, end):
        if any(taken[start:end]):
            return False
        for i in range(start, end):
            taken[i] = True
        return True

    for region, key in sorted(L.REGIONS.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"(?<![\w]){re.escape(region)}(?![\w])", text):
            if claim(*m.span()):
                found.append((m.start(), Place(key, "region", None, None, None, region)))
    for name in _STATE_KEYS:
        for m in re.finditer(rf"(?<![\w]){re.escape(name.lower())}(?![\w])", text):
            if claim(*m.span()):
                found.append((m.start(), _place_for("state", name, m.group(0))))
    for name in _CITY_KEYS:
        for m in re.finditer(rf"(?<![\w]){re.escape(name.lower())}(?![\w])", text):
            if claim(*m.span()):
                # "Delhi" city and the state are the same thing for every dataset here
                place = _place_for("state", "Delhi", m.group(0)) if name in ("Delhi",) else _place_for("city", name, m.group(0))
                if name == "Chandigarh":
                    place = _place_for("state", "Chandigarh", m.group(0))
                found.append((m.start(), place))
    for alias in _ALIAS_KEYS:
        for m in re.finditer(rf"(?<![\w]){re.escape(alias)}(?![\w])", text):
            if alias in L.AMBIGUOUS_ALIASES:
                raw = re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", original, re.I)
                before = text[max(0, m.start() - 12):m.start()]
                upper = raw and raw.group(0).isupper()
                prep = re.search(r"\b(in|for|of|at|across|over|vs|versus|and|than|with)\s+$", before)
                if not (upper or (prep and alias not in ("up", "uk"))):
                    continue
            if claim(*m.span()):
                found.append((m.start(), _place_for("state", L.STATE_ALIASES[alias], m.group(0))))
    india = False
    for word in sorted(L.INDIA_WORDS, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w]){re.escape(word)}(?![\w])", text):
            if claim(*m.span()):
                india = True
    # fuzzy single words and bigrams for typos (keral, maharastra, karnatka, bangaluru)
    tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-z]{5,}", text)]
    grams = tokens + [(f"{a} {b}", s) for (a, s), (b, _) in zip(tokens, tokens[1:])]
    for gram, start in grams:
        if gram in STOP_FUZZY or any(taken[start:start + len(gram)]) or gram in timeparse.MONTHS:
            continue
        match = get_close_matches(gram, _FUZZY_POOL.keys(), n=1, cutoff=0.84)
        if match:
            kind, name = _FUZZY_POOL[match[0]]
            if claim(start, start + len(gram)):
                found.append((start, _place_for(kind, name, gram, fuzzy=True)))
    found.sort(key=lambda item: item[0])
    unique, seen = [], set()
    for _, place in found:
        key = (place.kind, place.name)
        if key not in seen:
            seen.add(key)
            unique.append(place)
    return unique, india


def extract_variables(text: str) -> list[str]:
    t = " " + text.lower() + " "
    hits = []
    for var, words in L.VARIABLES.items():
        for word in sorted(words, key=len, reverse=True):
            m = re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", t)
            if m:
                hits.append((m.start(), var))
                break
    hits.sort()
    out = []
    for _, var in hits:
        if var not in out:
            out.append(var)
    # "wind gust" contains "wind": keep the more specific
    if "wind_gust" in out and "wind" in out and re.search(r"\bgusts?\b", t) and not re.search(r"\bwind speed\b", t):
        out.remove("wind")
    if "thunderstorm" in out and "rainfall" in out and re.search(r"\bstorm rain\b", t):
        out.remove("thunderstorm")
    if "heat_stress" in out and "temperature" in out and re.search(r"\b(?:heat stress|wet bulb|feels like|heat index)\b", t):
        out.remove("temperature")
    if "water_vapour" in out and "soil_moisture" in out:
        out.remove("water_vapour")
    return out


def extract_statistic(text: str) -> str | None:
    t = " " + text.lower() + " "
    order = ("count", "trend", "total", "mean", "max", "min")
    best = None
    for stat in order:
        for word in L.STATISTICS[stat]:
            m = re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", t)
            if m and (best is None or (stat in ("count", "trend") and best[1] not in ("count", "trend"))):
                best = (m.start(), stat)
                break
        if best and best[1] in ("count", "trend"):
            break
    return best[1] if best else None


SPECIFIC_ASC = r"\b(lowest|least|minimum|driest|coldest|coolest|calmest|weakest|fewest|smallest|kam)\b"
SPECIFIC_DESC = r"\b(highest|maximum|wettest|hottest|warmest|windiest|strongest|heaviest|largest|biggest|rainiest|humidest)\b"


def extract_direction(text: str) -> str | None:
    t = text.lower()
    a, d = re.search(SPECIFIC_ASC, t), re.search(SPECIFIC_DESC, t)
    if a or d:
        if a and d:
            return "asc" if a.start() < d.start() else "desc"
        return "asc" if a else "desc"
    asc = re.search(r"\b(lowest|least|minimum|min|driest|coldest|coolest|calmest|weakest|bottom|fewest|smallest|kam|less|lower)\b", t)
    desc = re.search(r"\b(highest|most|maximum|max|wettest|hottest|warmest|windiest|strongest|top|heaviest|largest|biggest|sabse|zyada|jyada|more|higher|rainiest|humid(?:est)?)\b", t)
    if asc and desc:
        return "asc" if asc.start() < desc.start() else "desc"
    return "asc" if asc else "desc" if desc else None


def extract_top_k(text: str) -> int | None:
    m = re.search(r"\b(?:top|bottom|first|best|worst)\s+(\d{1,2}|three|five|ten|two|four)\b", text.lower()) or \
        re.search(r"\b(\d{1,2}|three|five|ten)\s+(?:states|places|cities|days|wettest|driest|hottest|coldest)\b", text.lower())
    if not m:
        return None
    return {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}.get(m.group(1), None) or int(m.group(1))


def extract_threshold(text: str) -> dict | None:
    t = text.lower()
    m = re.search(r"(more than|greater than|over|above|exceed(?:ed|ing|s)?|at least|>=|>|beyond|crossing|crossed|less than|below|under|<=|<|at most|upto|up to)\s*(-?\d+(?:\.\d+)?)\s*(mm|cm|°c|degrees?|c\b|%|percent|km/?h|kmph|m/s|hpa)?", t)
    if not m:
        return None
    op_word = m.group(1)
    op = ">=" if op_word in ("at least", ">=") else "<=" if op_word in ("at most", "<=", "upto", "up to") else \
         "<" if op_word in ("less than", "below", "under", "<") else ">"
    value = float(m.group(2))
    unit = (m.group(3) or "").replace("degrees", "°c").replace("degree", "°c").replace("percent", "%")
    if unit == "cm":
        value, unit = value * 10, "mm"
    return {"op": op, "value": value, "unit": unit or None, "text": m.group(0)}


def extract_sources(text: str) -> list[str]:
    t = " " + text.lower() + " "
    out = []
    for name, words in L.SOURCE_HINTS.items():
        if any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", t) for w in words):
            out.append(name)
    return out


def extract_advice(text: str) -> list[str]:
    t = " " + text.lower() + " "
    return [topic for topic, words in L.ADVICE_TOPICS.items()
            if any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", t) for w in words)]


MODEL_METRICS = {
    "mae": r"\bmae\b|mean absolute error|absolute error",
    "rmse": r"\brmse\b|root mean square",
    "bias": r"\bbias\b|over[- ]?predict|under[- ]?predict|over[- ]?estimat|under[- ]?estimat",
    "coverage": r"\bcoverage\b|interval|band|uncertainty|confidence|range",
    "recall": r"\brecall\b|\bpod\b|hit rate|detect|catch|caught|miss(?:ed)?",
    "precision": r"\bprecision\b|false alarm",
    "skill": r"\bskill\b|better than|beat|outperform|vs persistence|against persistence|compared to persistence|baseline",
    "accuracy": r"accura|how good|how well|reliab|trust|error|performance|perform|correct",
}


def extract_model_metric(text: str) -> str | None:
    for name, pattern in MODEL_METRICS.items():
        if re.search(pattern, text, re.I):
            return name
    return None


FOLLOWUP = r"^\s*(?:and|what about|how about|same for|also|ok and|okay and|then|now|aur|what of|and in|and for|compare (?:it|that|this)|for)\b|\b(?:there|that day|that state|that period|same period|same day|same place|it|those days|then)\s*\??\s*$"


def extract(text: str, ref=None) -> Slots:
    original = text
    english, hinglish = hinglish_to_english(original)
    base = normalise(english)
    places, india = extract_places(original + " " + (english if english != original else ""))
    s = Slots()
    s.places = places
    s.india_scope = india or any(p.kind == "region" for p in places)
    s.variables = extract_variables(base)
    s.statistic = extract_statistic(base)
    s.direction = extract_direction(original + " " + base)
    if s.direction == "asc" and s.statistic == "max":
        s.statistic = "min"
    s.top_k = extract_top_k(base)
    s.threshold = extract_threshold(base)
    s.sources = extract_sources(base)
    s.advice = extract_advice(original + " " + base)
    s.model_metric = extract_model_metric(base)
    s.time = timeparse.parse(original if not hinglish else original + " " + english, ref)
    s.lead_hours = s.time.lead_hours
    cell = re.search(r"\b(?:grid(?:\s+cell)?|cell)\s*(?:id\s*)?#?\s*(\d{3,6})\b", base, re.I)
    s.grid_cell = cell.group(1) if cell else None
    coord = re.search(r"\b(-?\d{1,2}\.\d+)\s*(?:°?\s*n)?\s*[,/ ]\s*(-?\d{2,3}\.\d+)\s*(?:°?\s*e)?\b", base, re.I)
    if coord:
        lat, lon = float(coord.group(1)), float(coord.group(2))
        if 5 <= lat <= 38 and 66 <= lon <= 99:
            s.coordinates = (lat, lon)
    s.state_scope = bool(re.search(r"\b(?:which|what|which all|konsa|kaunsa|kaun sa|list|rank|ranking|top|all)\b.*\bstates?\b|\bstates?\b.*\b(?:highest|lowest|most|least|wettest|driest|hottest|coldest|rank)|\bstate[- ]?wise\b|\bacross (?:all )?states\b|\bamong states\b", base, re.I))
    s.compare = bool(re.search(r"\b(?:compare|comparison|versus|vs\.?|compared|difference between|differ|(?<!more )(?<!less )(?<!greater )than|against|better|worse|wetter|drier|hotter|colder|warmer|cooler|windier)\b", base, re.I)) \
        or (len([p for p in places if p.kind in ("state", "city")]) >= 2 and re.search(r"\b(?:and|or)\b", base))
    s.followup = bool(re.search(FOLLOWUP, base, re.I)) or (len(base.split()) <= 4 and not s.variables)
    s.language = "hinglish" if hinglish else "en"
    return s

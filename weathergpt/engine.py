"""WeatherGPT v2 engine: understand -> resolve -> fetch -> reason -> answer.

answer(question, selection, session_id) returns the /api/weathergpt payload, or None to let the prototype's
original WeatherGPT path handle the question (grid-cell archive, 2014 grid, long-window model rankings).
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from . import intent as I
from . import lexicon as L
from .slots import Place, Slots, extract
from .timeparse import today as _today

SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_SESSIONS = 500


@dataclass
class Answer:
    lead: str
    sections: list = field(default_factory=list)   # (title, text)
    ranks: list = field(default_factory=list)      # strings
    sources: list = field(default_factory=list)    # {"file"/"name", "method"}
    followups: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    handler: str = ""

    def text(self) -> str:
        lines = [self.lead]
        lines += [f"{i}. {r}" for i, r in enumerate(self.ranks, 1)]
        for title, body in self.sections:
            if body:
                lines.append(f"{title}: {body}")
        return "\n".join(lines)


# ------------------------------------------------------------------ intent resolution (model + rules)
def resolve_intent(q: str, s: Slots) -> tuple[str, float, list]:
    pred = I.predict(q)
    intent, conf, ranked = pred["intent"] or "observed", pred["confidence"], pred["ranked"]
    t = q.lower()
    rules = [
        (r"^\s*(hi+|hello+|hey+|namaste|yo|good (morning|evening|afternoon))\W*$", "greeting"),
        (r"^\s*(thanks?|thank you|thx|bye|goodbye|ok thanks|shukriya|dhanyavad)\b.{0,20}$", "thanks_bye"),
        (r"\b(radar|satellite|imagery)\b", "imagery"),
        (r"\b(what can you (do|answer)|what (questions|kind of questions) can i ask|your capabilities|example questions|sample (queries|questions))\b", "capabilities"),
        (r"\b(source of (this|that|your)|which (file|dataset) (did|do) you use|provenance|cite (your )?source|is this (observed|live|archived|from))\b", "provenance"),
    ]
    for pattern, name in rules:
        if re.search(pattern, t):
            return name, max(conf, 0.9), ranked
    foreign = re.search(r"\b(london|paris|new york|tokyo|dubai|singapore|bangkok|beijing|sydney|toronto|berlin|moscow|"
                        r"usa|america|uk|england|france|china|japan|pakistan|nepal|bangladesh|sri lanka|australia|canada|germany|europe|africa)\b", t)
    if foreign and not s.places:
        return "out_of_scope", 0.95, ranked
    weather_words = r"\b(weather|mausam|rain|rainfall|forecast|temperature|temp|hot|cold|humid|wind|storm|cloud|monsoon|imd|era5|ecmwf|hydra|model|expert|gate|interval|radar|satellite|data|dataset|source|normal|range|percent|average|states?|band|trust|uncertain|spread|confidence|deficit|excess|heat|flood|umbrella|barish|garmi|thand|degrees|mm|climate|season|drought|cyclone|snow|fog|dry|wet|sunny|thunder|lightning|gust|pressure|cape|replay|accuracy|predict)"
    if not s.places and not s.variables and not s.india_scope and not re.search(weather_words, t) \
            and intent not in ("greeting", "thanks_bye", "capabilities"):
        return "out_of_scope", max(conf, 0.8), ranked
    if re.search(r"\b(alerts?|warnings?|watch(es)?|tiers?|heavy[- ]rain risk|chance of heavy|probability of heavy)\b", t) \
            and s.time.start and s.time.start <= _today():
        return "extremes", max(conf, 0.8), ranked
    modelish = bool(re.search(r"\b(hydra|model|forecast(s|ing)? (error|skill|accuracy)|backtest|replay|gate|expert)\b", t))
    if re.search(r"\b(cumulative|season(al)? (total|rainfall|so far|to date)|so far this season)\b", t) and not s.state_scope:
        return "anomaly", max(conf, 0.75), ranked
    if re.search(r"\b(predicted|forecast|hydra|model)\s+(vs\.?|versus|against|and)\s+(observed|actual|real)\b|\b(observed|actual)\s+(vs\.?|versus)\s+(predicted|forecast|hydra)\b", t):
        return "model_vs_actual", max(conf, 0.8), ranked
    if re.search(r"^\s*(why|kyun|kyu|what caused|reason for|what explains|what drives|what made)\b", t) or re.search(r"\bkyun\b|\bkyu\b", t):
        if re.search(r"\b(experts?|disagree|interval|spread|band|uncertain)\b", t) and not re.search(r"\b(miss|under|over)", t):
            return "uncertainty", max(conf, 0.85), ranked
        return "why", max(conf, 0.85), ranked
    if modelish and re.search(r"\b(weights?|allocation|dominat|which expert|gate (change|move|static|dynamic)|static gate)\b", t) \
            and not re.search(r"^\s*(what is|what are|explain|how does)\b.*\b(gate|weights?)\b(?!.*\b(in|for)\b)", t):
        return "model_weights", max(conf, 0.8), ranked
    if modelish and re.search(r"\b(how (well|good|accurate|reliable)|accura|reliab)\b", t) and not re.search(r"\bon \d|\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t):
        return "model_accuracy", max(conf, 0.8), ranked
    if modelish and (s.time.start or re.search(r"\b(miss(es|ed)?|vs\.? (actual|observed)|versus (actual|observed)|predicted vs|did (hydra|the model) (predict|catch|get))\b", t)) \
            and re.search(r"\b(did|vs\.?|versus|actual|observed|right|close|miss|misses|missed|catch|caught|predict(ed)?|wrong|worst)\b", t) \
            and not re.search(r"\b(accura|mae|rmse|recall|precision|bias|skill)\b", t):
        return "model_vs_actual", max(conf, 0.75), ranked
    if modelish and s.model_metric in ("mae", "rmse", "bias", "recall", "precision", "skill", "accuracy") \
            and not re.search(r"^\s*(how does|what is the|explain)\b.*\b(work|architecture|gate)\b", t):
        return "model_accuracy", max(conf, 0.75), ranked
    if s.threshold or re.search(r"\b(how many|number of|count|kitne)\b.*\b(days?|din)\b|\bdays (with|when|above|over|below)\b", t):
        if not modelish or intent == "threshold_days":
            return "threshold_days", max(conf, 0.75), ranked
    if s.advice and re.search(r"\b(should|can (i|we)|is it (safe|ok|good|too)|need|good (time|day)|risk|le jaun|theek|safe|best time|postpone|plan)\b", t):
        return "advice", max(conf, 0.75), ranked
    if re.search(r"\b(normal|departure|deficit|deficient|excess|anomal|than usual|percent of normal|imd category)\b", t) \
            and not s.state_scope and not re.search(r"\bwhich states?\b", t):
        if intent not in ("rank_places", "threshold_days", "why", "compare"):
            return "anomaly", max(conf, 0.7), ranked
    if re.search(r"\b(trend|increasing|decreasing|getting (wetter|drier|hotter|colder)|over time|evolv|progress|picking up|weaken|strengthen|badh rahi)\b", t):
        return "trend", max(conf, 0.7), ranked
    if s.direction and re.search(r"\b(which|what|on which)\s+(day|date)s?\b|\bwhen (did|was)\b|\b(wettest|driest|hottest|coldest|rainiest|heaviest) (day|date)s?\b|\bpeak (rain(fall)?|day)\b|\bkab\b", t) \
            and not s.state_scope:
        return "rank_days", max(conf, 0.75), ranked
    if s.state_scope or (re.search(r"\bwhere\b|\bkahan\b|\bkis (state|rajya)\b", t) and s.direction and not [p for p in s.places if p.kind in ("state", "city")]):
        if not modelish:
            return "rank_places", max(conf, 0.75), ranked
    n_places = len([p for p in s.places if p.kind in ("state", "city", "region")])
    if s.compare and (n_places >= 2 or re.search(r"\b(vs\.?|versus|compared? (to|with))\b", t)):
        return "compare", max(conf, 0.7), ranked
    if intent == "compare" and n_places < 2 and not re.search(r"\b(vs\.?|versus|compared? (to|with)|or)\b", t):
        intent = "observed"
    if intent == "extremes" and re.search(r"\b(will|coming|expected|forecast|likely|tomorrow|next|upcoming|hogi)\b", t):
        intent = "forecast"
    # time decides between observed / current / forecast
    if intent in ("observed", "current_weather", "forecast"):
        if s.time.start and s.time.start > _today():
            intent = "forecast"
        elif s.time.relative and s.time.kind == "date" and s.time.lead_hours == 0 and not s.time.start < _today():
            intent = "current_weather" if re.search(r"\b(now|right now|current|currently|live|at the moment|abhi|today|aaj)\b", t) else intent
        elif s.time.start and s.time.end and s.time.end < _today():
            intent = "observed"
    return intent, conf, ranked


# ------------------------------------------------------------------ follow-up memory
def merge_followup(s: Slots, intent: str, conf: float, q: str, last: dict | None) -> tuple[Slots, str, bool]:
    if not last:
        return s, intent, False
    from datetime import date
    from .timeparse import TimeSpec
    short = len(q.split()) <= 4
    explicit_switch = bool(re.match(r"^\s*(and|what about|how about|same for|aur|also|now|ok and)\b", q, re.I)) and len(q.split()) <= 6
    if last.get("intent") and intent not in ("greeting", "thanks_bye", "capabilities") and (explicit_switch or (s.followup and short and conf < 0.6)):
        intent = last["intent"]
    if s.india_scope or s.state_scope or intent in ("greeting", "thanks_bye", "capabilities", "data_coverage", "out_of_scope",
                                                     "imagery", "provenance") or (intent == "rank_places" and not explicit_switch):
        return s, intent, False
    inherits = False
    pronoun = bool(re.search(r"\b(there|that state|that city|that place|it|its|same place|wahan|udhar)\b", q, re.I))
    if not s.places and last.get("places") and (s.followup or short or pronoun or explicit_switch):
        s.places = [Place(**p) for p in last["places"]]
        inherits = True
    if not s.variables and last.get("variables") and (s.followup or short or explicit_switch or pronoun):
        s.variables = list(last["variables"])
        inherits = True
    if not s.time.start and last.get("focus_date") and re.search(r"\b(it|that day|that date|then|that peak|that one)\b", q, re.I):
        d = date.fromisoformat(last["focus_date"])
        s.time = TimeSpec("date", d, d, last["focus_date"], granularity="day")
        inherits = True
    if not s.time.start and last.get("time") and last["time"].get("start") and (s.followup or short or explicit_switch or pronoun) \
            and intent not in ("current_weather",):
        lt = last["time"]
        s.time = TimeSpec(lt["kind"], date.fromisoformat(lt["start"]), date.fromisoformat(lt["end"]), lt["text"], lt["lead_hours"],
                          lt["relative"], lt["granularity"])
        inherits = True
    if inherits and intent == "compare" and len(s.places) < 2:
        intent = "observed"
    return s, intent, inherits


def remember(session_id: str | None, intent: str, s: Slots, focus_date: str | None = None) -> None:
    if not session_id:
        return
    SESSIONS[session_id] = {"intent": intent, "places": [p.__dict__ for p in s.places], "variables": s.variables,
                            "time": s.time.as_dict(), "at": time.time(), "focus_date": focus_date}
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


# ------------------------------------------------------------------ legacy hand-off
def defer_to_legacy(q: str, s: Slots, selection: dict | None = None) -> bool:
    """Keep questions owned by the original evidence-specific route there.

    WeatherGPT v2 broadens state, city, date, and follow-up coverage.  The
    existing route remains the authoritative path for archive grid cells,
    literal coordinates, and selected ECMWF/production evidence where it can
    preserve the exact supplied grid context.
    """
    t = q.lower()
    if s.grid_cell:
        return True
    if re.search(r"(?<!\d)-?\d{1,2}(?:\.\d+)?\s*,\s*-?\d{1,3}(?:\.\d+)?(?!\d)", t):
        return True
    if re.search(r"\b2014\b", t) and "rain" in t:
        return True
    if re.search(r"\b(one|1|\d+)\s+years?\s+of\s+data\b|\bprone to\b", t):
        return True
    if re.search(r"\b(aifs|gfs|gefs|icon|weathernext)\b", t):
        return True
    if selection and selection.get("source") in {"ecmwf", "production"} and not s.places and not s.time.start:
        return True
    return False


def selection_place(selection: dict | None) -> Place | None:
    if not selection:
        return None
    state = selection.get("state")
    if state and state in L.STATES:
        return Place(state, "state", state, matched="map selection")
    lat, lon = selection.get("latitude"), selection.get("longitude")
    if lat is None or lon is None:
        return None
    try:
        from backend import data as D
        st = D.state_at(float(lat), float(lon))
    except Exception:
        st = None
    if st:
        return Place(f"selected map point ({float(lat):.2f}° N, {float(lon):.2f}° E)", "city", st, float(lat), float(lon), "map selection")
    return None


def answer(question: str, selection: dict | None = None, session_id: str | None = None) -> dict | None:
    from . import handlers as HD

    q = " ".join(question.strip().split())
    s = extract(q)
    if defer_to_legacy(q, s, selection):
        return None
    intent, conf, ranked = resolve_intent(q, s)
    s, intent, inherited = merge_followup(s, intent, conf, q, SESSIONS.get(session_id) if session_id else None)
    if s.coordinates and not s.places:
        lat, lon = s.coordinates
        try:
            from backend import data as D
            st = D.state_at(lat, lon)
        except Exception:
            st = None
        s.places = [Place(f"{lat:.2f}° N, {lon:.2f}° E", "city", st, lat, lon, "coordinates")]
    used_selection = False
    if not s.places and not s.india_scope and intent in HD.NEEDS_PLACE:
        place = selection_place(selection)
        if place:
            s.places = [place]
            used_selection = True
    handler = HD.HANDLERS.get(intent, HD.h_observed)
    try:
        ans: Answer = handler(q, s)
    except Exception as exc:  # never break the chat; report what failed
        import logging
        logging.getLogger("weathergpt").exception("handler %s failed", intent)
        ans = Answer(f"I understood this as a {intent.replace('_', ' ')} question but could not complete it ({type(exc).__name__}).",
                     [("Try", "Name a state or city and a date or period, e.g. 'rainfall in Kerala on 20 September 2026'.")],
                     handler="error")
    if used_selection:
        ans.sections.append(("Location", f"No place was named, so I used the map selection: {s.places[0].name}."))
    if inherited:
        ans.sections.append(("Context", "Carried over the place/period/variable from your previous question."))
    remember(session_id, intent, s, ans.data.get("focus_date"))
    return {
        "answer": ans.text(),
        "engine": f"WeatherGPT v2 · {ans.handler or intent}",
        "evidence": ans.sources or [{"file": "weathergpt", "method": "WeatherGPT v2 reasoning"}],
        "followups": ans.followups[:4],
        "data": ans.data,
        "parsed_query": legacy_view(intent, s),
        "weathergpt": {"intent": intent, "confidence": round(conf, 3), "candidates": [(n, round(p, 3)) for n, p in ranked],
                       "slots": s.as_dict(), "inherited_context": inherited},
    }


def legacy_view(intent: str, s: Slots) -> dict:
    """Shape the frontend's 'Parsed intent' footer already understands."""
    loc = s.places[0].name if s.places else ("India" if s.india_scope else None)
    return {"location": loc, "weather_variable": s.variables[0] if s.variables else None, "query_bucket": intent,
            "question_operation": intent, "ranking_direction": "ascending" if s.direction == "asc" else None,
            "time_range": {"text": s.time.text, "start_date": s.time.start.isoformat() if s.time.start else None,
                           "end_date": s.time.end.isoformat() if s.time.end else None}}

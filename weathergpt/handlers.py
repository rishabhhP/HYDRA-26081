"""One handler per intent. Each returns an Answer with numbers, meaning, source and coverage."""
from __future__ import annotations

import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import datahub as H
from . import insights as N
from . import lexicon as L
from . import liveclient as LC
from . import values as V
from .engine import Answer
from .slots import Place, Slots
from .timeparse import TimeSpec, today

f = N.f
NEEDS_PLACE = {"current_weather", "forecast", "observed", "rank_days", "threshold_days", "trend", "anomaly", "advice",
               "why", "model_vs_actual", "extremes"}


def _src(s: V.Series) -> dict:
    return {"file": s.source, "method": s.source_key}


def _places(s: Slots, limit: int = 4) -> list[Place]:
    out = [p for p in s.places if p.kind in ("state", "city", "region")]
    if not out and s.india_scope:
        out = [Place("India", "india")]
    return out[:limit]


def _var(s: Slots, default="rainfall") -> str:
    return s.variables[0] if s.variables else default


def _default_window(var: str, ts: TimeSpec) -> TimeSpec:
    """No period given: use the most recent day of the best offline source for rainfall, today otherwise."""
    if ts.start:
        return ts
    if var == "rainfall" and H.imd_coverage():
        d = H.imd_coverage()[1]
        return TimeSpec("date", d, d, f"latest IMD day ({d})", granularity="day")
    return TimeSpec("date", today(), today(), "today", lead_hours=0, relative=True, granularity="day")


def _period(ts: TimeSpec) -> str:
    if not ts.start:
        return ""
    return f"on {ts.start:%d %b %Y}" if ts.start == ts.end else f"from {ts.start:%d %b %Y} to {ts.end:%d %b %Y}"


def _coverage_hint(var: str) -> str:
    parts = []
    if var == "rainfall":
        c = H.imd_coverage()
        r = H.replay_coverage()
        if c:
            parts.append(f"IMD observed state rainfall {c[0]} to {c[1]}")
        if r:
            parts.append(f"ERA5 state rainfall {r[0]} to {r[1]}")
    d = H.ecmwf_date()
    if d and var != "rainfall":
        parts.append(f"ECMWF analysis for {d}")
    parts.append("any other date through the live Open-Meteo connection (needs internet on the server)")
    return "; ".join(parts) + "."


def _unavailable(place: str, var: str, ts: TimeSpec, why: str) -> Answer:
    return Answer(f"I couldn't get {var.replace('_', ' ')} for {place} {_period(ts)}.",
                  [("Reason", why), ("Available", _coverage_hint(var))],
                  followups=[f"What was the rainfall in {place} last week?", "What data do you have?"], handler="unavailable")


# ================================================================== observed values
def describe_series(s: V.Series, ts: TimeSpec, stat: str | None) -> tuple[str, list, list]:
    """Lead sentence + sections + ranks for one series."""
    unit, var = s.unit, s.variable
    sections, ranks = [], []
    single = len(s.values) == 1
    if var == "rainfall":
        total, mean = s.stat("total"), s.stat("mean")
        if single:
            v = s.values[0]
            lead = f"{s.place}: {f(v)} {unit} of rainfall on {s.dates[0]:%d %b %Y}."
            sections.append(("What this means", N.rain_meaning(v, s.state_mean)))
        else:
            arr = s.arr()
            wet = int(np.nansum(arr >= 2.5))
            imax = int(np.nanargmax(arr))
            want = stat or "total"
            headline = {"mean": f"averaged {f(mean)} {unit}/day", "max": f"peaked at {f(arr[imax])} {unit} on {s.dates[imax]:%d %b}",
                        "min": f"had a minimum of {f(np.nanmin(arr))} {unit}/day"}.get(want, f"received {f(total)} {unit} in total")
            lead = f"{s.place} {headline} {_period(TimeSpec('range', s.dates[0], s.dates[-1]))}."
            sections.append(("Summary", f"total {f(total)} {unit}, daily mean {f(mean)} {unit}, wettest day {s.dates[imax]:%d %b} "
                                        f"({f(arr[imax])} {unit}), {wet} of {len(arr)} days with at least 2.5 {unit}."))
            sections.append(("What this means", N.rain_meaning(mean, True).replace("mm/day", "mm/day on average") if s.state_mean else
                             f"an average of {f(mean)} mm/day; the wettest day alone was {N.rain_class(arr[imax])}."))
            if len(arr) <= 10:
                ranks = [f"{d:%a %d %b}: {f(v)} {unit}" for d, v in zip(s.dates, arr)]
        if s.source_key == "imd":
            rows = s.extra["rows"]
            if single:
                r = rows.iloc[0]
                sections.insert(1, ("Versus normal", f"normal {f(r['daily_normal_mm'])} mm; " + N.departure_meaning(r["daily_departure_pct"], r["daily_category"])))
                sections.append(("Season so far", f"cumulative {f(r['cumulative_actual_mm'], 0)} mm against a normal of "
                                                  f"{f(r['cumulative_normal_mm'], 0)} mm ({N.departure_meaning(r['cumulative_departure_pct'], r['cumulative_category'])})"))
            else:
                normal = rows["daily_normal_mm"].sum()
                dep = (total - normal) / normal * 100 if normal else None
                sections.insert(1, ("Versus normal", f"normal for these days {f(normal)} mm; " + N.departure_meaning(dep, _category(dep))))
        if s.source_key == "replay":
            rows = s.extra["rows"]
            hyd = rows["hydra_mm"].sum() if not single else rows["hydra_mm"].iloc[0]
            sections.append(("HYDRA", f"HYDRA's +24h forecast for the same {'day' if single else 'days'} was {f(hyd)} {unit}"
                                      + (" in total." if not single else ".") + " Ask 'how close was HYDRA' for the full comparison."))
            if single and rows["actual_local_max_mm"].notna().iloc[0]:
                sections.append(("Local extreme", f"wettest ERA5 grid cell inside {s.place}: {f(rows['actual_local_max_mm'].iloc[0])} mm "
                                                  f"({N.rain_class(rows['actual_local_max_mm'].iloc[0])}). The state mean hides this."))
    else:
        agg = s.extra.get("aggregate", {})
        bits = [f"{v['label']} {f(v['value'])} {v['unit']}" for v in agg.values() if v.get("value") is not None]
        value = s.stat(stat or ("max" if var == "temperature" else "mean"))
        lead = f"{s.place} {_period(TimeSpec('range', s.dates[0], s.dates[-1]))}: " + ("; ".join(bits) if bits else f"{f(value)} {s.unit}") + "."
        kind = "max" if var == "temperature" else "mean"
        sections.append(("What this means", N.generic_meaning(var, value, s.unit, kind, s.state_mean)))
        if 1 < len(s.values) <= 10:
            ranks = [f"{d}: {f(v)} {s.unit}" for d, v in zip(s.dates, s.values)]
    if s.note.strip():
        sections.append(("Coverage", s.note.strip()))
    sections.append(("Source", s.source))
    return lead, sections, ranks


def _category(dep: float | None) -> str | None:
    if dep is None:
        return None
    return ("large_excess" if dep >= 60 else "excess" if dep >= 20 else "normal" if dep > -20 else "deficient" if dep > -60
            else "large_deficient" if dep > -99.5 else "no_rain")


def h_observed(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    var = _var(s)
    ts = _default_window(var, s.time)
    if ts.start and ts.start > today():
        return h_forecast(q, s)
    if var != "rainfall" and H.ecmwf_date() and ts.start == ts.end == H.ecmwf_date():
        return _ecmwf_answer(places, s.variables or [var], ts)
    leads, sections, ranks, sources = [], [], [], []
    for place in places:
        for v in (s.variables or [var])[:3]:
            series = V.series(place, v, ts, s.sources)
            if not series.ok:
                if len(places) == 1 and len(s.variables or [var]) == 1:
                    return _unavailable(place.name, v, ts, series.message or "No source covers this place and period.")
                sections.append((f"{place.name} {v}", series.message or "not available"))
                continue
            lead, sec, rk = describe_series(series, ts, s.statistic)
            leads.append(lead)
            sections += sec if len(places) == 1 else [x for x in sec if x[0] in ("What this means", "Versus normal")]
            ranks += rk if len(places) == 1 else []
            sources.append(_src(series))
    if not leads:
        return _unavailable(places[0].name, var, ts, "None of the sources returned data.")
    ans = Answer(" ".join(leads), sections, ranks, sources, handler="observed")
    p = places[0].name
    ans.followups = [f"Was {p} above normal {ts.text or 'then'}?", f"Which day was wettest in {p} {ts.text or ''}".strip() + "?",
                     f"How close was HYDRA in {p}?", f"Compare {p} with a neighbouring state"]
    if s.advice:
        ans.sections += [("Advice", t) for t in N.advice(s.advice, places[0].state, V.Series.stat(series, "mean") if series.ok and v == "rainfall" else None, None, None, None, series.state_mean)]
    return ans


def _ecmwf_answer(places: list[Place], variables: list[str], ts: TimeSpec) -> Answer:
    leads, sections = [], []
    for place in places:
        for var in variables[:4]:
            r = V.ecmwf_value(place, var)
            if not r:
                continue
            val = r["value"] * (100 if r["unit"] == "fraction" else 1)
            unit = "%" if r["unit"] == "fraction" else r["unit"]
            extra = f" (range {f(r['min'])} to {f(r['max'])} {unit} across {r.get('cells', 'its')} grid cells)" if "max" in r and unit != "%" else ""
            leads.append(f"{place.name} {var.replace('_', ' ')} on {r['date']}: {f(val)} {unit}{extra}.")
            sections.append(("What this means", N.generic_meaning(var, val, unit, "mean", place.kind != "city")))
    if not leads:
        return _unavailable(places[0].name, variables[0], ts, "The ECMWF snapshot has no field for this variable.")
    sections += [("Coverage", "ECMWF IFS zero-hour analysis on its grid; state values average every 0.25° cell inside the state."),
                 ("Source", "ECMWF IFS analysis snapshot (forecast_india_2026-09-24_0h_clean.csv)")]
    return Answer(" ".join(leads), sections, sources=[{"file": "forecast_india_2026-09-24_0h_clean.csv", "method": "ECMWF analysis"}],
                  followups=["Which state had the highest CAPE that day?", "How humid was Kerala that day?"], handler="ecmwf")


# ================================================================== current conditions and forecasts
def h_current(q: str, s: Slots) -> Answer:
    places = _places(s)
    if not places:
        return Answer("Which place should I check? Name a state, city, or select a point on the map.", handler="clarify",
                      followups=["Weather in Mumbai now", "Is it raining in Kolkata right now?"])
    place = places[0]
    name, lat, lon = V.point_of(place)
    report = LC.current(name, lat, lon, place.state)
    if report.get("status") == "available":
        o = report["observation"]
        fields = (("temperature_max_C", "temperature", "°C", 1), ("rainfall_mm", "rainfall", "mm", 1), ("humidity_percent", "humidity", "%", 0),
                  ("wind_speed_kmh", "wind", "km/h", 0), ("cloud_cover_percent", "cloud cover", "%", 0), ("pressure_hpa", "pressure", "hPa", 0))
        bits = [f"{label} {o[k]:.{d}f} {u}" for k, label, u, d in fields if isinstance(o.get(k), (int, float))]
        lead = f"Right now in {o.get('city', name)}: " + "; ".join(bits) + (f". {o['description']}." if o.get("description") else ".")
        sections = []
        if isinstance(o.get("temperature_max_C"), (int, float)):
            sections.append(("Temperature", N.temp_meaning(o["temperature_max_C"], "max")))
        if isinstance(o.get("humidity_percent"), (int, float)):
            sections.append(("Humidity", N.humidity_meaning(o["humidity_percent"])))
        out24 = o.get("outlook_24h") or {}
        if out24:
            sections.append(("Next 24 hours", f"rain {f(out24.get('rainfall_total_mm'))} mm, max rain chance {f(out24.get('precipitation_probability_max_percent'), 0)}%, "
                                              f"peak gust {f(out24.get('wind_gust_max_kmh'), 0)} km/h"))
        tips = N.advice(s.advice or ["umbrella"], place.state, out24.get("rainfall_total_mm"), out24.get("precipitation_probability_max_percent"),
                        o.get("temperature_max_C"), o.get("wind_speed_kmh"), False)
        sections += [("Advice", t) for t in tips]
        sections += [("Coverage", f"provider value at {o.get('city', name)} ({lat:.2f}° N, {lon:.2f}° E), not a state average."),
                     ("Source", f"{report.get('source')} · updated {report.get('updated_at', '')[:16]}")]
        return Answer(lead, sections, sources=[{"file": report.get("source"), "method": "live"}], handler="current",
                      followups=[f"Will it rain in {place.name} tomorrow?", f"How does {place.name} compare with normal this season?"])
    # offline fallback: newest IMD day and the ECMWF snapshot
    sections = [("Live data", f"unavailable ({report.get('message', 'no connection')}). Showing the newest offline data instead.")]
    lead = f"I can't reach the live weather providers for {place.name} right now."
    st = place.state
    if st:
        cov = H.imd_coverage()
        if cov:
            row = H.imd_series(st, cov[1], cov[1])
            if not row.empty:
                r = row.iloc[0]
                sections.append(("Latest IMD day", f"{r['date']}: {f(r['daily_actual_mm'])} mm, {N.departure_meaning(r['daily_departure_pct'], r['daily_category'])}"))
        e = V.ecmwf_value(place, "temperature")
        if e:
            sections.append(("Latest ECMWF analysis", f"{e['date']}: temperature {f(e['value'])} °C"))
    return Answer(lead, sections, handler="current-offline",
                  followups=[f"Rainfall in {place.name} last week", "What data do you have?"])


def h_forecast(q: str, s: Slots) -> Answer:
    places = _places(s)
    if not places:
        return Answer("Which place should I forecast? Name a state or city (or select one on the map).", handler="clarify")
    ts = s.time if s.time.start else TimeSpec("date", today() + timedelta(days=1), today() + timedelta(days=1), "tomorrow", 24, True, "day")
    if ts.end < today():
        ts = TimeSpec("date", today() + timedelta(days=1), today() + timedelta(days=1), "tomorrow", 24, True, "day")
    elif ts.start < today():
        ts = TimeSpec("range", today(), ts.end, ts.text, ts.lead_hours, True, ts.granularity)
    variables = s.variables or ["rainfall", "temperature"]
    leads, sections, sources = [], [], []
    for place in places[:3]:
        hydra_rows = []
        if place.state:
            for d in pd.date_range(ts.start, ts.end):
                out = V.hydra_forecast(place.state, d.date())
                if out:
                    hydra_rows.append(out)
        for out in hydra_rows:
            fc = out["forecast"]
            parts = []
            for key, label in (("tp_mm", "rainfall"), ("t2m_C_mean", "mean temperature"), ("wind_speed_mean", "mean wind")):
                v = fc.get(key)
                if v:
                    dom = max(v["experts"], key=lambda e: e["weight"])["name"] if v.get("experts") else "n/a"
                    parts.append(f"{label} {f(v['value'])} {v['unit']} (80% range {f(v['interval80'][0])}-{f(v['interval80'][1])}; dominant expert {dom})")
            leads.append(f"HYDRA forecast for {place.state}, valid {out['valid_date']} (+{out['lead_hours']}h): " + "; ".join(parts) + ".")
            sources.append({"file": "runtime/hydra_daily_mean_state_blend.json", "method": "HYDRA published state cycle"})
        if not hydra_rows and "hydra" in s.sources and place.state and H.hydra_state(place.state):
            cyc = H.hydra_valid_dates()
            out = H.hydra_state(place.state)["outlooks"][0]
            tp = out["forecast"]["tp_mm"]
            sections.append(("HYDRA", f"the newest published HYDRA state cycle is valid {cyc[0]} to {cyc[-1]}, not {ts.start}; "
                                      f"its +24h rainfall for {place.state} was {f(tp['value'])} mm/day. A fresh HYDRA forecast needs the latest ERA5 days "
                                      "staged and scripts/train_hydra_daily_mean_state_blend.py re-run."))
        live = V.live_series(place, variables[0], ts) if not hydra_rows or "hydra" not in s.sources else None
        if live is not None:
            r = LC.daily([V.point_of(place)], variables, ts.start, ts.end)
            if r.get("status") == "available":
                agg = r["points"][0]["aggregate"]
                bits = [f"{v['label']} {f(v['value'])} {v['unit']}" for v in agg.values() if v.get("value") is not None]
                leads.append(f"Forecast for {place.name} {_period(ts)}: " + "; ".join(bits) + ".")
                rain = agg.get("precipitation_sum", {}).get("value")
                prob = agg.get("precipitation_probability_max", {}).get("value")
                tmax = agg.get("temperature_2m_max", {}).get("value")
                wind = agg.get("wind_speed_10m_max", {}).get("value")
                if rain is not None:
                    sections.append(("Rain outlook", N.rain_meaning(rain / max(ts.days, 1), False) + (f" Peak rain chance {f(prob, 0)}%." if prob is not None else "")))
                if tmax is not None:
                    sections.append(("Temperature outlook", N.temp_meaning(tmax, "max")))
                sections += [("Advice", t) for t in N.advice(s.advice or ["umbrella"], place.state, rain, prob, tmax, wind, False)]
                sections.append(("Coverage", f"Open-Meteo at {'the city' if place.kind == 'city' else 'a central point of the state'} "
                                             f"({r['points'][0]['latitude']:.2f}° N, {r['points'][0]['longitude']:.2f}° E)."))
                sources.append({"file": r["provider"], "method": "live forecast"})
            elif not hydra_rows:
                cycle = H.hydra_valid_dates()
                sections.append(("Live forecast", f"unavailable ({r.get('message')})."))
                if cycle and place.state and "hydra" not in s.sources:
                    out = H.hydra_state(place.state)["outlooks"][0]
                    tp = out["forecast"]["tp_mm"]
                    sections.append(("Latest HYDRA cycle", f"the newest published HYDRA cycle is valid {cycle[0]} to {cycle[-1]} "
                                                           f"(not your date): rainfall {f(tp['value'])} mm/day for {place.state} at +24h."))
                    sections.append(("Fresh HYDRA forecast", "needs the newest ERA5 days staged and the training script re-run; see README."))
                leads.append(f"I can't produce a forecast for {place.name} {_period(ts)} right now.")
    if not leads:
        return _unavailable(places[0].name, variables[0], ts, "No forecast source answered.")
    if re.search(r"\b(heavy|very heavy|alert|warning|risk|chance|probability|extreme)\b", q.lower()):
        from . import calibrated as CAL
        cov = CAL.coverage()
        if cov:
            sections.append(("Calibrated heavy-rain risk", f"HYDRA v3.1 calibrated probabilities and alert tiers exist for {cov[0]} to {cov[1]} "
                                                           "(replay). Future dates need the daily post-processing run; until then use the provider "
                                                           "rain chance above."))
    if any("HYDRA forecast" in l for l in leads):
        sections.append(("How to read HYDRA", "the 80% range is a calibrated interval; expert weights show how the neural gate blended its experts."))
    return Answer(" ".join(leads), sections, sources=sources, handler="forecast",
                  followups=[f"Should I carry an umbrella in {places[0].name}?", f"How accurate is HYDRA in {places[0].state or places[0].name}?",
                             f"Chance of heavy rain in {places[0].name} this week"])


# ================================================================== rankings
def _state_frame(var: str, ts: TimeSpec, prefer: list[str]) -> tuple[pd.DataFrame, str, str, bool]:
    """value per state for a period -> (frame[place, value, extra...], source, unit, is_observed)."""
    if var == "rainfall":
        win_imd = V._overlap(ts, H.imd_coverage())
        win_rep = V._overlap(ts, H.replay_coverage())
        if win_imd and not ("era5" in prefer and win_rep):
            df = H.imd_states_on(*win_imd)
            g = df.groupby("place").agg(value=("daily_actual_mm", "sum"), normal=("daily_normal_mm", "sum"), days=("date", "nunique")).reset_index()
            g["departure"] = (g["value"] - g["normal"]) / g["normal"].replace(0, np.nan) * 100
            return g, f"IMD observed state rainfall {win_imd[0]} to {win_imd[1]}", "mm", True
        if win_rep:
            df = H.replay()
            df = df[(df["date"] >= win_rep[0]) & (df["date"] <= win_rep[1])]
            g = df.groupby("place").agg(value=("actual_mm", "sum"), hydra=("hydra_mm", "sum"), days=("date", "nunique"),
                                        local_max=("actual_local_max_mm", "max")).reset_index()
            return g, f"ERA5 state-mean rainfall {win_rep[0]} to {win_rep[1]}", "mm", True
    if H.ecmwf_date() and ts.start == ts.end == H.ecmwf_date() and var in H.ECMWF_FIELDS:
        t = H.ecmwf_states()
        fld, unit = H.ECMWF_FIELDS[var]
        g = pd.DataFrame({"place": t["place"], "value": t[f"{fld}_mean"] * (100 if unit == "fraction" else 1), "max": t[f"{fld}_max"]})
        return g, f"ECMWF analysis {H.ecmwf_date()}", "%" if unit == "fraction" else unit, True
    if ts.start and all(V.hydra_forecast("Kerala", d.date()) for d in pd.date_range(ts.start, ts.end)) and var in ("rainfall", "temperature", "wind"):
        key = {"rainfall": "tp_mm", "temperature": "t2m_C_mean", "wind": "wind_speed_mean"}[var]
        rows = []
        for st in L.STATES:
            vals = [V.hydra_forecast(st, d.date())["forecast"][key]["value"] for d in pd.date_range(ts.start, ts.end) if V.hydra_forecast(st, d.date())]
            rows.append({"place": st, "value": float(np.sum(vals) if var == "rainfall" else np.mean(vals))})
        return pd.DataFrame(rows), "HYDRA published state forecast", V.UNITS[var] if var != "wind" else "m/s", False
    # live: one representative point per state
    points = [LC.state_point(st) for st in L.STATES]
    start, end = ts.start or today(), ts.end or today()
    r = LC.daily(points, [var], start, end)
    if r.get("status") != "available":
        return pd.DataFrame(), r.get("message", "live data unavailable"), "", False
    rows = []
    for p in r["points"]:
        agg = next(iter(p["aggregate"].values()))
        rows.append({"place": p["name"], "value": agg["value"]})
    unit = next(iter(r["points"][0]["aggregate"].values()))["unit"]
    return pd.DataFrame(rows), f"{r['provider']}, one central point per state, {start} to {end}", unit, end < today()


def h_rank_places(q: str, s: Slots) -> Answer:
    var = _var(s)
    ts = _default_window(var, s.time)
    g, source, unit, observed = _state_frame(var, ts, s.sources)
    if g.empty:
        return _unavailable("Indian states", var, ts, source)
    region = next((p.name for p in s.places if p.kind == "region"), None)
    if region and region in L.REGION_STATES:
        g = g[g["place"].isin(L.REGION_STATES[region])]
    by = "departure" if re.search(r"\b(departure|deficit|deficient|excess|normal)\b", q.lower()) and "departure" in g else "value"
    if by == "departure" and not s.time.start and H.imd_coverage():
        last = H.imd_states_on(H.imd_coverage()[1], H.imd_coverage()[1])
        g = pd.DataFrame({"place": last["place"], "value": last["cumulative_actual_mm"], "normal": last["cumulative_normal_mm"],
                          "departure": last["cumulative_departure_pct"], "days": 1})
        source = f"IMD season-cumulative rainfall as of {H.imd_coverage()[1]}"
        ts = TimeSpec("date", H.imd_coverage()[1], H.imd_coverage()[1], "season to date")
    asc = s.direction == "asc" or bool(re.search(r"\b(deficit|deficient|least|lowest|driest)\b", q.lower())) and by == "departure"
    g = g.dropna(subset=[by]).sort_values(by, ascending=asc)
    k = s.top_k or 5
    ranks = []
    for _, r in g.head(k).iterrows():
        text = f"{r['place']}: {f(r['value'])} {unit}"
        if "departure" in r and pd.notna(r.get("departure")):
            text += f" ({'+' if r['departure'] >= 0 else ''}{f(r['departure'], 0)}% vs normal)"
        if "local_max" in r and pd.notna(r.get("local_max")):
            text += f"; wettest cell {f(r['local_max'])} mm"
        ranks.append(text)
    top = g.iloc[0]
    word = "lowest" if asc else "highest"
    metric = "rainfall departure from normal" if by == "departure" else f"{var.replace('_', ' ')}"
    when = "for the season to date" if ts.text == "season to date" else _period(ts)
    lead = f"{top['place']} had the {word} {metric} {when}" + (f": {f(top[by])}{'%' if by == 'departure' else ' ' + unit}." if pd.notna(top[by]) else ".")
    sections = []
    if var == "rainfall" and by == "value":
        days = int(g["days"].max()) if "days" in g else 1
        sections.append(("What this means", N.rain_meaning(top["value"] / max(days, 1), True)))
        bottom = g.iloc[-1]
        sections.append(("Spread", f"{bottom['place']} was at the other end with {f(bottom['value'])} {unit}; {len(g)} states/UTs compared."))
    if by == "departure":
        n_def = int((g["departure"] <= -20).sum())
        n_exc = int((g["departure"] >= 20).sum())
        sections.append(("Overall", f"{n_exc} states/UTs in excess (≥ +20%) and {n_def} deficient (≤ −20%) over this period."))
    sections += [("Source", source), ("Coverage", "state values are area averages (IMD/ERA5/ECMWF) or one central point per state (live)."
                                                   if True else "")]
    return Answer(lead, sections, ranks, [{"file": source, "method": "ranking"}], handler="rank_places", data={"ranking": g.head(k).to_dict("records")},
                  followups=[f"Why was {top['place']} so {'dry' if asc else 'wet'}?", f"Was {top['place']} above normal?",
                             f"Compare {top['place']} and {g.iloc[1]['place'] if len(g) > 1 else 'Kerala'}"])


def h_rank_days(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    place = places[0]
    var = _var(s)
    ts = s.time if s.time.start else _whole_coverage(var, place)
    series = V.series(place, var, ts, s.sources)
    if not series.ok:
        return _unavailable(place.name, var, ts, series.message)
    df = pd.DataFrame({"date": series.dates, "value": series.values}).dropna()
    asc = s.direction == "asc"
    df = df.sort_values("value", ascending=asc)
    k = s.top_k or (1 if re.search(r"\b(the|which) (day|date)\b|\bwhen\b", q.lower()) and not re.search(r"\bdays\b", q.lower()) else 5)
    k = max(k, 3)
    ranks = [f"{pd.Timestamp(r['date']):%a %d %b %Y}: {f(r['value'])} {series.unit}" for _, r in df.head(k).iterrows()]
    top = df.iloc[0]
    word = {("rainfall", False): "wettest", ("rainfall", True): "driest", ("temperature", False): "hottest", ("temperature", True): "coldest",
            ("wind", False): "windiest", ("wind", True): "calmest", ("humidity", False): "most humid", ("humidity", True): "least humid"}.get((var, asc), "lowest" if asc else "highest")
    lead = f"The {word} day for {series.place} " \
           f"{_period(TimeSpec('range', series.dates[0], series.dates[-1]))} was {pd.Timestamp(top['date']):%d %b %Y} with {f(top['value'])} {series.unit}."
    sections = []
    if var == "rainfall":
        sections.append(("What this means", N.rain_meaning(top["value"], series.state_mean)))
        if series.source_key == "replay":
            rows = series.extra["rows"].set_index("date")
            d = top["date"]
            if d in rows.index:
                r = rows.loc[d]
                sections.append(("HYDRA that day", N.model_error_text(r["actual_mm"], r["hydra_mm"], r["lo80"], r["hi80"])
                                 + (f" Wettest grid cell: {f(r['actual_local_max_mm'])} mm." if pd.notna(r.get("actual_local_max_mm")) else "")))
        if series.source_key == "imd":
            r = series.extra["rows"].set_index("date").loc[top["date"]]
            sections.append(("Versus normal", N.departure_meaning(r["daily_departure_pct"], r["daily_category"])))
    sections += [("Coverage", series.note.strip() or f"{len(df)} days compared."), ("Source", series.source)]
    return Answer(lead, sections, ranks, [_src(series)], handler="rank_days", data={"focus_date": str(pd.Timestamp(top["date"]).date())},
                  followups=[f"Why was {pd.Timestamp(top['date']):%d %b} so wet in {series.place}?", f"Did HYDRA predict {pd.Timestamp(top['date']):%d %b %Y} in {series.place}?"])


def _whole_coverage(var: str, place: Place) -> TimeSpec:
    cov = H.imd_coverage() if var == "rainfall" else None
    if cov:
        return TimeSpec("range", cov[0], cov[1], "the full IMD record", granularity="custom")
    return TimeSpec("range", today() - timedelta(days=30), today() - timedelta(days=1), "the last 30 days", relative=True)


# ================================================================== comparison
def _two_periods(q: str) -> list[TimeSpec]:
    from .timeparse import parse
    parts = re.split(r"\s+(?:vs\.?|versus|compared (?:to|with)|against|or|and|with)\s+", q, flags=re.I)
    specs = []
    for part in parts:
        ts = parse(part)
        if ts.start and all((ts.start, ts.end) != (x.start, x.end) for x in specs):
            specs.append(ts)
    return specs[:3]


def h_compare(q: str, s: Slots) -> Answer:
    var = _var(s)
    places = [p for p in s.places if p.kind in ("state", "city", "region")]
    periods = _two_periods(q)
    entities = []
    if len(places) >= 2:
        ts = _default_window(var, s.time)
        entities = [(p.name, p, ts) for p in places[:5]]
    elif places and len(periods) >= 2:
        entities = [(f"{places[0].name} {ts.text}", places[0], ts) for ts in periods]
    elif len(places) == 1:
        return Answer(f"What should I compare {places[0].name} with? Name a second place or a second period.", handler="clarify",
                      followups=[f"Compare {places[0].name} and Kerala rainfall last week", f"Compare {places[0].name} rainfall in August vs September 2025"])
    else:
        return Answer("Name the two places (or two periods) to compare.", handler="clarify")
    results = []
    for label, place, ts in entities:
        srs = V.series(place, var, ts, s.sources)
        if srs.ok:
            how = s.statistic or ("total" if var == "rainfall" else "max" if var == "temperature" else "mean")
            results.append((label, srs.stat(how), srs, how))
    if len(results) < 2:
        return _unavailable(" and ".join(e[0] for e in entities), var, entities[0][2], "Not enough data for both sides of the comparison.")
    results.sort(key=lambda r: r[1] if r[1] is not None else -1e9, reverse=True)
    how = results[0][3]
    ranks = []
    for label, val, srs, _ in results:
        extra = ""
        if srs.source_key == "imd":
            rows = srs.extra["rows"]
            normal = rows["daily_normal_mm"].sum()
            dep = (rows["daily_actual_mm"].sum() - normal) / normal * 100 if normal else None
            extra = f" ({'+' if (dep or 0) >= 0 else ''}{f(dep, 0)}% vs normal)"
        ranks.append(f"{label}: {f(val)} {srs.unit} {how}{extra} [{srs.source_key.upper()}]")
    a, b = results[0], results[-1]
    lead = N.compare_text(a[0], a[1], b[0], b[1], a[2].unit, var) or f"{a[0]} {f(a[1])} vs {b[0]} {f(b[1])}."
    sections = []
    if var == "rainfall" and all(r[2].source_key == "imd" for r in results):
        sections.append(("Relative to normal", "departures account for each state's own climate, so a drier state can still be 'wetter than usual'."))
    if len({r[2].source_key for r in results}) > 1:
        sections.append(("Caution", "the two sides come from different sources; compare with care."))
    sections.append(("Source", "; ".join(sorted({r[2].source for r in results}))))
    return Answer(lead, sections, ranks, [_src(r[2]) for r in results], handler="compare",
                  followups=[f"Which day was wettest in {a[0]}?", f"Trend of rainfall in {b[0]}"])


# ================================================================== threshold days
def h_threshold_days(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    place = places[0]
    var = _var(s)
    ts = s.time if s.time.start else _whole_coverage(var, place)
    series = V.series(place, var, ts, s.sources)
    if not series.ok:
        return _unavailable(place.name, var, ts, series.message)
    t = q.lower()
    arr = series.arr()
    dates = series.dates
    label = ""
    if series.source_key == "imd" and re.search(r"\b(large excess|excess|deficient|deficit|normal|no rain)\b", t) and not s.threshold:
        cats = series.extra["category"]
        want = next(c for c in ("large_excess", "large_deficient", "excess", "deficient", "no_rain", "normal")
                    if c.replace("_", " ") in t or (c == "deficient" and "deficit" in t))
        mask = np.array([c == want or (want == "deficient" and c in ("deficient", "large_deficient")) or (want == "excess" and c in ("excess", "large_excess")) for c in cats])
        label = f"in the IMD '{want.replace('_', ' ')}' category"
    elif s.threshold:
        op, val = s.threshold["op"], s.threshold["value"]
        mask = {">": arr > val, ">=": arr >= val, "<": arr < val, "<=": arr <= val}[op]
        label = f"{s.threshold['text'].strip()}"
    elif re.search(r"\b(dry|no rain)\b", t):
        mask, label = arr < 1.0, "dry (< 1 mm)"
    elif re.search(r"\bheavy\b", t):
        thr = 64.5 if not series.state_mean else 20.0
        mask, label = arr >= thr, f"heavy (≥ {thr} mm{' state mean' if series.state_mean else ''})"
    else:
        mask, label = arr >= 2.5, "rainy (≥ 2.5 mm, IMD rainy-day definition)"
    hits = [d for d, m in zip(dates, mask) if m]
    n = len(arr)
    lead = f"{series.place} had {len(hits)} of {n} days {label} {_period(TimeSpec('range', dates[0], dates[-1]))}."
    ranks = [f"{pd.Timestamp(d):%a %d %b %Y}: {f(v)} {series.unit}" for d, v, m in zip(dates, arr, mask) if m][:15]
    sections = [("Share of days", f"{f(len(hits) / n * 100, 0)}%")]
    if len(hits) > 15:
        sections.append(("Note", f"showing the first 15 of {len(hits)} days."))
    if series.state_mean and s.threshold and s.threshold["value"] >= 50 and var == "rainfall":
        sections.append(("Scale", "a state average rarely exceeds 50 mm; local downpours of that size happen far more often. "
                                  "Ask about local extremes for grid-cell values."))
    sections += [("Coverage", series.note.strip() or f"{n} days checked."), ("Source", series.source)]
    return Answer(lead, sections, ranks, [_src(series)], handler="threshold_days",
                  followups=[f"Trend of rainfall in {series.place}", f"Was {series.place} above normal?"])


# ================================================================== trend and anomaly
def h_trend(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    place, var = places[0], _var(s)
    ts = s.time if s.time.start and s.time.days >= 7 else _whole_coverage(var, place)
    series = V.series(place, var, ts, s.sources)
    if not series.ok or len(series.values) < 6:
        return _unavailable(place.name, var, ts, series.message or "A trend needs at least a week of daily values.")
    tr = N.trend(series.dates, series.values)
    df = pd.DataFrame({"date": pd.to_datetime(series.dates), "v": series.arr()})
    weekly = df.set_index("date")["v"].resample("W").agg(["sum", "mean", "count"])
    how = "sum" if var == "rainfall" else "mean"
    ranks = [f"week ending {d:%d %b %Y}: {f(r[how])} {series.unit}{' total' if how == 'sum' else ' mean'} ({int(r['count'])} days)"
             for d, r in weekly.iterrows() if r["count"] > 0][-12:]
    lead = (f"{series.place} {var}: {tr['strength']} {tr['direction']} trend {_period(TimeSpec('range', series.dates[0], series.dates[-1]))} "
            f"({'+' if tr['slope_per_day'] >= 0 else ''}{f(tr['slope_per_day'], 2)} {series.unit}/day per day).")
    sections = [("First half vs second half", f"{f(tr['first_half'])} → {f(tr['second_half'])} {series.unit}/day"
                 + (f" ({'+' if (tr['change_pct'] or 0) >= 0 else ''}{f(tr['change_pct'], 0)}%)" if tr["change_pct"] is not None else "")),
                ("What this means", "daily weather is noisy, so a 'weak' trend means the change is small relative to day-to-day swings."
                 if tr["strength"] != "clear" else ("rainfall is clearly building through the period." if tr["slope_per_day"] > 0 and var == "rainfall"
                                                     else "rainfall is clearly easing off through the period." if var == "rainfall" else "the change is consistent through the period."))]
    if series.source_key == "imd":
        rows = series.extra["rows"]
        cat = rows['cumulative_category'].iloc[-1]
        sections.append(("Season context", f"latest cumulative departure {f(rows['cumulative_departure_pct'].iloc[-1], 0)}%"
                                           + (f" ({cat.replace('_', ' ')})." if isinstance(cat, str) else ".")))
    sections += [("Coverage", series.note.strip() or f"{len(series.values)} days."), ("Source", series.source)]
    return Answer(lead, sections, ranks, [_src(series)], handler="trend",
                  followups=[f"Was {series.place} above normal?", f"Wettest day in {series.place}"])


def h_anomaly(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    place = places[0]
    ts = s.time if s.time.start else None
    st = "COUNTRY : INDIA" if place.kind == "india" else place.name if place.kind == "region" else place.state
    cov = H.imd_coverage()
    season_default = ts is None or ts.granularity == "season" and ts.end and ts.end >= cov[1] if cov else False
    if cov and (ts is None or V._overlap(ts, cov)):
        win = V._overlap(ts, cov) if ts else (cov[1], cov[1])
        if season_default:
            win = (cov[1], cov[1])
        df = H.imd_series(st, *win)
        if not df.empty:
            label = H.IMD_REGIONS.get(st, st)
            last = df.iloc[-1]
            actual, normal = df["daily_actual_mm"].sum(), df["daily_normal_mm"].sum()
            dep = (actual - normal) / normal * 100 if normal else None
            if season_default:
                lead = (f"{label}, season to date (as of {last['date']:%d %b %Y}): {f(last['cumulative_actual_mm'], 0)} mm against a normal of "
                        f"{f(last['cumulative_normal_mm'], 0)} mm, {N.departure_meaning(last['cumulative_departure_pct'], last['cumulative_category'])}")
                dep = last["cumulative_departure_pct"]
            elif win[0] == win[1]:
                lead = f"{label} on {win[0]:%d %b %Y}: {f(actual)} mm against a normal of {f(normal)} mm, {N.departure_meaning(dep, last['daily_category'])}"
            else:
                lead = f"{label} {_period(TimeSpec('range', *win))}: {f(actual)} mm against a normal of {f(normal)} mm, {N.departure_meaning(dep, _category(dep))}"
            sections = [("That day (IMD)", f"{last['date']:%d %b}: {f(last['daily_actual_mm'])} mm vs {f(last['daily_normal_mm'])} mm normal, {N.departure_meaning(last['daily_departure_pct'], last['daily_category'])}"),
                        ("This week (IMD)", f"{f(last['weekly_actual_mm'])} mm vs {f(last['weekly_normal_mm'])} mm normal, {N.departure_meaning(last['weekly_departure_pct'], last['weekly_category'])}"),
                        ("This month (IMD)", f"{f(last['monthly_actual_mm'])} mm vs {f(last['monthly_normal_mm'])} mm, {N.departure_meaning(last['monthly_departure_pct'], last['monthly_category'])}"),
                        ("Season to date", f"{f(last['cumulative_actual_mm'], 0)} mm vs {f(last['cumulative_normal_mm'], 0)} mm, {N.departure_meaning(last['cumulative_departure_pct'], last['cumulative_category'])}"),
                        ("How IMD classifies", "large excess ≥ +60%, excess +20 to +59%, normal −19 to +19%, deficient −20 to −59%, large deficient −60 to −99%."),
                        ("Source", f"IMD state-wise rainfall (as of {last['date']})")]
            if place.kind == "city":
                sections.insert(0, ("Scale", f"IMD departures are state values; {place.name} is in {place.state}."))
            return Answer(lead, sections, sources=[{"file": "rainfall_statewise_daily_imd_clean.csv", "method": "IMD departures"}], handler="anomaly",
                          followups=[f"Which states are most deficient this season?", f"Rainfall trend in {label}", f"Why was {label} {'wet' if (dep or 0) > 0 else 'dry'}?"])
    # outside IMD: Open-Meteo archive, same window in the previous 10 years as the normal
    ts = ts or TimeSpec("range", today() - timedelta(days=30), today() - timedelta(days=1), "last 30 days")
    if place.kind in ("india", "region"):
        return _unavailable(place.name, "rainfall", ts, "Regional normals outside the IMD window need the live connection per state; ask for a state.")
    point = V.point_of(place)
    years = 10
    r = LC.daily([point], ["rainfall"], date(ts.start.year - years, ts.start.month, min(ts.start.day, 28)), ts.end)
    if r.get("status") != "available":
        return _unavailable(place.name, "rainfall", ts, f"IMD normals cover {cov[0]} to {cov[1]} only, and the live archive is unreachable ({r.get('message')}).")
    p = r["points"][0]
    dfd = pd.DataFrame({"date": pd.to_datetime(p["dates"]), "v": p["series"]["precipitation_sum"]}).dropna()
    totals = []
    for k in range(years + 1):
        a = pd.Timestamp(ts.start.replace(year=ts.start.year - k))
        b = pd.Timestamp(ts.end.replace(year=ts.end.year - k))
        totals.append(dfd[(dfd["date"] >= a) & (dfd["date"] <= b)]["v"].sum())
    current, normal = totals[0], float(np.mean(totals[1:]))
    dep = (current - normal) / normal * 100 if normal else None
    lead = f"{place.name} {_period(ts)}: {f(current)} mm against a {years}-year average of {f(normal)} mm for the same dates, {N.departure_meaning(dep, _category(dep))}"
    return Answer(lead, [("Method", f"Open-Meteo archive at {point[1]:.2f}° N, {point[2]:.2f}° E; 'normal' = mean of the same calendar window in the previous {years} years (not the IMD 1971-2020 normal)."),
                         ("Source", r["provider"])], sources=[{"file": r["provider"], "method": "archive anomaly"}], handler="anomaly")


# ================================================================== HYDRA model questions
EXPERT_TEXT = {
    "climatology": "average rainfall for that time of year at that place (train-period only)",
    "persistence": "tomorrow looks like today",
    "recent3": "mean of the last three days",
    "anom_persistence": "climatology plus today's anomaly (today's departure carries forward)",
    "gbm_era5": "gradient-boosted trees on every ERA5 atmospheric predictor plus rainfall history",
    "hurdle": "P(rain ≥ 1 mm) from a classifier × wet-day amount from a gamma regressor",
    "spatial": "uses only rain, CAPE and moisture in the surrounding ±0.5° and ±1° neighbourhood",
    "monsoon": "a specialist trained only on monsoon-season rows",
    "upper_q": "an 85th-percentile model that leans high, so the gate can reach for peaks",
}


def h_model_how(q: str, s: Slots) -> Answer:
    meta = H.replay_meta()
    t = q.lower()
    experts = meta.get("experts") or list(EXPERT_TEXT)
    version = meta.get("model_version") or "hydra-rain-v2"
    for name, text in EXPERT_TEXT.items():
        if name.replace("_", " ") in t or name in t or (name == "upper_q" and "quantile" in t) or (name == "spatial" and "neighbour" in t):
            return Answer(f"The '{name}' expert: {text}.", [("Role", "it is one of the inputs the neural gate weighs for every cell and day; ask 'which expert dominates in <state>' to see its weight.")],
                          handler="model_how")
    lead = (f"HYDRA ({version}) forecasts rainfall for every 0.25° ERA5 grid cell, then averages the cells inside each state. "
            f"A neural gate looks at the atmospheric state and blends {len(experts)} experts into one forecast.")
    sections = [("Experts", "; ".join(f"{e}: {EXPERT_TEXT.get(e, '')}" for e in experts)),
                ("Inputs", "ERA5 temperature, dew point, humidity, CAPE, pressure and its tendency, wind components, cloud, radiation, "
                           "boundary-layer height, water vapour and gusts when available, plus rainfall history and neighbourhood rain."),
                ("Gate outputs", "expert weights, P(rain ≥ 1 mm), heavy-rain probabilities (≥ 20 mm, ≥ 64.5 mm, ≥ state p95), lower/upper interval bounds and amount-if-wet."),
                ("Training", f"truth up to {meta.get('training_cutoff', 'the cutoff')} only; extreme days get extra loss weight so peaks are not ignored."),
                ("Interval", (meta.get("calibration") or "conformal 80% interval")),
                ("Lead", f"+{meta.get('lead_hours', 24)} hours; each replay value used only data available on its issue day.")]
    if re.search(r"\bv2\b|\bv3\b|changed|difference", t):
        sections.insert(0, ("v2 vs v3", "v2 used four rainfall-only experts on state averages with a nearly static gate and a p90 band labelled 80%. "
                                        "v3 trains at grid level on ERA5 atmospheric predictors, adds five trained experts and heavy-rain heads, "
                                        "and calibrates a true 80% interval."))
    return Answer(lead, sections, handler="model_how", sources=[{"file": "runtime/hydra_rolling_rainfall_replay.json", "method": "model metadata"}],
                  followups=["Which expert dominates in Maharashtra?", "How accurate is HYDRA?", "What does the 80% interval mean?"])


def _replay_for(s: Slots, min_days: int = 1) -> tuple[pd.DataFrame, str]:
    df = H.replay()
    places = [p for p in s.places if p.state]
    label = "all states"
    if places:
        df = df[df["place"] == places[0].state]
        label = places[0].state
    if s.time.start and V._overlap(s.time, H.replay_coverage()) and s.time.days >= min_days:
        a, b = V._overlap(s.time, H.replay_coverage())
        df = df[(df["date"] >= a) & (df["date"] <= b)]
        label += f", {a} to {b}"
    return df, label


def h_model_weights(q: str, s: Slots) -> Answer:
    df, label = _replay_for(s)
    if df.empty:
        return _unavailable(label, "rainfall", s.time, f"the HYDRA replay covers {H.replay_coverage()}.")
    experts = H.replay_experts()
    w = df[[f"w_{e}" for e in experts]]
    mean = w.mean().sort_values(ascending=False)
    ranks = [f"{c[2:]}: {f(v * 100)}% average weight (std {f(w[c].std() * 100)} pts)" for c, v in mean.items()]
    dom = mean.index[0][2:]
    lead = f"In {label}, HYDRA's gate leaned most on the '{dom}' expert ({f(mean.iloc[0] * 100)}% average weight)."
    sections = []
    if len(df) == 1:
        r = df.iloc[0]
        lead = f"On {r['date']} in {r['place']}, the dominant expert was '{dom}' ({f(mean.iloc[0] * 100)}%); forecast {f(r['hydra_mm'])} mm vs {f(r['actual_mm'])} mm observed."
        ranks = [f"{e}: weight {f(r[f'w_{e}'] * 100)}%, value {f(r[f'exp_{e}'])} mm" for e in sorted(experts, key=lambda e: -r[f"w_{e}"])]
    else:
        wet = df["actual_mm"] >= 5
        if wet.any() and (~wet).any():
            shift = (df[wet][[f"w_{e}" for e in experts]].mean() - df[~wet][[f"w_{e}" for e in experts]].mean()).sort_values()
            sections.append(("Wet vs dry days", f"on wet days the gate shifts most towards '{shift.index[-1][2:]}' (+{f(shift.iloc[-1] * 100)} pts) "
                                                f"and away from '{shift.index[0][2:]}' ({f(shift.iloc[0] * 100)} pts)."))
        std = float(w.std().mean())
        sections.append(("Gate dynamics", f"mean weight std {f(std * 100, 2)} pts across days: "
                                          + ("the gate responds to conditions." if std >= 0.02 else "nearly static, so the blend barely adapts.")))
    sections += [("Meaning", "weights are blend allocations, not confidence and not causal attribution."),
                 ("Source", "HYDRA rolling replay (runtime/hydra_rolling_rainfall_replay.json)")]
    return Answer(lead, sections, ranks, [{"file": "runtime/hydra_rolling_rainfall_replay.json", "method": "gate weights"}], handler="model_weights",
                  followups=[f"How accurate is HYDRA in {label.split(',')[0]}?", f"What is the {dom} expert?"])


def _metrics(df: pd.DataFrame) -> dict:
    a, p = df["actual_mm"].to_numpy(float), df["hydra_mm"].to_numpy(float)
    e = p - a
    pers = df["exp_persistence"].to_numpy(float) if "exp_persistence" in df else None
    clim = df["exp_climatology"].to_numpy(float) if "exp_climatology" in df else None
    out = {"n": len(df), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
           "coverage": float(np.mean((a >= df["lo80"]) & (a <= df["hi80"]))), "width": float(np.mean(df["hi80"] - df["lo80"]))}
    if pers is not None:
        out["skill_persistence"] = 1 - out["mae"] / np.mean(np.abs(pers - a))
    if clim is not None:
        out["skill_climatology"] = 1 - out["mae"] / np.mean(np.abs(clim - a))
    heavy = a >= 20
    out["heavy_days"] = int(heavy.sum())
    out["heavy_recall"] = float(np.mean(p[heavy] >= 20)) if heavy.any() else None
    fc = p >= 20
    out["heavy_precision"] = float(np.mean(a[fc] >= 20)) if fc.any() else None
    i = int(np.argmax(a))
    out["peak"] = (str(df["date"].iloc[i]), float(a[i]), float(p[i]))
    return out


def h_model_accuracy(q: str, s: Slots) -> Answer:
    df, label = _replay_for(s, min_days=7)
    if df.empty:
        return _unavailable(label, "rainfall", s.time, f"the HYDRA replay covers {H.replay_coverage()}.")
    t = q.lower()
    if re.search(r"\bmonsoon\b", t) and not s.time.start:
        df = df[pd.to_datetime(df["date"]).dt.month.isin([6, 7, 8, 9])]
        label += ", monsoon months"
    m = _metrics(df)
    lead = (f"HYDRA's +24h rainfall in {label}: MAE {f(m['mae'], 2)} mm/day, RMSE {f(m['rmse'], 2)}, bias {'+' if m['bias'] >= 0 else ''}{f(m['bias'], 2)} mm/day "
            f"over {m['n']} state-days.")
    sk_p, sk_c = m.get("skill_persistence"), m.get("skill_climatology")
    sections = [("Versus baselines", f"MAE skill vs persistence {'+' if (sk_p or 0) >= 0 else ''}{f((sk_p or 0) * 100, 0)}%, vs climatology {'+' if (sk_c or 0) >= 0 else ''}{f((sk_c or 0) * 100, 0)}% "
                                     "(positive = HYDRA better)."),
                ("Heavy rain (state mean ≥ 20 mm)", f"{m['heavy_days']} observed days; " + (f"recall {f(m['heavy_recall'] * 100, 0)}%, " if m['heavy_recall'] is not None else "no heavy days to score, ") +
                                                     (f"precision {f(m['heavy_precision'] * 100, 0)}%." if m['heavy_precision'] is not None else "HYDRA never forecast a state mean of 20 mm or more.")),
                ("Peak day", f"{m['peak'][0]}: observed {f(m['peak'][1])} mm, HYDRA {f(m['peak'][2])} mm ({f(m['peak'][2] / m['peak'][1] * 100 if m['peak'][1] else 0, 0)}% of the peak)."),
                ("80% interval", f"covered {f(m['coverage'] * 100, 1)}% of observations (target 80%), mean width {f(m['width'])} mm.")]
    verdict = []
    if sk_p is not None:
        verdict.append("beats persistence" if sk_p > 0 else "does not beat persistence on MAE")
    if m["heavy_recall"] is not None:
        verdict.append("catches most heavy state-days" if m["heavy_recall"] >= 0.6 else "still misses many heavy state-days")
    if m["bias"] < -0.3:
        verdict.append("tends to under-predict")
    elif m["bias"] > 0.3:
        verdict.append("tends to over-predict")
    sections.insert(0, ("Verdict", ", ".join(verdict) + "." if verdict else "mixed."))
    ranks = []
    if not [p for p in s.places if p.state]:
        per = H.replay()
        rows = []
        for st, g in per.groupby("place"):
            mm = _metrics(g)
            rows.append((st, mm.get("skill_persistence", 0), mm["mae"]))
        rows.sort(key=lambda r: r[1], reverse=True)
        best, worst = rows[:3], rows[-3:]
        if re.search(r"\b(worst|best|rank|which state|where)\b", t):
            order = rows if not re.search(r"\bworst|weakest\b", t) else rows[::-1]
            ranks = [f"{r[0]}: MAE {f(r[2], 2)} mm/day, skill vs persistence {'+' if r[1] >= 0 else ''}{f(r[1] * 100, 0)}%" for r in order[:s.top_k or 10]]
        sections.append(("Best states (skill vs persistence)", ", ".join(f"{r[0]} {f(r[1] * 100, 0)}%" for r in best)))
        sections.append(("Weakest states", ", ".join(f"{r[0]} {f(r[1] * 100, 0)}%" for r in worst)))
    cyc = H.hydra_cycle()
    if s.variables and s.variables[0] in ("temperature", "wind") and cyc:
        key = {"temperature": "t2m_C_mean", "wind": "wind_speed_mean"}[s.variables[0]]
        g = cyc.get("training_metrics", {}).get(key, {}).get("1", {}).get("global", {})
        if g:
            sections.append((f"{s.variables[0].title()} (+24h cycle backtest)", f"MAE {f(g.get('mae'), 2)}, RMSE {f(g.get('rmse'), 2)} over {g.get('samples')} samples."))
    val = H.validation()
    if val.get("state_scale"):
        pooled = val["state_scale"]["pooled"]["continuous"]
        sections.append(("Rolling validation", f"multi-origin MAE {f(pooled.get('mae'), 2)}, RMSE {f(pooled.get('rmse'), 2)} ({val['design']['folds']} folds)."))
    try:
        from . import calibrated as CAL
        if CAL.available():
            g = CAL.gate()
            target = next((p.state for p in s.places if p.state), None)
            card = (CAL.state_summary(target) or {}).get("reliability") if target else None
            if card:
                sections.append(("HYDRA v3.1 (post-processed)", f"{target}: MAE {card['mae_mm']:.2f} vs {card['raw_mae_mm']:.2f} mm/day raw, "
                                 f"skill vs persistence {card['mae_skill_vs_persistence'] * 100:+.0f}%, reliability {card['reliability']}, "
                                 f"{card['heavy_20mm_events']} heavy days in the test period ({card['heavy_rain_evidence']})."))
            checks = {c["check"]: c["detail"] for c in g.get("checks", [])}
            if checks:
                sections.append(("v3.1 release gate", f"{g['status']}; MAE {checks.get('MAE not worse than raw v3', 'n/a')}, "
                                 f"80% range coverage {checks.get('80% interval coverage within 75-85%', 'n/a')}, "
                                 f"P(≥20 mm) Brier skill {checks.get('P(>=20 mm) calibrated (Brier skill > 0)', 'n/a')}"
                                 + (f"; failing: {', '.join(g['failed'])}" if g.get("failed") else "") + "."))
    except Exception:
        pass
    sections.append(("Source", "HYDRA rolling replay vs ERA5 state-mean rainfall (runtime/hydra_rolling_rainfall_replay.json)"))
    return Answer(lead, sections, ranks, [{"file": "runtime/hydra_rolling_rainfall_replay.json", "method": "replay verification"}], handler="model_accuracy",
                  followups=["Which state does HYDRA predict best?", "Did HYDRA catch the biggest rain day in Maharashtra?", "What does the 80% interval mean?"])


def h_model_vs_actual(q: str, s: Slots) -> Answer:
    df, label = _replay_for(s)
    if df.empty:
        return _unavailable(label, "rainfall", s.time, f"the HYDRA replay covers {H.replay_coverage()}.")
    t = q.lower()
    if re.search(r"\b(miss|misses|worst|wrong|biggest error)\b", t):
        df = df.assign(err=(df["hydra_mm"] - df["actual_mm"]).abs()).sort_values("err", ascending=False)
        ranks = [f"{r['date']} {r['place']}: observed {f(r['actual_mm'])} mm, HYDRA {f(r['hydra_mm'])} mm "
                 f"({'under' if r['hydra_mm'] < r['actual_mm'] else 'over'} by {f(r['err'])})" for _, r in df.head(s.top_k or 5).iterrows()]
        under = int((df.head(20)["hydra_mm"] < df.head(20)["actual_mm"]).sum())
        return Answer(f"HYDRA's largest errors in {label}:", [("Pattern", f"{under} of the 20 largest errors are under-predictions: big misses are mostly peaks the model softened."),
                                                               ("Source", "HYDRA rolling replay")], ranks, handler="model_vs_actual")
    if re.search(r"\b(peak|wettest|heaviest|biggest)\b", t):
        df = df.sort_values("actual_mm", ascending=False).head(1)
    if len(df) == 1:
        r = df.iloc[0]
        dom = max(H.replay_experts(), key=lambda e: r[f"w_{e}"])
        lead = (f"{r['place']} on {r['date']:%d %b %Y}: observed {f(r['actual_mm'])} mm (ERA5 state mean); HYDRA forecast {f(r['hydra_mm'])} mm "
                f"with an 80% interval of {f(r['lo80'])}-{f(r['hi80'])} mm, issued {r['issue_date']}.")
        sections = [("Verdict", N.model_error_text(r["actual_mm"], r["hydra_mm"], r["lo80"], r["hi80"])),
                    ("Dominant expert", f"{dom} ({f(r[f'w_{dom}'] * 100)}% weight, value {f(r[f'exp_{dom}'])} mm)")]
        if pd.notna(r.get("actual_local_max_mm")):
            sections.append(("Local extreme", f"wettest cell observed {f(r['actual_local_max_mm'])} mm vs HYDRA's wettest cell {f(r['local_peak_mm'])} mm."))
        if pd.notna(r.get("p_rain")):
            sections.append(("Rain chance", f"HYDRA expected {f(r['p_rain'] * 100, 0)}% of the state's cells to be wet."))
        try:
            from . import calibrated as CAL
            row = CAL.state_day(r["place"], r["date"]) if CAL.available() else None
        except Exception:
            row = None
        if row:
            sections.append(("HYDRA v3.1", CAL.risk_sentence(row) + f" ({CAL.label(row)})"))
        sections.append(("Source", "HYDRA rolling replay"))
        return Answer(lead, sections, handler="model_vs_actual", followups=[f"Why did HYDRA {'under' if r['hydra_mm'] < r['actual_mm'] else 'over'}-predict in {r['place']}?",
                                                                            f"How accurate is HYDRA in {r['place']}?"])
    m = _metrics(df)
    ranks = [f"{r['date']}: observed {f(r['actual_mm'])}, HYDRA {f(r['hydra_mm'])} mm" for _, r in df.sort_values("date").head(14).iterrows()]
    lead = f"HYDRA vs observed in {label}: totals {f(df['hydra_mm'].sum())} mm forecast vs {f(df['actual_mm'].sum())} mm observed; MAE {f(m['mae'], 2)} mm/day."
    worst = df.assign(err=df["hydra_mm"] - df["actual_mm"]).reindex((df["hydra_mm"] - df["actual_mm"]).abs().sort_values(ascending=False).index).head(3)
    sections = [("Verdict", "HYDRA under-forecast the period's total." if df["hydra_mm"].sum() < 0.9 * df["actual_mm"].sum() else
                 "HYDRA over-forecast the period's total." if df["hydra_mm"].sum() > 1.1 * df["actual_mm"].sum() else "HYDRA got the period's total about right."),
                ("Largest misses", "; ".join(f"{r['date']} ({'under' if r['err'] < 0 else 'over'} by {f(abs(r['err']))} mm)" for _, r in worst.iterrows())),
                ("Interval", f"{f(m['coverage'] * 100, 0)}% of days inside the 80% interval (target 80%)."), ("Source", "HYDRA rolling replay")]
    if len(df) > 14:
        sections.insert(2, ("Listing", f"first 14 of {len(df)} days shown."))
    return Answer(lead, sections, ranks, handler="model_vs_actual")


def h_uncertainty(q: str, s: Slots) -> Answer:
    meta = H.replay_meta()
    df, label = _replay_for(s)
    lead = ("HYDRA's 80% interval is a range that should contain the observed rainfall on about 8 of every 10 days. "
            "It is built by split-conformal calibration on days the model never trained on.")
    sections = [("Method", meta.get("calibration") or "conformal 80% interval"),
                ("Expert spread", "the standard deviation of the experts' forecasts: a disagreement signal, not a probability."),
                ("Weights are not confidence", "a 70% weight means the gate leaned on that expert, not that the forecast is 70% likely.")]
    if not df.empty:
        m = _metrics(df)
        sections.insert(1, ("Check", f"in {label} the interval covered {f(m['coverage'] * 100, 1)}% of observed days (target 80%), mean width {f(m['width'])} mm."))
        if s.time.start and len(df) == 1:
            r = df.iloc[0]
            sections.insert(0, ("That day", f"{r['date']}: interval {f(r['lo80'])}-{f(r['hi80'])} mm, expert spread {f(r['spread_mm'])} mm, observed {f(r['actual_mm'])} mm."))
    sections.append(("Wide intervals", "intervals widen in wet regimes and in states with volatile rain, because the calibration is stratified by state, season, lead and regime."))
    return Answer(lead, sections, handler="uncertainty", followups=["How accurate is HYDRA?", "Why do experts disagree?"])


# ================================================================== extremes, advice, why
def _calibrated_extremes(q: str, s: Slots, place: Place) -> Answer | None:
    """HYDRA v3.1 calibrated heavy-rain risk for replay dates (out-of-sample when available)."""
    from . import calibrated as CAL
    cov = CAL.coverage()
    if not cov or not s.time.start or s.time.start > cov[1] or s.time.end < cov[0]:
        return None
    day = max(s.time.start, cov[0]) if s.time.start == s.time.end else None
    if place.kind == "india" or not place.state:
        if day is None:
            return None
        res = CAL.alerts(day, "local_64.5" if re.search(r"64\.5|very heavy|extreme|cloudburst|district|anywhere|local", q.lower()) else "state_20")
        if res.get("status") != "available":
            return None
        issued = res["issued"]
        ranks = [f"{x['state']}: {CAL.pct(x['probability'])} ({x['tier'].upper()})" + ("" if x["observed"] is None else
                 f", {'happened' if x['observed'] else 'did not happen'} ({x['observed_mm']:.0f} mm)") for x in issued[:12]]
        hit = [x for x in issued if x["observed"]]
        missed = [x["state"] for x in res["states"] if x["observed"] and not x["tier"]]
        lead = (f"On {day:%d %b %Y}, HYDRA v3.1 issued a heavy-rain tier for {len(issued)} state(s) for '{res['event_label']}'."
                if issued else f"On {day:%d %b %Y}, HYDRA v3.1 issued no heavy-rain tier for '{res['event_label']}'.")
        sections = [("Outcome", f"{len(hit)} of {len(issued)} flagged states saw the event" + (f"; missed (no tier): {', '.join(missed[:8])}" if missed else "; no unflagged state saw it") + "."),
                    ("Tiers", "watch ≥ 20%, alert ≥ 40%, warning ≥ 60% calibrated probability."),
                    ("Status", CAL.label({"release_gate": res["release_gate"], "provenance": "out_of_sample" if "oos" in res["source"] else "fitted"})),
                    ("Source", f"HYDRA v3.1 post-processing ({res['source']})")]
        return Answer(lead, sections, ranks, [{"file": res["source"], "method": "calibrated heavy-rain risk"}], handler="extremes-calibrated")
    if day is None:
        summary = CAL.state_summary(place.state)
        if not summary:
            return None
        rows = [r for r in summary["rows"] if s.time.start.isoformat() <= r["valid_date"] <= s.time.end.isoformat()]
        flagged = [r for r in rows if r["alert"]["state_20"] or r["alert"]["local_64.5"]]
        if not rows:
            return None
        ranks = [f"{r['valid_date']}: state ≥20 {CAL.pct(r['probabilities']['state_20'])} ({(r['alert']['state_20'] or '-').upper()}), "
                 f"≥64.5 somewhere {CAL.pct(r['probabilities']['local_64.5'])} ({(r['alert']['local_64.5'] or '-').upper()}); observed {r['observed_mm']:.1f} mm"
                 for r in sorted(flagged, key=lambda r: -max(r['probabilities']['state_20'] or 0, r['probabilities']['local_64.5'] or 0))[:10]]
        lead = f"HYDRA v3.1 flagged heavy-rain risk on {len(flagged)} of {len(rows)} days in {place.state} {_period(s.time)}."
        card = summary.get("reliability") or {}
        sections = [("Reliability", f"{card.get('reliability', 'n/a')} ({card.get('heavy_20mm_events', 'n/a')} heavy days in the test period; {card.get('heavy_rain_evidence', '')})"),
                    ("Status", CAL.label(summary)), ("Source", "HYDRA v3.1 post-processing")]
        return Answer(lead, sections, ranks, handler="extremes-calibrated")
    row = CAL.state_day(place.state, day)
    if not row:
        return None
    lead = f"HYDRA v3.1 heavy-rain risk for {place.state} on {day:%d %b %Y}: " + CAL.risk_sentence(row)
    sections = [("Outcome", CAL.outcome_sentence(row)), ("HYDRA v3 (raw)", f"{row['hydra_v3_mm']:.1f} mm")]
    card = row.get("reliability") or {}
    if card:
        sections.append(("Reliability", f"{place.state} is rated {card['reliability']} ({card['heavy_20mm_events']} heavy days in the test period)."))
    sections += [("Status", CAL.label(row)), ("Source", "HYDRA v3.1 post-processing")]
    return Answer(lead, sections, handler="extremes-calibrated")


def h_extremes(q: str, s: Slots) -> Answer:
    places = _places(s) or [Place("India", "india")]
    place = places[0]
    t = q.lower()
    calibrated = None
    if not (s.time.start and s.time.start > today()) and not re.search(r"\bheat ?wave|extreme heat\b", t):
        calibrated = _calibrated_extremes(q, s, place)
        asks_risk = re.search(r"\b(alerts?|warnings?|watch|tiers?|risk|chance|probabilit|forecast|predict|flag)", t)
        if calibrated and (asks_risk or place.kind == "india" or not place.state):
            return calibrated
    observed = _observed_extremes(q, s, place, places, t)
    if calibrated and observed.handler != "unavailable":
        observed.sections.insert(len(observed.sections) - 1, ("HYDRA v3.1 risk for this period", calibrated.lead))
    return observed


def _observed_extremes(q: str, s: Slots, place: Place, places: list, t: str) -> Answer:
    if re.search(r"\b(will|coming|expected|upcoming|hogi)\b", t) or s.time.start and s.time.start > today() or re.search(r"\b(probability|chance|likely|risk|alert|sambhavna)\b", t) and (not s.time.start or s.time.start >= today()):
        return h_forecast(q, s)
    if re.search(r"\bheat ?wave|extreme heat\b", t):
        s.variables = ["temperature"]
        ts = s.time if s.time.start else TimeSpec("range", today() - timedelta(days=30), today() - timedelta(days=1), "last 30 days")
        srs = V.series(place, "temperature", ts, s.sources) if place.kind != "india" else None
        if not srs or not srs.ok:
            return _unavailable(place.name, "temperature", ts, srs.message if srs else "Name a state or city.")
        tmax = srs.extra.get("series", {}).get("temperature_2m_max") or srs.values
        days = [d for d, v in zip(srs.dates, tmax) if v is not None and v >= 40]
        return Answer(f"{place.name} had {len(days)} day(s) with maximum temperature ≥ 40 °C {_period(ts)}.",
                      [("Hottest", f"{f(max(v for v in tmax if v is not None))} °C"), ("Definition", "IMD plains heatwave threshold: maximum ≥ 40 °C (plus departure criteria)."),
                       ("Coverage", srs.note), ("Source", srs.source)], [str(d) for d in days[:15]], handler="extremes")
    ts = s.time if s.time.start else (TimeSpec("range", *H.replay_coverage(), "the replay period") if H.replay_coverage() else _whole_coverage("rainfall", place))
    if place.state and V._overlap(ts, H.replay_coverage()):
        df = H.replay_series(place.state, *V._overlap(ts, H.replay_coverage()))
        loc = df["actual_local_max_mm"]
        heavy, vheavy = df[loc >= 64.5], df[loc >= 115.6]
        top = df.sort_values("actual_local_max_mm", ascending=False).head(s.top_k or 5)
        ranks = [f"{r['date']:%d %b %Y}: wettest cell {f(r['actual_local_max_mm'])} mm ({N.rain_class(r['actual_local_max_mm'])}); state mean {f(r['actual_mm'])} mm; "
                 f"HYDRA wettest cell {f(r['local_peak_mm'])} mm" for _, r in top.iterrows()]
        lead = (f"{place.state} had {len(heavy)} day(s) with at least one ERA5 grid cell ≥ 64.5 mm (IMD 'heavy') and {len(vheavy)} ≥ 115.6 mm "
                f"('very heavy') {_period(TimeSpec('range', df['date'].min(), df['date'].max()))}.")
        return Answer(lead, [("Scale", "these are 0.25° grid-cell values, much closer to what a district experiences than the state mean."),
                             ("HYDRA", "compare HYDRA's wettest-cell forecast in each line to see whether it saw the extreme coming."),
                             ("Source", "ERA5 local maxima in the HYDRA v3 replay")], ranks, handler="extremes",
                      followups=[f"Did HYDRA predict the heaviest day in {place.state}?", f"How many days above 20 mm in {place.state}?"])
    srs = V.series(place, "rainfall", ts, s.sources)
    if not srs.ok:
        return _unavailable(place.name, "rainfall", ts, srs.message)
    arr = srs.arr()
    thr = 20.0 if srs.state_mean else 64.5
    days = [(d, v) for d, v in zip(srs.dates, arr) if v >= thr]
    lead = f"{srs.place} had {len(days)} day(s) at or above {thr} mm{' state mean' if srs.state_mean else ''} {_period(TimeSpec('range', srs.dates[0], srs.dates[-1]))}."
    return Answer(lead, [("Coverage", srs.note), ("Source", srs.source)], [f"{d}: {f(v)} mm" for d, v in days[:15]], [_src(srs)], handler="extremes")


def h_advice(q: str, s: Slots) -> Answer:
    places = _places(s)
    if not places:
        return Answer("Where is this for? Name the city or state.", handler="clarify")
    place = places[0]
    topics = s.advice or ["general"]
    ts = s.time if s.time.start else TimeSpec("date", today() + timedelta(days=1), today() + timedelta(days=1), "tomorrow", 24, True, "day")
    if re.search(r"\b(today|now|aaj|abhi)\b", q.lower()):
        ts = TimeSpec("date", today(), today(), "today", 0, True, "day")
    elif ts.end < today():
        ts = TimeSpec("range", ts.start + timedelta(days=7), ts.end + timedelta(days=7), ts.text, None, True, ts.granularity) \
            if "weekend" in (ts.text or "") else TimeSpec("date", today() + timedelta(days=1), today() + timedelta(days=1), "tomorrow", 24, True, "day")
    elif ts.start < today():
        ts = TimeSpec("range", today(), ts.end, ts.text, None, True, ts.granularity)
    if re.search(r"\bbest time to visit\b", q.lower()):
        return Answer(f"For {place.name}, the general rule: avoid the peak monsoon months if heavy rain is a concern and the hottest pre-monsoon weeks if heat is.",
                      [("Use data", f"ask 'rainfall in {place.state or place.name} in <month>' or 'trend of rainfall in {place.state or place.name}' to compare months.")], handler="advice")
    point = V.point_of(place)
    r = LC.daily([point], ["rainfall", "temperature", "wind"], ts.start, ts.end) if ts.end >= today() - timedelta(days=80) else {"status": "unavailable"}
    rain = prob = tmax = wind = None
    lines = []
    if r.get("status") == "available":
        agg = r["points"][0]["aggregate"]
        rain = agg.get("precipitation_sum", {}).get("value")
        prob = agg.get("precipitation_probability_max", {}).get("value")
        tmax = agg.get("temperature_2m_max", {}).get("value")
        wind = agg.get("wind_speed_10m_max", {}).get("value")
        lines.append(f"rain {f(rain)} mm, rain chance up to {f(prob, 0)}%, max {f(tmax)} °C, max wind {f(wind, 0)} km/h")
        source = r["provider"]
    else:
        srs = V.series(place, "rainfall", _default_window("rainfall", TimeSpec()), [])
        if srs.ok:
            rain = srs.stat("mean")
            lines.append(f"live forecast unavailable; using the latest observed rainfall ({f(rain)} mm/day, {srs.source})")
            source = srs.source
        else:
            return _unavailable(place.name, "rainfall", ts, "No live forecast and no recent observation.")
    tips = N.advice(topics, place.state, rain, prob, tmax, wind, r.get("status") != "available")
    lead = f"{place.name} {_period(ts)}: " + "; ".join(lines) + "."
    sections = [("Advice", tip) for tip in tips] + [("Source", source),
                ("Note", "general weather guidance, not an official warning; follow IMD and state disaster-management alerts.")]
    return Answer(lead, sections, handler="advice", followups=[f"Hourly detail isn't available; ask 'forecast for {place.name} next 3 days'"])


CLIMATE_DRIVERS = {
    "monsoon": "the south-west monsoon: moist westerly winds from the Arabian Sea and Bay of Bengal, lows and depressions moving along the monsoon trough, "
               "and orographic lift where winds hit the Western Ghats, the Himalaya and the north-east hills",
    "post_monsoon": "the retreating / north-east monsoon (strongest over Tamil Nadu, Puducherry, south Andhra and Kerala) and Bay of Bengal depressions or cyclones",
    "winter": "western disturbances bringing rain and snow to the north-west and the Himalaya, with dry northerlies elsewhere",
    "pre_monsoon": "afternoon thunderstorms from strong surface heating (high CAPE), nor'westers in the east, and occasional cyclones",
}


def h_why(q: str, s: Slots) -> Answer:
    places = _places(s)
    t = q.lower()
    if re.search(r"\b(hydra|model)\b", t):
        if places and places[0].state:
            s2 = Slots(places=places, time=s.time)
            return Answer("Likely reasons HYDRA misses or softens peaks:", [
                ("State averaging", "a downpour over a few districts is diluted when cells are averaged to the state."),
                ("Predictability", "convective bursts that start after the issue day are hard to anticipate a day ahead from daily-mean fields."),
                ("Loss trade-off", "even with extreme weighting, a point forecast hedges between wet and dry outcomes; check the upper interval and heavy-rain heads."),
                ("Evidence", h_model_vs_actual("biggest misses", s2).text().replace("\n", " · ")[:900])], handler="why")
        return h_model_how(q, s)
    if not places:
        return Answer("Which place and period do you mean? E.g. 'why was Kerala so wet in August 2025?'", handler="clarify")
    place = places[0]
    var = _var(s)
    ts = _default_window(var, s.time)
    month = ts.start.month if ts.start else today().month
    season = ("winter" if month in (1, 2) else "pre_monsoon" if month in (3, 4, 5) else "monsoon" if month in (6, 7, 8, 9) else "post_monsoon")
    evidence = []
    srs = V.series(place, "rainfall", ts, s.sources)
    if srs.ok:
        lead, sec, _ = describe_series(srs, ts, None)
        evidence.append(lead)
        evidence += [b for a, b in sec if a in ("Versus normal",)]
        arr = srs.arr()
        if len(arr) > 3:
            share = np.sort(arr)[::-1][:3].sum() / max(np.nansum(arr), 1e-9) * 100
            evidence.append(f"the 3 wettest days delivered {f(share, 0)}% of the period's rain" + (", so a few intense spells dominate" if share > 50 else ", so rain was spread out"))
    if H.ecmwf_date() and ts.start == ts.end == H.ecmwf_date() and place.state:
        cape, tcwv, msl, rh = (V.ecmwf_value(place, v) for v in ("cape", "water_vapour", "pressure", "humidity"))
        table = H.ecmwf_states()
        if cape and tcwv:
            evidence.append(f"ECMWF that day: CAPE {f(cape['value'], 0)} J/kg (India state median {f(table['mucape_mean'].median(), 0)}), "
                            f"water vapour {f(tcwv['value'], 0)} kg/m² (median {f(table['tcwv_mean'].median(), 0)}), pressure {f(msl['value'], 0)} hPa, humidity {f(rh['value'], 0)}%")
    lead = f"Why {place.name} {_period(ts)}: the data show what happened; the drivers below are the usual causes for this season and region."
    sections = [("What the data show", "; ".join(evidence) if evidence else "no matching observations; see coverage.")]
    sections.append(("Usual drivers", CLIMATE_DRIVERS[season]))
    if place.state in ("Kerala", "Karnataka", "Goa", "Maharashtra") and season == "monsoon":
        sections.append(("Local factor", "the Western Ghats force the moist westerlies upward, so coastal and ghat districts get some of India's heaviest rain."))
    if place.state in ("Meghalaya", "Assam", "Arunachal Pradesh") and season in ("monsoon", "pre_monsoon"):
        sections.append(("Local factor", "the Khasi-Jaintia hills and the eastern Himalaya lift Bay of Bengal moisture, giving extreme totals (Cherrapunji, Mawsynram)."))
    if place.state in ("Tamil Nadu", "Puducherry") and season == "post_monsoon":
        sections.append(("Local factor", "Tamil Nadu gets most of its annual rain from the north-east monsoon in October-December."))
    if place.state in ("Rajasthan", "Gujarat", "Ladakh"):
        sections.append(("Local factor", "far from moisture sources and at the end of the monsoon's reach, so rain is lower and more variable."))
    sections.append(("Caution", "this is an evidence-based explanation, not a diagnosed cause; the prototype has no synoptic charts for the period."))
    return Answer(lead, sections, sources=[_src(srs)] if srs.ok else [], handler="why")


# ================================================================== meta: greeting, help, coverage, imagery, provenance, out of scope
def h_greeting(q: str, s: Slots) -> Answer:
    return Answer("Hi! I'm WeatherGPT for HYDRA. Ask me about rainfall, temperature or other weather for any Indian state or city - live, forecast or past - and about how HYDRA performs.",
                  followups=["Which state got the most rain last week?", "Will it rain in Mumbai tomorrow?", "How accurate is HYDRA in Kerala?"], handler="greeting")


def h_thanks(q: str, s: Slots) -> Answer:
    return Answer("You're welcome. Ask again any time.", handler="thanks")


def h_capabilities(q: str, s: Slots) -> Answer:
    ranks = ["Live: 'weather in Pune now', 'is it raining in Kochi'",
             "Forecast: 'will it rain in Delhi tomorrow', 'next 3 days in Kolkata', 'HYDRA forecast for Kerala'",
             "Past: 'rainfall in Kerala on 20 Sep 2026', 'total rain in Odisha in August 2025', any date via the live archive",
             "Rankings: 'top 5 wettest states last week', 'which state was hottest on 24 Sep 2026'",
             "Days: 'wettest day in Assam in August 2025', 'how many days above 20 mm in Bihar'",
             "Compare: 'Kerala vs Karnataka rainfall in September 2026', 'August vs September 2025 in Gujarat'",
             "Normals: 'is Bihar in deficit', 'departure from normal this season'",
             "Trends and why: 'is rainfall increasing in Assam', 'why was Kerala so wet'",
             "HYDRA: 'how accurate is HYDRA in Maharashtra', 'which expert dominates', 'did HYDRA predict 18 Aug 2025'",
             "Advice: 'should I carry an umbrella in Bengaluru tomorrow', 'is it safe to drive to Manali'",
             "Hinglish works too: 'kal Mumbai mein barish hogi kya'"]
    return Answer("Here is what I can answer (follow-ups like 'and Goa?' keep the context):", [("Data", "ask 'what data do you have' for exact coverage.")], ranks, handler="capabilities")


def h_coverage(q: str, s: Slots) -> Answer:
    ranks = []
    for c in H.coverage():
        end = c["end"] or "16 days ahead"
        ranks.append(f"{c['source']}: {', '.join(c['variables'][:6])} · {c['start']} to {end} · {c['scope']} · {c['detail']}")
    return Answer("Data WeatherGPT can use:", [("Picking a source", "for each question I use the most authoritative source that covers the place and dates, "
                                                                    "and say which one in the answer.")], ranks, handler="coverage")


def h_imagery(q: str, s: Slots) -> Answer:
    return Answer("Radar and IMD satellite layers are visual map layers: open the Layers panel and enable 'Radar' or 'IMD satellite IR'.",
                  [("Limits", "I don't turn image pixels into rainfall numbers; ask for rainfall values instead.")], handler="imagery")


def h_provenance(q: str, s: Slots) -> Answer:
    return Answer("Every answer names its source in the 'Source' line. Where values come from:",
                  [("Observed rainfall", "IMD state-wise daily file (nwpblend/rainfall_statewise_daily_imd_clean.csv)"),
                   ("Reanalysis and HYDRA replay", "runtime/hydra_rolling_rainfall_replay.json"),
                   ("HYDRA state cycle", "runtime/hydra_daily_mean_state_blend.json"),
                   ("ECMWF", "nwpblend/forecast_india_2026-09-24_0h_clean.csv"),
                   ("Live", "IndianAPI (IMD city reports) and Open-Meteo forecast/archive APIs"),
                   ("State averages", "IMD publishes state values; ERA5/ECMWF state values average every 0.25° cell inside the state boundary.")], handler="provenance")


def h_out_of_scope(q: str, s: Slots) -> Answer:
    return Answer("That's outside what I can help with - I answer weather and HYDRA-model questions for India.",
                  followups=["What can you do?", "Which state got the most rain last week?"], handler="out_of_scope")


HANDLERS = {"greeting": h_greeting, "thanks_bye": h_thanks, "capabilities": h_capabilities, "data_coverage": h_coverage,
            "current_weather": h_current, "forecast": h_forecast, "observed": h_observed, "rank_places": h_rank_places,
            "rank_days": h_rank_days, "compare": h_compare, "threshold_days": h_threshold_days, "trend": h_trend,
            "anomaly": h_anomaly, "model_how": h_model_how, "model_weights": h_model_weights, "model_accuracy": h_model_accuracy,
            "model_vs_actual": h_model_vs_actual, "uncertainty": h_uncertainty, "extremes": h_extremes, "advice": h_advice,
            "why": h_why, "imagery": h_imagery, "provenance": h_provenance, "out_of_scope": h_out_of_scope}

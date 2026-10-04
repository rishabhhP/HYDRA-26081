"""Turn numbers into meaning: intensity classes, anomalies, trends, comparisons, model errors and advice."""
from __future__ import annotations

import numpy as np

HILL_STATES = {"Himachal Pradesh", "Uttarakhand", "Jammu and Kashmir", "Ladakh", "Sikkim", "Arunachal Pradesh",
               "Meghalaya", "Mizoram", "Nagaland", "Manipur", "Kerala", "Goa", "Karnataka", "Tripura", "Assam"}
COASTAL_STATES = {"Gujarat", "Maharashtra", "Goa", "Karnataka", "Kerala", "Tamil Nadu", "Puducherry", "Andhra Pradesh",
                  "Odisha", "West Bengal", "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"}


def f(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}"


def rain_class(mm: float | None) -> str:
    """IMD 24-hour rainfall intensity classes (station scale; state means run much lower)."""
    if mm is None:
        return "unknown"
    if mm < 0.1:
        return "no rain"
    if mm < 2.5:
        return "very light rain"
    if mm < 15.6:
        return "light rain"
    if mm < 64.5:
        return "moderate rain"
    if mm < 115.6:
        return "heavy rain"
    if mm < 204.5:
        return "very heavy rain"
    return "extremely heavy rain"


def rain_meaning(mm: float, state_mean: bool) -> str:
    cls = rain_class(mm)
    if state_mean:
        if mm >= 20:
            return (f"A state-wide average of {f(mm)} mm/day is very high: averaging over the whole state usually hides "
                    "heavy to very heavy falls in many districts on such a day.")
        if mm >= 10:
            return f"{f(mm)} mm/day averaged over the whole state is a widespread wet day; some districts likely saw heavy rain."
        if mm >= 2.5:
            return f"{f(mm)} mm/day across the state means scattered to fairly widespread light-to-moderate rain."
        if mm >= 0.1:
            return f"{f(mm)} mm/day state-wide is mostly dry with isolated showers."
        return "Essentially dry across the state."
    return f"{f(mm)} mm in a day falls in IMD's '{cls}' class."


def temp_meaning(t: float, kind: str = "mean") -> str:
    if t is None:
        return ""
    if kind == "max" and t >= 45:
        return f"{f(t)} °C is severe-heatwave territory; outdoor work in the afternoon is dangerous."
    if kind == "max" and t >= 40:
        return f"{f(t)} °C meets IMD's plains heatwave threshold (40 °C maximum); heat-stress precautions apply."
    if t >= 35:
        return f"{f(t)} °C is very hot; hydrate and avoid midday sun."
    if t >= 30:
        return f"{f(t)} °C is warm to hot."
    if t >= 20:
        return f"{f(t)} °C is mild and comfortable for most people."
    if t >= 10:
        return f"{f(t)} °C is cool; a light layer helps, especially at night."
    if t >= 4:
        return f"{f(t)} °C is cold; cold-wave conditions start near 4 °C minimum in the plains."
    return f"{f(t)} °C is very cold; frost or freezing is possible in hilly areas."


def humidity_meaning(rh: float) -> str:
    if rh is None:
        return ""
    if rh >= 85:
        return f"{f(rh, 0)}% humidity is very humid and muggy; sweat evaporates slowly, so heat feels worse."
    if rh >= 60:
        return f"{f(rh, 0)}% humidity is humid."
    if rh >= 30:
        return f"{f(rh, 0)}% humidity is comfortable."
    return f"{f(rh, 0)}% humidity is dry; dust and dehydration risk rise."


def wind_meaning(v: float, unit: str) -> str:
    kmh = v * 3.6 if unit == "m/s" else v
    if kmh is None:
        return ""
    scale = ((1, "calm"), (12, "light breeze"), (29, "moderate breeze"), (39, "fresh breeze"), (50, "strong breeze"),
             (62, "near gale"), (75, "gale"), (89, "strong gale"), (999, "storm-force"))
    label = next(name for limit, name in scale if kmh < limit)
    extra = " Fishing and small boats should be cautious." if kmh >= 40 else ""
    return f"{f(kmh, 0)} km/h is a {label} on the Beaufort scale.{extra}"


def cape_meaning(c: float) -> str:
    if c is None:
        return ""
    if c >= 2500:
        return f"CAPE of {f(c, 0)} J/kg is very unstable air: strong thunderstorms are possible if anything triggers them."
    if c >= 1000:
        return f"CAPE of {f(c, 0)} J/kg is moderately unstable: thunderstorms are plausible."
    if c >= 300:
        return f"CAPE of {f(c, 0)} J/kg is marginal instability."
    return f"CAPE of {f(c, 0)} J/kg is stable air; deep thunderstorms are unlikely."


def generic_meaning(var: str, value: float, unit: str, kind: str = "mean", state_mean: bool = True) -> str:
    if value is None:
        return ""
    if var == "rainfall":
        return rain_meaning(value, state_mean)
    if var in ("temperature", "heat_stress", "dew_point"):
        return temp_meaning(value, kind)
    if var == "humidity":
        return humidity_meaning(value)
    if var in ("wind", "wind_gust"):
        return wind_meaning(value, unit)
    if var in ("cape", "thunderstorm"):
        return cape_meaning(value)
    if var == "cloud":
        pct = value * 100 if unit == "fraction" else value
        return f"{f(pct, 0)}% cloud cover is " + ("overcast." if pct >= 80 else "mostly cloudy." if pct >= 50 else "partly cloudy." if pct >= 20 else "mostly clear.")
    if var == "pressure":
        return f"{f(value, 0)} hPa is " + ("low pressure, often linked with monsoon troughs or depressions." if value < 1002 else
                                          "near normal." if value < 1014 else "high pressure, usually settled weather.")
    if var == "water_vapour":
        return f"{f(value, 0)} kg/m² of column water vapour is " + ("very moist air that can feed heavy rain." if value >= 55 else
                                                                   "moist." if value >= 40 else "fairly dry.")
    return ""


def departure_meaning(pct: float | None, category: str | None) -> str:
    if pct is None:
        return ""
    sign = "above" if pct > 0 else "below"
    cat = (category or "").replace("_", " ")
    tone = ("much wetter than usual" if pct >= 60 else "wetter than usual" if pct >= 20 else "close to normal" if pct > -20
            else "drier than usual" if pct > -60 else "much drier than usual" if pct > -100 else "completely dry against a wet normal")
    return f"{f(abs(pct), 0)}% {sign} normal (IMD category: {cat}), i.e. {tone}."


def trend(dates: list, values: list[float]) -> dict:
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return {"ok": False}
    x = np.arange(len(y))[ok]
    y = y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    half = len(y) // 2
    first, second = y[:half].mean(), y[half:].mean()
    r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else 0.0
    strength = "clear" if abs(r) >= 0.5 else "weak" if abs(r) >= 0.2 else "no consistent"
    direction = "rising" if slope > 0 else "falling"
    return {"ok": True, "slope_per_day": slope, "first_half": first, "second_half": second, "r": r,
            "strength": strength, "direction": direction if strength != "no consistent" else "flat",
            "change_pct": (second - first) / first * 100 if first else None}


def compare_text(a_name, a, b_name, b, unit, var) -> str:
    if a is None or b is None:
        return ""
    if abs(a - b) < 1e-9:
        return f"{a_name} and {b_name} are the same ({f(a)} {unit})."
    hi, lo = (a_name, b_name) if a > b else (b_name, a_name)
    hv, lv = max(a, b), min(a, b)
    ratio = f" ({f(hv / lv, 1)}× as much)" if lv >= 1 and hv / lv <= 20 and var in ("rainfall",) else ""
    word = {"rainfall": "wetter", "temperature": "warmer", "humidity": "more humid", "wind": "windier", "wind_gust": "gustier"}.get(var, "higher")
    return f"{hi} was {word} than {lo} by {f(hv - lv)} {unit}{ratio}."


def model_error_text(actual: float, predicted: float, lo: float | None, hi: float | None) -> str:
    err = predicted - actual
    if actual is None or predicted is None:
        return ""
    if abs(err) < max(0.5, 0.1 * max(actual, 0.1)):
        verdict = "HYDRA was close"
    elif err < 0:
        verdict = f"HYDRA under-predicted by {f(-err)} mm" + (f" ({f(predicted / actual * 100, 0)}% of what fell)" if actual > 0 else "")
    else:
        verdict = f"HYDRA over-predicted by {f(err)} mm"
    inside = ""
    if lo is not None and hi is not None:
        inside = " The observed value fell inside its 80% interval." if lo <= actual <= hi else " The observed value fell outside its 80% interval."
    return verdict + "." + inside


def advice(topics: list[str], state: str | None, rain_mm: float | None, rain_prob: float | None, tmax: float | None,
           wind_kmh: float | None, state_mean: bool) -> list[str]:
    tips = []
    if rain_prob is not None:
        wet = rain_prob >= 50 or (rain_mm is not None and rain_mm >= 10)
        maybe = not wet and (rain_prob >= 25 or (rain_mm is not None and rain_mm >= 2.5))
    else:
        wet = rain_mm is not None and rain_mm >= (5 if state_mean else 2.5)
        maybe = not wet and rain_mm is not None and rain_mm >= 0.5
    very_wet = rain_mm is not None and rain_mm >= (20 if state_mean else 64.5)
    for topic in topics or ["general"]:
        if topic in ("umbrella", "general") and (rain_mm is not None or rain_prob is not None):
            tips.append("Umbrella: yes, carry one." if wet else "Umbrella: worth keeping one handy; a passing shower is possible." if maybe
                        else "Umbrella: probably not needed.")
        if topic == "travel":
            if very_wet and state in HILL_STATES:
                tips.append("Travel: heavy rain in hilly terrain raises landslide and road-closure risk; check state disaster-management and road advisories before driving.")
            elif very_wet:
                tips.append("Travel: expect waterlogging and delays in low-lying roads; allow extra time.")
            elif wet:
                tips.append("Travel: fine overall, plan for showers.")
            else:
                tips.append("Travel: weather looks favourable.")
            if tmax is not None and tmax >= 40:
                tips.append("Travel: avoid road travel and sightseeing between 12:00 and 16:00 because of heat.")
        if topic == "agriculture":
            if very_wet:
                tips.append("Farming: postpone fertiliser or pesticide spraying (washout) and clear field drainage; standing water harms most crops.")
            elif wet:
                tips.append("Farming: rain helps sowing and soil moisture; avoid spraying just before showers.")
            elif rain_mm is not None:
                tips.append("Farming: little rain expected; plan irrigation for moisture-sensitive stages.")
            tips.append("Farming: this is weather screening only; crop, soil and irrigation data are not in the prototype.")
        if topic == "outdoor":
            tips.append("Outdoor plans: " + ("keep a covered backup; rain is likely." if wet else "conditions look usable.")
                        + (" Shift strenuous activity away from the afternoon heat." if tmax is not None and tmax >= 37 else ""))
        if topic == "flood":
            if very_wet:
                tips.append("Flood screening: rainfall is in the range that can cause urban waterlogging and flash flooding in vulnerable areas.")
            else:
                tips.append("Flood screening: rainfall alone does not indicate a flood threat here.")
            tips.append("HYDRA has no river-level, drainage or terrain model, so this is a rainfall indicator, not a flood forecast.")
        if topic == "health" and tmax is not None:
            tips.append("Health: " + ("high heat-illness risk; hydrate, shade, check on elderly and children." if tmax >= 40 else
                                      "moderate heat; stay hydrated." if tmax >= 35 else "no special heat precaution needed."))
        if topic == "marine" and wind_kmh is not None:
            tips.append("Marine: " + ("unsafe for small craft." if wind_kmh >= 50 else "caution for small boats." if wind_kmh >= 30 else "sea-state risk from wind looks low."))
            tips.append("Marine: follow INCOIS / IMD fishermen warnings; HYDRA does not model waves.")
        if topic == "energy":
            tips.append("Energy: cloud and rain reduce solar output; check the radiation and cloud values above.")
        if topic == "aviation":
            tips.append("Aviation: use official METAR/TAF from the airport; WeatherGPT gives area weather only.")
    return list(dict.fromkeys(tips))

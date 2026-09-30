"""Generate deterministic JSONL intent and slot data for WeatherGPT."""

from __future__ import annotations

import json
import random
from pathlib import Path


SEED = 20260930
COUNTS = {
    "point_query": 150,
    "range_stats": 200,
    "comparison": 100,
    "conditional_filter": 100,
    "trend_anomaly": 80,
    "inference_explain": 120,
    "model_explain": 150,
    "compound": 50,
    "out_of_scope": 30,
    "clarification_needed": 20,
}
SLOT_KEYS = (
    "variable", "time_point", "time_range", "time_type", "aggregation",
    "condition", "location", "comparison_target", "inference_type",
    "model_explain_type", "units", "output_format",
)
LOCATIONS = [
    "Kochi", "Kottayam", "Thiruvananthapuram", "Kozhikode", "Wayanad",
    "Mumbai", "Pune", "Nashik", "Nagpur", "Kolhapur", "Delhi", "Noida",
    "Jaipur", "Udaipur", "Ahmedabad", "Surat", "Bengaluru", "Mysuru",
    "Chennai", "Madurai", "Coimbatore", "Hyderabad", "Visakhapatnam",
    "Vijayawada", "Bhubaneswar", "Kolkata", "Siliguri", "Guwahati", "Shillong",
    "Patna", "Ranchi", "Lucknow", "Varanasi", "Dehradun", "Srinagar",
    "Shimla", "Chandigarh", "Amritsar", "Indore", "Bhopal", "Raipur",
]
VARIABLES = [
    ("temperature", ["celsius", "fahrenheit"], ["temperature", "temp", "heat"]),
    ("max_temp", ["celsius", "fahrenheit"], ["maximum temperature", "daily high", "max temp"]),
    ("min_temp", ["celsius", "fahrenheit"], ["minimum temperature", "overnight low", "min temp"]),
    ("feels_like", ["celsius", "fahrenheit"], ["feels-like temperature", "apparent temperature", "feels like"]),
    ("rainfall", ["mm", "inches"], ["rainfall", "rain", "precipitation"]),
    ("humidity", [None], ["humidity", "relative humidity", "moisture in the air"]),
    ("wind_speed", ["kmph"], ["wind speed", "winds", "wind"]),
    ("pressure", [None], ["air pressure", "pressure", "surface pressure"]),
    ("cloud_cover", [None], ["cloud cover", "cloudiness", "cloud amount"]),
    ("rain_probability", [None], ["rain probability", "chance of rain", "rain chance"]),
]
POINTS = [
    ("today", "present", "today"), ("tomorrow", "forecast", "tomorrow"),
    ("day_after_tomorrow", "forecast", "the day after tomorrow"),
    ("next_monday", "forecast", "next Monday"), ("2025-08-12", "past", "12 Aug 2025"),
    ("2025-09-18", "past", "18 September 2025"), ("2026-01-01", "forecast", "1 Jan 2026"),
]
RANGES = [
    ("last_7_days", "past", "the last 7 days"), ("last_30_days", "past", "the last 30 days"),
    ("this_week", "present", "this week"), ("last_week", "past", "last week"),
    ("monsoon_2025", "past", "the 2025 monsoon"), ("summer_2025", "past", "summer 2025"),
    ({"start": "2025-06-01", "end": "2025-09-30"}, "past", "June to September 2025"),
    ({"start": "2025-08-01", "end": "2025-08-31"}, "past", "August 2025"),
    ({"start": "2025-07-10", "end": "2025-07-20"}, "past", "10 to 20 July 2025"),
]


def empty_slots(**values):
    result = {key: None for key in SLOT_KEYS}
    result.update(values)
    return result


def variable(rng):
    name, units, words = rng.choice(VARIABLES)
    return name, rng.choice(units), rng.choice(words)


def location(rng, allow_null=False):
    return None if allow_null and rng.random() < 0.12 else rng.choice(LOCATIONS)


def format_request(rng):
    return rng.choice([
        ("", None), (" Give only the number.", "number"),
        (" Return a short summary.", "summary"), (" Show it as a table.", "table"),
        (" Plot it as a chart.", "chart"), (" Explain it briefly.", "explanation"),
    ])


def point_query(rng):
    name, units, word = variable(rng)
    point, time_type, phrase = rng.choice(POINTS)
    place = location(rng, allow_null=True)
    place_text = f" in {place}" if place else " for this location"
    tail, output = format_request(rng)
    templates = [
        f"What is the {word}{place_text} {phrase}?",
        f"Show {word}{place_text} for {phrase}.",
        f"{word} {phrase}{place_text}",
        f"Could you give me {word}{place_text} on {phrase}?",
        f"Need the {word}{place_text}, {phrase}.",
        f"Fetch {phrase}'s {word}{place_text}.",
        f"Please report {word}{place_text} for {phrase}.",
        f"Tell me the {word}{place_text} at {phrase}.",
        f"How much {word} is expected{place_text} {phrase}?",
        f"Check {word}{place_text} for {phrase}, please.",
    ]
    return rng.choice(templates) + tail, empty_slots(variable=name, time_point=point, time_type=time_type, location=place, units=units, output_format=output)


def range_stats(rng):
    name, units, word = variable(rng)
    window, time_type, phrase = rng.choice(RANGES)
    aggregate, agg_word = rng.choice([
        ("mean", "average"), ("max", "highest"), ("min", "lowest"), ("sum", "total"),
        ("median", "median"), ("std_dev", "standard deviation"), ("percentile", "90th percentile"),
        ("count_days", "number of rainy days"),
    ])
    if aggregate == "count_days":
        name, units, word = "rainfall", "mm", "rainfall"
    place = location(rng, allow_null=True)
    place_text = f" in {place}" if place else " across the available locations"
    tail, output = format_request(rng)
    templates = [
        f"What was the {agg_word} {word}{place_text} during {phrase}?",
        f"Calculate {word}'s {agg_word}{place_text} for {phrase}.",
        f"{agg_word} {word}{place_text}, {phrase}",
        f"Give me the {agg_word} of {word}{place_text} over {phrase}.",
        f"Summarise {word}{place_text} for {phrase} using the {agg_word}.",
        f"Find the {agg_word} {word} recorded{place_text} in {phrase}.",
        f"For {phrase}, report {agg_word} {word}{place_text}.",
        f"Run a {agg_word} statistic on {word}{place_text} for {phrase}.",
        f"How did {word}{place_text} aggregate over {phrase}? Use {agg_word}.",
        f"I need {agg_word} {word}{place_text} across {phrase}.",
    ]
    return rng.choice(templates) + tail, empty_slots(variable=name, time_range=window, time_type=time_type, aggregation=aggregate, location=place, units=units, output_format=output)


def comparison(rng):
    name, units, word = variable(rng)
    time_point, time_type, phrase = rng.choice(POINTS)
    left = location(rng)
    if rng.random() < 0.55:
        right = location(rng)
        while right == left:
            right = location(rng)
        target = {"type": "location", "value": right}
        text_options = [
            f"Compare {word} in {left} and {right} for {phrase}.",
            f"Is {left} or {right} warmer/wetter for {phrase}? Check {word}.",
            f"{word}: {left} versus {right}, {phrase}",
            f"Which has more {word} on {phrase}, {left} or {right}?",
            f"Give a {word} comparison between {left} and {right} for {phrase}.",
        ]
    else:
        other, other_type, other_phrase = rng.choice(POINTS)
        target = {"type": "time", "value": other}
        text_options = [
            f"Compare {word} in {left} on {phrase} with {other_phrase}.",
            f"How will {word} in {left} differ between {phrase} and {other_phrase}?",
            f"{word} at {left}: {phrase} versus {other_phrase}",
            f"Is {left}'s {word} higher on {phrase} or {other_phrase}?",
            f"Put {left}'s {word} for {phrase} beside {other_phrase}.",
        ]
        time_type = "forecast" if "forecast" in (time_type, other_type) else time_type
    tail, output = format_request(rng)
    return rng.choice(text_options) + tail, empty_slots(variable=name, time_point=time_point, time_type=time_type, location=left, comparison_target=target, units=units, output_format=output)


def conditional_filter(rng):
    name, units, word = variable(rng)
    window, time_type, phrase = rng.choice(RANGES)
    place = rng.choice(["Kerala", "Maharashtra", "Karnataka", "Tamil Nadu", "Assam", "Rajasthan", "India"])
    if rng.random() < 0.28:
        condition = rng.choice(["rainy_day", "heatwave", "dry_spell"])
        condition_text = condition.replace("_", " ")
        aggregation = "count_days" if condition != "heatwave" else "count_days"
        name = "rainfall" if condition in {"rainy_day", "dry_spell"} else "max_temp"
        word = "rainfall" if name == "rainfall" else "maximum temperature"
    else:
        threshold = rng.choice([
            (">", "20", "mm"), (">=", "35", "celsius"), ("<", "40", "percent"),
            (">", "25", "kmph"), ("<=", "60", "percent"),
        ])
        op, value, unit = threshold
        condition = {"operator": op, "value": value, "unit": unit}
        condition_text = f"{op} {value} {unit}"
        aggregation = "count_days"
    templates = [
        f"List places in {place} where {word} was {condition_text} during {phrase}.",
        f"Find {place} locations with {word} {condition_text} over {phrase}.",
        f"Show days in {place} when {word} met {condition_text} in {phrase}.",
        f"Filter {place} by {word} {condition_text} for {phrase}.",
        f"Which areas of {place} satisfy {word} {condition_text} during {phrase}?",
        f"Count the dates in {place} with {word} {condition_text} across {phrase}.",
    ]
    tail, output = format_request(rng)
    return rng.choice(templates) + tail, empty_slots(variable=name, time_range=window, time_type=time_type, aggregation=aggregation, condition=condition, location=place, units=units, output_format=output)


def trend_anomaly(rng):
    name, units, word = variable(rng)
    window, time_type, phrase = rng.choice(RANGES)
    place = location(rng, allow_null=True)
    subject = f"in {place}" if place else "across the selected area"
    kind, inference = rng.choice([("trend", "trend_explanation"), ("anomaly", "cause")])
    templates = [
        f"What is the {kind} in {word} {subject} over {phrase}?",
        f"Detect any {word} {kind} {subject} during {phrase}.",
        f"Was {word} unusual {subject} in {phrase}?",
        f"Show the {word} {kind} for {phrase} {subject}.",
        f"Check whether {word} changed abnormally {subject} across {phrase}.",
        f"Analyse {word} for a {kind} {subject} over {phrase}.",
    ]
    tail, output = format_request(rng)
    return rng.choice(templates) + tail, empty_slots(variable=name, time_range=window, time_type=time_type, aggregation=kind, location=place, inference_type=inference, units=units, output_format=output)


def inference_explain(rng):
    name, units, word = variable(rng)
    point, time_type, phrase = rng.choice(POINTS)
    place = location(rng, allow_null=True)
    place_text = f" in {place}" if place else " for this location"
    inference = rng.choice(["why", "confidence", "risk", "advice", "cause", "trend_explanation"])
    lead = {
        "why": "Why is", "confidence": "How certain is", "risk": "What is the risk from",
        "advice": "What should I plan for given", "cause": "What may be driving", "trend_explanation": "Explain the trend in",
    }[inference]
    templates = [
        f"{lead} {word}{place_text} {phrase}?",
        f"Explain {word}{place_text} for {phrase}: {inference}.",
        f"I need {inference} guidance about {word}{place_text} on {phrase}.",
        f"Give the {inference} behind the {word} outlook{place_text} for {phrase}.",
        f"Can WeatherGPT explain {word}{place_text} at {phrase}?",
        f"Provide a {inference} assessment for {word}{place_text}, {phrase}.",
    ]
    tail, output = format_request(rng)
    return rng.choice(templates) + tail, empty_slots(variable=name, time_point=point, time_type=time_type, location=place, inference_type=inference, units=units, output_format=output or "explanation")


def model_explain(rng):
    name, units, word = variable(rng)
    point, time_type, phrase = rng.choice(POINTS)
    place = location(rng, allow_null=True)
    place_text = f" for {place}" if place else " for this forecast"
    kind = rng.choice(["head_weights", "weight_distribution", "head_contribution", "feature_importance", "weight_change", "head_reliability"])
    phrases = {
        "head_weights": ["Which Hydra heads have the most weight", "Show the Hydra head weights", "Why did the blend weight the heads this way"],
        "weight_distribution": ["Display the Hydra weight distribution", "Break down the learned weight allocation", "How is the blend weight distributed"],
        "head_contribution": ["What did each Hydra head contribute", "Show head contribution to the forecast", "Which head influenced the output most"],
        "feature_importance": ["Which input features mattered most", "Show feature importance for the Hydra forecast", "What features influenced this prediction"],
        "weight_change": ["Why did the Hydra weights change", "Explain the change in head weights", "When did the blend allocation shift"],
        "head_reliability": ["Which Hydra head is most reliable here", "Explain head reliability for this forecast", "How should I interpret head reliability"],
    }[kind]
    templates = [
        f"{rng.choice(phrases)}{place_text} {phrase} for {word}?",
        f"Hydra {kind.replace('_', ' ')}: {word}{place_text}, {phrase}.",
        f"For {word}{place_text} on {phrase}, {rng.choice(phrases).lower()}?",
        f"Please explain {rng.choice(phrases).lower()}{place_text} for {phrase}'s {word}.",
        f"Model details requested: {rng.choice(phrases).lower()}{place_text} at {phrase}.",
        f"Why did Hydra predict this {word}{place_text} on {phrase}; show {kind.replace('_', ' ')}.",
    ]
    tail, output = format_request(rng)
    return rng.choice(templates) + tail, empty_slots(variable=name, time_point=point, time_type=time_type, location=place, model_explain_type=kind, units=units, output_format=output or "explanation")


def compound(rng):
    name, units, word = variable(rng)
    point, time_type, phrase = rng.choice(POINTS)
    left, right = location(rng), location(rng)
    while right == left:
        right = location(rng)
    secondary = rng.choice(["comparison", "range_stats", "model_explain", "inference_explain"])
    if secondary == "comparison":
        text = f"Compare {word} in {left} and {right} for {phrase}, then explain why Hydra chose its head weights."
        target = {"type": "location", "value": right}
        model_kind, inference = "head_weights", "why"
    elif secondary == "range_stats":
        text = f"Give {word} in {left} for {phrase} and the average for the last 7 days, with an explanation."
        target, model_kind, inference = None, None, "trend_explanation"
    elif secondary == "model_explain":
        text = f"What is the {word} forecast in {left} for {phrase}, and show Hydra's weight distribution."
        target, model_kind, inference = None, "weight_distribution", None
    else:
        text = f"Show {word} in {left} for {phrase} and explain the risk compared with {right}."
        target, model_kind, inference = {"type": "location", "value": right}, None, "risk"
    return text, empty_slots(variable=name, time_point=point, time_type=time_type, location=left, comparison_target=target, inference_type=inference, model_explain_type=model_kind, units=units, output_format="explanation"), secondary


def out_of_scope(rng):
    texts = [
        "Will it snow in Kerala this monsoon?", "What stock should I buy after the Mumbai rain forecast?",
        "Predict Bitcoin prices using Hyderabad humidity.", "Which farm crop will make the most profit in Wayanad?",
        "Book a flight from Delhi if it rains tomorrow.", "Tell me the exact date of the next cyclone landfall.",
        "Can you diagnose my headache from Chennai pressure?", "Who will win the cricket match based on Bengaluru weather?",
        "Give the property price impact of flooding in Kochi.", "Is my school closed tomorrow because of weather?",
        "Set an alarm when rain starts in Pune.", "Should I cancel my wedding because of Jaipur's forecast?",
        "Find a hotel in Goa with perfect weather.", "Will weather make my visa application succeed?",
        "Tell me who caused the Delhi smog today.", "Can rain forecast the outcome of the election?",
        "Calculate my insurance premium from Mumbai rainfall.", "Is it safe to take medicine based on humidity?",
        "Predict earthquake risk from pressure in Guwahati.", "Which cryptocurrency rises when Chennai gets hot?",
        "Can you issue an official flood evacuation order?", "Will the monsoon fix my crop's market price?",
        "What is the best driving route based only on rain?", "Tell me the winning lottery numbers from cloud cover.",
        "Can you verify whether my office will declare a holiday?", "Will rain prove that a train will be late?",
        "Estimate the resale value of my home after a heatwave.", "Can weather data diagnose dengue in Kochi?",
        "Should I sell my shares because of wind in Mumbai?", "Predict next year's exam result from rainfall.",
    ]
    suffix = rng.choice(["", " Please be concise.", " This is for planning.", " I need a direct answer."])
    return rng.choice(texts) + suffix, empty_slots(output_format="summary")


def clarification_needed(rng):
    texts = [
        ("What is the weather in Kochi?", {"location": "Kochi"}),
        ("Will it rain tomorrow?", {"variable": "rainfall", "time_point": "tomorrow", "time_type": "forecast"}),
        ("Show the highest temperature.", {"variable": "temperature", "aggregation": "max"}),
        ("Weather for 12 Aug 2025", {"time_point": "2025-08-12", "time_type": "past"}),
        ("Compare Mumbai and Pune.", {"comparison_target": {"type": "location", "value": "Pune"}, "location": "Mumbai"}),
        ("Need a forecast chart.", {"output_format": "chart", "time_type": "forecast"}),
        ("Is it going to be bad this weekend in Kerala?", {"location": "Kerala", "time_range": "this_week", "time_type": "forecast"}),
        ("Tell me about Hydra weights.", {"model_explain_type": "head_weights", "output_format": "explanation"}),
        ("Is it hot in Mumbai?", {"location": "Mumbai", "variable": "temperature"}),
        ("Show rain for Kerala.", {"location": "Kerala", "variable": "rainfall"}),
        ("What happened last week?", {"time_range": "last_week", "time_type": "past"}),
        ("Give the wind forecast.", {"variable": "wind_speed", "time_type": "forecast"}),
        ("Need humidity statistics.", {"variable": "humidity"}),
        ("Which city was highest?", {"aggregation": "max"}),
        ("Can you compare last month?", {"time_range": "last_30_days", "time_type": "past"}),
        ("Show a rainfall graph.", {"variable": "rainfall", "output_format": "chart"}),
        ("Was the weather unusual?", {"aggregation": "anomaly"}),
        ("Why did Hydra predict that?", {"inference_type": "why", "output_format": "explanation"}),
        ("Need pressure for tomorrow.", {"variable": "pressure", "time_point": "tomorrow", "time_type": "forecast"}),
        ("What does this forecast mean?", {"output_format": "explanation", "time_type": "forecast"}),
    ]
    text, values = rng.choice(texts)
    suffix = rng.choice(["", " Please clarify.", " I need help with this.", " Ask me what is missing."])
    return text + suffix, empty_slots(**values)


def noisy(text: str, rng: random.Random) -> str:
    replacements = [("tomorrow", "tmrw"), ("temperature", "temprature"), ("rainfall", "rainfal"), ("humidity", "humdity"), ("please", "pls")]
    old, new = rng.choice(replacements)
    if old in text.lower():
        text = text.replace(old, new).replace(old.title(), new.title())
    else:
        text = rng.choice(["bhai ", "pls ", "ente ", "yaar "]) + text
    return text.replace("?", "") if rng.random() < 0.55 else text


BUILDERS = {
    "point_query": point_query, "range_stats": range_stats, "comparison": comparison,
    "conditional_filter": conditional_filter, "trend_anomaly": trend_anomaly,
    "inference_explain": inference_explain, "model_explain": model_explain,
    "out_of_scope": out_of_scope, "clarification_needed": clarification_needed,
}


def scaled_counts(total: int) -> dict[str, int]:
    """Keep the requested intent proportions when generating a smaller batch."""

    source_total = sum(COUNTS.values())
    raw = {intent: total * count / source_total for intent, count in COUNTS.items()}
    result = {intent: int(value) for intent, value in raw.items()}
    for intent, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:total - sum(result.values())]:
        result[intent] += 1
    return result


def generate(
    include_noise: bool = True,
    counts: dict[str, int] | None = None,
    seed: int = SEED,
    excluded_texts: set[str] | None = None,
):
    rng = random.Random(seed)
    counts = counts or COUNTS
    records = []
    seen_texts = set(excluded_texts or ())
    for intent, count in counts.items():
        for _ in range(count):
            if intent == "compound":
                text, slots, secondary = compound(rng)
            else:
                text, slots = BUILDERS[intent](rng)
                secondary = None
            attempts = 0
            while text in seen_texts and attempts < 500:
                attempts += 1
                if intent == "compound":
                    text, slots, secondary = compound(rng)
                else:
                    text, slots = BUILDERS[intent](rng)
            if text in seen_texts:
                raise RuntimeError(f"Could not create a unique {intent} query")
            seen_texts.add(text)
            records.append({"text": text, "intent": intent, "secondary_intent": secondary, "slots": slots})
    if include_noise:
        noisy_indexes = set(rng.sample(range(len(records)), round(len(records) * 0.15)))
        for index in noisy_indexes:
            records[index]["text"] = noisy(records[index]["text"], rng)
    return records


def main():
    output = Path(__file__).resolve().parents[1] / "training" / "weathergpt_intent_slots_1000.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = generate()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output)
    print(len(records))


if __name__ == "__main__":
    main()

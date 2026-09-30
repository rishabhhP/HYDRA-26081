"""Generate 50 realistic WeatherGPT acceptance queries in JSONL form."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "training" / "acceptance" / "weathergpt_forecast_app_50.jsonl"
SLOT_KEYS = ("variable", "time_point", "time_range", "time_type", "aggregation", "condition", "location", "comparison_target", "inference_type", "model_explain_type", "units", "output_format")


def case(text, intent, expected=None, secondary_intent=None, **slots):
    filled = {key: None for key in SLOT_KEYS}
    filled.update(slots)
    return {"text": text, "intent": intent, "secondary_intent": secondary_intent, "slots": filled, "expected": expected or {}}


CASES = [
    case("Will it rain at grid cell 12000 tomorrow?", "point_query", {"weather_variable": "rainfall", "grid_cell": "12000", "lead": 24}),
    case("What is the temperature at 11.50, 84.50 in 48 hours?", "point_query", {"weather_variable": "temperature", "lead": 48}),
    case("Show wind at grid cell 12000 tomorrow.", "point_query", {"weather_variable": "wind", "grid_cell": "12000", "lead": 24}),
    case("Humidity in Kerala now", "point_query", {"weather_variable": "humidity", "domain": "live_layers"}),
    case("Current rainfall in Maharashtra", "point_query", {"weather_variable": "rainfall", "domain": "live_layers"}),
    case("Wind in Tamil Nadu today", "point_query", {"weather_variable": "wind", "domain": "live_layers"}),
    case("What is the next 24 hour rain outlook in Kerala?", "point_query", {"weather_variable": "rainfall", "domain": "live_layers"}),
    case("Lightning watch in Maharashtra today", "point_query", {"weather_variable": "thunderstorm", "layer": "open_meteo_lightning", "domain": "live_layers"}),
    case("Thunderstorm potential in Karnataka now", "point_query", {"weather_variable": "thunderstorm", "layer": "open_meteo_thunderstorm", "domain": "live_layers"}),
    case("Wind gusts in Kerala today", "point_query", {"weather_variable": "wind_gust", "layer": "open_meteo_wind_gust", "domain": "live_layers"}),
    case("What was rainfall in Kerala on 20 September 2026?", "range_stats", {"weather_variable": "rainfall", "domain": "live_layers", "location": "Kerala"}, variable="rainfall", time_point="2026-09-20", time_type="past", location="Kerala"),
    case("Which state had the highest rainfall from 20 September 2026 to 25 September 2026?", "range_stats", {"weather_variable": "rainfall", "domain": "live_layers", "scope": "states"}, variable="rainfall", time_range={"start": "2026-09-20", "end": "2026-09-25"}, time_type="past", aggregation="sum"),
    case("Where had the highest temperature from 20 September 2026 to 25 September 2026?", "range_stats", {"weather_variable": "temperature", "domain": "live_layers", "operation": "location_ranking"}, variable="temperature", time_range={"start": "2026-09-20", "end": "2026-09-25"}, time_type="past", aggregation="max"),
    case("Which month was wettest in India in 2014?", "range_stats", {"weather_variable": "rainfall", "bucket": "historical_rainfall_2014", "operation": "time_ranking"}, variable="rainfall", time_range="2014", time_type="past", aggregation="max", location="India"),
    case("What was the highest rainfall in India in 2014?", "range_stats", {"weather_variable": "rainfall", "bucket": "historical_rainfall_2014"}, variable="rainfall", time_range="2014", time_type="past", aggregation="sum", location="India"),
    case("What was rainfall at 22.75, 75.75 on 2014-07-01?", "point_query", {"weather_variable": "rainfall", "bucket": "historical_rainfall_2014"}, variable="rainfall", time_point="2014-07-01", time_type="past"),
    case("Where was humidity highest over the last 30 days?", "range_stats", {"weather_variable": "humidity", "domain": "live_layers"}, variable="humidity", time_range="last_30_days", time_type="past", aggregation="max"),
    case("Where are wind gusts strongest today?", "range_stats", {"weather_variable": "wind_gust", "layer": "open_meteo_wind_gust", "domain": "live_layers"}, variable="wind_speed", time_point="today", time_type="present", aggregation="max"),
    case("Compare rainfall in Mumbai and Pune tomorrow.", "comparison", {"operation": "comparison"}, variable="rainfall", time_point="tomorrow", time_type="forecast", location="Mumbai", comparison_target={"type": "location", "value": "Pune"}),
    case("How will temperature in Kochi differ between today and tomorrow?", "comparison", {"operation": "comparison"}, variable="temperature", time_point="today", time_type="present", location="Kochi", comparison_target={"type": "time", "value": "tomorrow"}),
    case("Is Bengaluru or Chennai windier tomorrow?", "comparison", {"operation": "comparison"}, variable="wind_speed", time_point="tomorrow", time_type="forecast", location="Bengaluru", comparison_target={"type": "location", "value": "Chennai"}),
    case("Compare the 24 hour rain outlook for Kerala and Tamil Nadu.", "comparison", {"operation": "comparison"}, variable="rainfall", time_point="tomorrow", time_type="forecast", location="Kerala", comparison_target={"type": "location", "value": "Tamil Nadu"}),
    case("List Kerala locations where rainfall was above 20 mm during the last 7 days.", "conditional_filter", {}, variable="rainfall", time_range="last_7_days", time_type="past", aggregation="count_days", condition={"operator": ">", "value": "20", "unit": "mm"}, location="Kerala"),
    case("Find places in Rajasthan with maximum temperature at least 35 celsius this week.", "conditional_filter", {}, variable="max_temp", time_range="this_week", time_type="present", aggregation="count_days", condition={"operator": ">=", "value": "35", "unit": "celsius"}, location="Rajasthan"),
    case("Show days in Maharashtra with a dry spell over the last 30 days.", "conditional_filter", {}, variable="rainfall", time_range="last_30_days", time_type="past", aggregation="count_days", condition="dry_spell", location="Maharashtra"),
    case("Which Assam areas had a heatwave in summer 2025?", "conditional_filter", {}, variable="max_temp", time_range="summer_2025", time_type="past", aggregation="count_days", condition="heatwave", location="Assam"),
    case("What is the rainfall trend in Kerala over the last 30 days?", "trend_anomaly", {}, variable="rainfall", time_range="last_30_days", time_type="past", aggregation="trend", location="Kerala", inference_type="trend_explanation"),
    case("Was temperature unusual in Mumbai during August 2025?", "trend_anomaly", {}, variable="temperature", time_range={"start": "2025-08-01", "end": "2025-08-31"}, time_type="past", aggregation="anomaly", location="Mumbai", inference_type="cause"),
    case("Detect a humidity anomaly in Chennai for the last week.", "trend_anomaly", {}, variable="humidity", time_range="last_week", time_type="past", aggregation="anomaly", location="Chennai", inference_type="cause"),
    case("Explain the wind trend in Bengaluru this week.", "trend_anomaly", {}, variable="wind_speed", time_range="this_week", time_type="present", aggregation="trend", location="Bengaluru", inference_type="trend_explanation"),
    case("Why is rainfall forecast at grid cell 12000 tomorrow?", "inference_explain", {"weather_variable": "rainfall", "grid_cell": "12000", "lead": 24}, variable="rainfall", time_point="tomorrow", time_type="forecast", inference_type="why"),
    case("How certain is the temperature forecast at 11.50, 84.50 tomorrow?", "inference_explain", {"weather_variable": "temperature", "lead": 24}, variable="temperature", time_point="tomorrow", time_type="forecast", inference_type="confidence"),
    case("Explain the heavy-rain risk for grid cell 12000 tomorrow.", "inference_explain", {"weather_variable": "rainfall", "grid_cell": "12000", "lead": 24}, variable="rainfall", time_point="tomorrow", time_type="forecast", inference_type="risk"),
    case("What should farmers plan for if it rains in Kerala tomorrow?", "inference_explain", {"weather_variable": "rainfall"}, variable="rainfall", time_point="tomorrow", time_type="forecast", location="Kerala", inference_type="advice"),
    case("What is causing the wind forecast at grid cell 12000?", "inference_explain", {"weather_variable": "wind", "grid_cell": "12000"}, variable="wind_speed", time_type="forecast", inference_type="cause"),
    case("Show HYDRA head weights for rainfall at grid cell 12000 tomorrow.", "model_explain", {"weather_variable": "rainfall", "grid_cell": "12000", "evidence": True}, variable="rainfall", time_point="tomorrow", time_type="forecast", model_explain_type="head_weights", output_format="explanation"),
    case("Give the HYDRA weight distribution for temperature at 11.50, 84.50 tomorrow.", "model_explain", {"weather_variable": "temperature", "evidence": True, "lead": 24}, variable="temperature", time_point="tomorrow", time_type="forecast", model_explain_type="weight_distribution", output_format="explanation"),
    case("Which HYDRA head contributes most to wind at grid cell 12000?", "model_explain", {"weather_variable": "wind", "grid_cell": "12000", "evidence": True}, variable="wind_speed", time_type="forecast", model_explain_type="head_contribution", output_format="explanation"),
    case("Why did the Hydra weights change for rainfall tomorrow?", "model_explain", {"weather_variable": "rainfall", "lead": 24, "evidence": True}, variable="rainfall", time_point="tomorrow", time_type="forecast", model_explain_type="weight_change", output_format="explanation"),
    case("How reliable is each Hydra head for temperature at grid cell 12000?", "model_explain", {"weather_variable": "temperature", "grid_cell": "12000", "evidence": True}, variable="temperature", time_type="forecast", model_explain_type="head_reliability", output_format="explanation"),
    case("What features influenced the rainfall prediction for grid cell 12000 tomorrow?", "model_explain", {"weather_variable": "rainfall", "grid_cell": "12000", "lead": 24}, variable="rainfall", time_point="tomorrow", time_type="forecast", model_explain_type="feature_importance", output_format="explanation"),
    case("Explain the saved 80% interval for rainfall at grid cell 12000 tomorrow.", "model_explain", {"weather_variable": "rainfall", "grid_cell": "12000", "lead": 24}, variable="rainfall", time_point="tomorrow", time_type="forecast", model_explain_type="head_reliability", output_format="explanation"),
    case("What do the Hydra expert weights mean?", "model_explain", {"bucket": "hydra_model_methodology", "operation": "explanation"}, model_explain_type="head_weights", output_format="explanation"),
    case("How does Hydra make a weather forecast?", "model_explain", {"bucket": "hydra_model_methodology", "operation": "explanation"}, model_explain_type="feature_importance", output_format="explanation"),
    case("Is radar available right now?", "point_query", {"domain": "imagery", "layer": "radar"}, time_point="today", time_type="present"),
    case("What does the IMD satellite layer show?", "inference_explain", {"domain": "imagery", "layer": "satellite"}, inference_type="why"),
    case("What weather questions can HYDRA answer?", "inference_explain", {"operation": "capability"}, inference_type="advice"),
    case("What data is missing for fresh HYDRA inference?", "model_explain", {"bucket": "hydra_model_methodology", "operation": "explanation"}, model_explain_type="feature_importance", output_format="explanation"),
    case("What is rainfall at 19.076, 72.8777 tomorrow, and show the Hydra weights.", "compound", {"weather_variable": "rainfall", "lead": 24, "evidence": True}, secondary_intent="model_explain", variable="rainfall", time_point="tomorrow", time_type="forecast", model_explain_type="weight_distribution", output_format="explanation"),
    case("Compare Mumbai and Pune rainfall tomorrow, then explain the forecast confidence.", "compound", {"operation": "comparison"}, secondary_intent="inference_explain", variable="rainfall", time_point="tomorrow", time_type="forecast", location="Mumbai", comparison_target={"type": "location", "value": "Pune"}, inference_type="confidence", output_format="explanation"),
]


def main() -> None:
    if len(CASES) != 50:
        raise ValueError(f"Expected 50 cases, found {len(CASES)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in CASES:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(OUTPUT)
    print(len(CASES))


if __name__ == "__main__":
    main()

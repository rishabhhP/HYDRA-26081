# HYDRA Weather Query Parser

This standalone FastAPI service turns a weather question into structured intent for WeatherGPT. It does **not** retrain or run the `nwpblend` forecast models. The parser is deterministic: optional Gemini text cleanup runs first, then location, variable, time, and event extraction validates the result.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn weather_query_parser.main:app --host 127.0.0.1 --port 8010
```

`GET /health` returns the service status. `POST /parse` accepts `{ "query": "chance of heavy rain in Chennai tomorrow" }`.

## Output

```json
{
  "original_query": "chance of heavy rain in Chennai tomorrow",
  "normalized_query": "chance of heavy rain in Chennai tomorrow",
  "ai_normalization_used": false,
  "parsed": {
    "location": "Chennai",
    "location_suggestion": null,
    "coordinates": {"lat": 13.0827, "lon": 80.2707},
    "weather_variable": "rainfall",
    "time_range": {"kind": "lead_hours", "lead_hours": 24, "start_date": null, "end_date": null, "text": "tomorrow"},
    "extreme_event_type": "heavy_rain"
  }
}
```

HYDRA uses `0`, `24`, `48`, and `72` hours, so exact single-day expressions map to those values. Multi-day requests such as “next week” stay explicit date ranges; the service does not silently choose one forecast step. Location suggestions are non-destructive: `Mumbay` yields `location: null` and `location_suggestion: "Mumbai"`.

The parser recognises the live map layers: current and next-24-hour rainfall, humidity, thunderstorm potential, wind-gust risk, heat stress, soil moisture, radar, and IMD satellite imagery. Questions using `today`, `now`, `current`, or `live` route temperature, rainfall, and wind to the live provider layers; explicit forecasts route to the trained HYDRA output. “Where” and “highest” questions rank only the actual available provider locations or model cells.

It also labels agriculture, aviation, marine, energy, urban, and emergency use cases. A use-case label changes the evidence and limitation shown in the answer; it never fabricates an unsupported crop-suitability, flood-impact, or exposure result. Unlisted place names are retained and resolved through Open-Meteo's India-only geocoding result before a live weather query is made.

## Optional Gemini cleanup

Set `GEMINI_API_KEY` and `GEMINI_MODEL` to enable cleanup. Gemini is never authoritative: any missing configuration or failure returns the original text to the deterministic parser.

## WeatherGPT connection

Set `HYDRA_QUERY_PARSER_URL=http://127.0.0.1:8010/parse` for the HYDRA API. WeatherGPT then uses exact parsed coordinates or recognised states and compatible lead times to select its evidence, and shows the parsed intent alongside the answer.

## Trained intent and slot evidence

`training/weathergpt_intent_slots_1000.jsonl` is the versioned corpus for the
optional learned intent and categorical-slot classifier. Train it with:

```powershell
.\.venv\Scripts\python.exe weather_query_parser\tools\train_weathergpt_intent_model.py
```

The command writes the classifier and held-out metrics to `training/models/`.
The `/parse` response includes `parsed.trained_intent` with its predicted
intent, confidence, and categorical-slot candidates. It is assistive evidence
only: deterministic rules remain authoritative for location matching, dates,
coordinates, source routing, and HYDRA model decisions.

To create ten additional diverse batches of 200 examples and retrain on the
combined corpus, run:

```powershell
.\.venv\Scripts\python.exe weather_query_parser\tools\generate_weathergpt_training_iterations.py
.\.venv\Scripts\python.exe weather_query_parser\tools\train_weathergpt_intent_model.py
```

The trainer automatically reads the base corpus and every JSONL file in
`training/iterations/`, rejects duplicate query text, and records the exact
datasets, counts, and held-out metrics in the model metrics file.

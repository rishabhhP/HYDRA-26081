# HYDRA v2 prototype

Map-first weather intelligence workspace with React/TypeScript, Leaflet and FastAPI. Scientific model code and checkpoints in `nwpblend/` are unchanged. Read `ARCHITECTURE.md` and `DATASET_ASSESSMENT.md` for the integration boundaries.

## Run locally (PowerShell, from HYDRA)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd install --prefix frontend
npm.cmd run build --prefix frontend
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. API documentation: http://127.0.0.1:8000/api/docs.
For frontend development run `npm.cmd run dev --prefix frontend` in a second terminal (port 5173, API proxy to 8000).

## Included working integrations

- Nine repository forecast rows: three real archived grid cells × 24/48/72-hour leads. Forecasts, expert weights, spread, CQR intervals and calibrated threshold probabilities are read directly from the CSV, with source SHA-256 and repository commit.
- September 24, 2026 ECMWF analysis across the supplied India grid. Switch source in the timeline. Zero-hour precipitation accumulation is not shown as daily rainfall.
- Repository held-out benchmark tables by target, season and lead.
- Historical rainfall NetCDF point queries (2014 file downloaded from crop-yield-analysis); missing values remain missing. Open Historical rainfall in the layer panel. The observation date is independent and explicitly displayed.
- IMD state CSV through `/api/imd?state=KERALA`.
- Deterministic weather scenarios and grounded evidence explanations. No risk numbers or live observations are fabricated.
- Independent floating panels, world/India/Kerala views, map picking, layer search, command palette, timeline, and analytics dialogs.
- Clickable boundaries for all 36 Indian states and union territories. A state click opens an area average from the supplied ECMWF analysis and the matching latest supplied IMD rainfall row. The HYDRA archive remains limited to its actual three cells.
- Stackable renderers for forecast points, rainfall heatmaps, temperature cells, wind arrows, disagreement rings, event rings, expert diamonds and regime squares.
- **Live IMD station weather** is a stackable map layer backed by IndianAPI. It fetches reports for distributed Indian cities through the server, caches them for 30 minutes, and never sends the provider key to the browser. Markers are live reporting stations rather than interpolated nationwide coverage.

The local Natural Earth basemap works without a tile service. Fonts fall back to system fonts when offline. Natural Earth geographic data is public domain, obtained from https://github.com/nvkelso/natural-earth-vector. India state boundaries are supplied by `vardhan-maps`, generated from OpenStreetMap and licensed under ODbL 1.0; they are best-effort operational boundaries rather than survey-grade geometry.

## Fresh trained-model inference

The supplied history and shared climatology requirements are still unmet. Copy the original `climatology_by_doy.npz` into `nwpblend/artifacts/blocked/` and supply the daily history Parquet described in that repository. Do not replace it with the rainfall-only NetCDF or the monthly Indore data.

```powershell
$env:HYDRA_HISTORY = 'C:\path\to\era5_india_2025_daily_processed.parquet'
.\.venv\Scripts\python.exe -m backend.inference --history $env:HYDRA_HISTORY --issue 2025-12-28
```

This validates 35 consecutive days and the complete trained grid, loads the original `BlendingForecaster`, performs inference, and writes an atomic forecast cycle plus checkpoint/history hashes under `runtime/`. Select **Fresh inference output** in the UI. This worker is intentionally separate from browser requests. CPU is supported. Runtime depends on available cores and memory.

### Historical model summary for WeatherGPT

For a timeline question such as “where is highly prone to heavy rainfall based on one year of data?”, build a repeated-inference summary first. The worker runs the unchanged trained forecaster once per target day, then stores total predicted rainfall and expected classifier-event days per grid cell. It needs the original full-grid history through the requested period plus the preceding 35 days, and the original climatology artifact.

```powershell
.\.venv\Scripts\python.exe -m backend.historical_inference --history $env:HYDRA_HISTORY --start 2025-01-01 --end 2025-12-31 --lead 1
```

WeatherGPT reads only the published summary whose coverage exactly matches the requested window. It does not reuse a one-day forecast as a one-year risk estimate.

`HYDRA_MODE=production` defaults the UI to fresh output; it never silently substitutes archive data. `HYDRA_MODEL_REPO` can change the server-side model repository path. No API accepts arbitrary filesystem paths. Keep the prototype on localhost; authentication and deployment hardening are needed before internet exposure.

## Attach a trained NLP model

WeatherGPT currently falls back to a deterministic evidence explainer. Set `HYDRA_NLP_API_URL` and, when needed, `HYDRA_NLP_API_KEY` to use the adapter in `backend/nlp.py`. HYDRA sends `{ "question": string, "context": HydraContext }`; the service returns `{ "answer": string, "model"?: string }`. The context includes the selected state or cell, valid time, forecast values, intervals, learned weights, regime, events and provenance. API credentials stay server-side. Restart the API after changing environment variables.

The service must treat the structured context as evidence, preserve units and valid times, and decline unsupported claims. Add authentication, rate limiting and response logging before exposing it publicly.

## Connect the deterministic WeatherGPT query parser

`weather_query_parser/` is a separate FastAPI service adapted from the supplied deterministic parser. It is vocabulary and rule based, so there is no model retraining step. It optionally uses Gemini only to normalize wording; a failure always falls back to deterministic extraction.

```powershell
cd weather_query_parser
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn weather_query_parser.main:app --host 127.0.0.1 --port 8010

# In the HYDRA API environment:
$env:HYDRA_QUERY_PARSER_URL = 'http://127.0.0.1:8010/parse'
```

WeatherGPT then parses Indian states/UTs and the prototype's mapped cities, coordinates, rainfall/temperature/wind, `0/24/48/72` hour leads, and the three `nwpblend` classifier events: heavy rain, heatwave, and high wind. Requests that name multi-day windows remain date ranges rather than being silently reduced to one forecast lead.

## WeatherGPT question catalog

WeatherGPT uses one shared intent catalog rather than a separate rule or model for each sentence. It extracts the data bucket, parameter, time range, geographic scope, requested operation, and ranking direction. The machine-readable version is available at `GET /api/weathergpt-capabilities`.

| Bucket | Parameters | Supported question types | Coverage limit |
| --- | --- | --- | --- |
| 2014 historical rainfall observations | rainfall, year, India/state scope, annual or monthly aggregation, highest/lowest | location ranking, wettest/driest month, point lookup, aggregate | Daily 0.25° rainfall grid for 2014 only |
| Live weather and retained timeline | temperature, rainfall, humidity, thunderstorm potential, gusts, heat stress, soil moisture, location, retained days | current lookup, location ranking, timeline ranking | Representative provider locations; timeline begins when HYDRA collects data |
| HYDRA short-range forecast | rainfall, temperature, wind, lead, state/location, heavy-rain/heatwave/high-wind event | point forecast, location ranking, forecast explanation | Limited to the supplied archive/fresh-inference coverage |
| Radar and satellite imagery | radar, satellite, time | availability and imagery explanation | Image data are not converted into city measurements |

For example, “Which month had the most rainfall in India in 2014?” resolves to the historical-rainfall bucket, a month-ranking operation, descending rainfall, India scope, and the 2014 calendar-year window. “What questions can HYDRA answer?” returns this catalog through WeatherGPT.

The full prototype-specific question set and response contract are in [docs/weathergpt_question_set.md](docs/weathergpt_question_set.md).

## Connect IndianAPI live weather

Create an IndianAPI Weather API key and set it only in the environment that runs FastAPI:

```powershell
$env:INDIAN_WEATHER_API_KEY = 'your-key'
$env:INDIAN_WEATHER_CACHE_MINUTES = '30' # optional; minimum is 5
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Enable **Live IMD station weather** in the map’s Observations layer. HYDRA calls IndianAPI's `/india/weather?city=...` endpoint server-side and plots its IMD-backed reports as amber-edged temperature markers. Hover a marker for the station name, daily high/low, rainfall, humidity, and forecast description. The layer is intentionally not interpolated between stations.

## Retain live-layer history

HYDRA stores the latest sample for each mapped city and UTC day in `runtime/live_weather_daily.csv`. Start the collector to keep recording after the app is launched:

```powershell
.\.venv\Scripts\python.exe -m backend.live_collector --interval-minutes 30
```

WeatherGPT can then answer timeline questions about retained live layers, such as humidity, current temperature, rainfall outlook, gusts, heat stress, and soil moisture. It uses only days that were actually saved and reports when the requested window has not yet accumulated.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
npm.cmd run build --prefix frontend
```

Browser tests (with API running on port 8000): `npm.cmd test --prefix frontend`. First install its browser with `node frontend/node_modules/@playwright/test/cli.js install chromium`.

## Docker

`docker compose up --build` serves port 8000. Model repo, sample observations and optional runtime outputs are mounted read-only. Supply production files on the host and run the inference worker locally or in a separate container. The image is CPU-oriented and includes PyTorch.

## Known limits

This prototype is not an operationally validated forecasting service. Archived HYDRA coverage is only three cells. Kochi and most locations have no archived HYDRA forecast; ECMWF analysis is available where the supplied grid covers them. Other NWP experts, global weather fields, operational ingestion, calibrated confidence/bust models, impact/exposure models and an LLM are integration pending. Historical benchmark skill is not local forecast verification. All pending modules explain the missing evidence.

# HYDRA prototype

Map-first weather intelligence workspace with React/TypeScript, Leaflet and FastAPI. It includes the real-data HYDRA rainfall v3 replay, the separate published daily-mean state cycle, and a WeatherGPT intent/slot model. Read `ARCHITECTURE.md` and `DATASET_ASSESSMENT.md` for integration boundaries.

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
- Clickable boundaries for all 36 Indian states and union territories. Selecting a state loads the separately published HYDRA daily-mean neural-gating forecast at +24h and +48h.
- Stackable renderers for forecast points, rainfall heatmaps, temperature cells, wind arrows, disagreement rings, event rings, expert diamonds and regime squares.
- **Live IMD station weather** is a stackable map layer backed by IndianAPI. It fetches reports for distributed Indian cities through the server, caches them for 30 minutes, and never sends the provider key to the browser. Markers are live reporting stations rather than interpolated nationwide coverage.

## Published HYDRA state forecast

The dashboard includes a retrained HYDRA daily-mean state cycle under `runtime/hydra_daily_mean_state_blend.json`. It is issued on **2025-12-31** and provides forecasts for **2026-01-01 (+24h)** and **2026-01-02 (+48h)** across all 36 Indian states and UTs.

For each state, lead, and target, HYDRA's neural gate blends four experts:

- climatology
- persistence
- recent-three-day persistence
- anomaly-persistence

The AI Insights state panel and its state-aware WeatherGPT briefing show the blend forecast, learned expert allocations, expert disagreement, an empirical 80% residual interval, and held-out chronological backtest MAE/RMSE. Expert weights show blend allocation; they do not establish causal feature importance. Disagreement is spread between expert forecasts. The error metrics describe held-out 2025 backtests and are not future observed error.

Open-Meteo is shown only as a live +24h/+48h provider comparison. It never replaces a HYDRA value, changes gate weights, or verifies this historical HYDRA issue cycle.

To rebuild the published state cycle after staging compatible daily-mean ERA5 inputs, run:

```powershell
.\.venv\Scripts\python.exe scripts/train_hydra_daily_mean_state_blend.py
```

The raw ERA5 archives are intentionally excluded from Git because of their size. The compact published HYDRA artifacts are committed under `runtime/hydra_daily_mean_state_blend/`.

## Statistics: six-month HYDRA rainfall v3 replay

The upper taskbar **Statistics** shows the published HYDRA rainfall v3 historical replay for every Indian state and UT. It covers **2025-07-01 through 2025-12-31**. For each valid day, HYDRA uses only data available on its issue day, produces a **+24-hour rainfall** prediction from its trained neural gate, and records nine experts: climatology, persistence, recent-three-day, anomaly-persistence, ERA5 gradient boosting, wet/dry hurdle, spatial neighbourhood, monsoon, and upper quantile.

The chart places the HYDRA blend, each expert, its conformal central 80% interval, and the held-back ERA5 state-average rainfall on one timeline. It also shows rain probability, heavy-rain area probabilities, local peak estimates, backtest error, and gate dynamics. Actual rainfall is comparison data only; it is never supplied to the forecast for the same day. The Overview does not use Open-Meteo or any hypothetical external-model line.

To rebuild the published replay after staging the compatible 2025 archives:

```powershell
.\.venv\Scripts\python.exe scripts/build_hydra_rolling_rainfall_replay.py
```

This writes `runtime/hydra_rolling_rainfall_replay.json`, which the local `/api/hydra-rolling-rainfall` endpoint serves to the Overview.

### v3.2 data validation and post-processing

The v3.2 add-on adds a separate, gauge-based truth path and a post-processing evaluation layer. It does not replace the published replay until its out-of-sample evaluation is complete.

- `hydra_imd/` downloads IMD 0.25° daily gridded rainfall, checks its grid layout and land/sea mask, then writes state means and local maxima to `runtime/imd_state_daily.csv`.
- `hydra_post/` compares ERA5 with IMD truth, evaluates rainfall amounts and heavy-rain probabilities at both +24 h and +48 h, uses issue-time-only lagged observations, and produces asymmetric 80% intervals plus calibrated alert probabilities.
- The rainfall feature builder now supports moisture-flux convergence and CAPE × humidity predictors. `HYDRA_EXCLUDE_FEATURES` enables the supplied ablation runner to compare them against the baseline on identical folds.

Install the optional download clients and run the sequence below after configuring a CDS API key in `%USERPROFILE%\.cdsapirc`:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m hydra_imd.download --years 2015-2025 --dir data/raw/imd
.\.venv\Scripts\python.exe -m hydra_imd.build --dir data/raw/imd
.\.venv\Scripts\python.exe -m hydra_post.truth_check
.\.venv\Scripts\python.exe scripts/download_era5_parallel.py --years 2020-2024 --workers 4
.\.venv\Scripts\python.exe scripts/run_feature_ablation.py --source-dir data/raw/era5_all_years --max-folds 8 --step-days 60
.\.venv\Scripts\python.exe scripts/build_hydra_rolling_rainfall_replay.py --source-dir data/raw/era5_all_years
.\.venv\Scripts\python.exe -m hydra_post.evaluate --truth imd
.\.venv\Scripts\python.exe -m hydra_post.evaluate --truth imd --lead 2
.\.venv\Scripts\python.exe -m hydra_post.apply
```

The IMD and CDS source data are excluded from Git. The IMD provider can be temporarily unavailable; its download can be rerun without affecting the prototype. Evaluation reports, not in-sample post-processed rows, determine whether a new post-processor is published to the UI.

### v3.3 calibrated replay release

The rainfall replay page includes a calibrated HYDRA panel when post-processing artifacts are available. It shows out-of-sample rainfall amounts, balanced 80% intervals, calibrated heavy-rain probabilities, alert tiers, state reliability, and the release-gate result. Historical WeatherGPT heavy-rain questions use the same replay-only artifacts.

Run the release gate after an evaluation and post-processing run:

```powershell
.\.venv\Scripts\python.exe -m hydra_post.release
```

The UI labels a release **Validated** only when its required checks pass. A failed check keeps the release **Provisional** and displays the reason. Before rebuilding the replay, snapshot the published artifacts; then check or roll back the result as needed:

```powershell
.\.venv\Scripts\python.exe scripts/publish_release.py snapshot --note "before multi-year retrain"
.\.venv\Scripts\python.exe scripts/publish_release.py check
.\.venv\Scripts\python.exe scripts/publish_release.py rollback
```

The calibrated panel uses out-of-sample rows whenever available. It does not present fitted rows as forecasts.

### v3.4 IMD-target release

The current rainfall release evaluates the post-processed HYDRA replay against IMD state-average rainfall for **2025-08-01 through 2025-12-31**. Each test month is predicted using only earlier data with a one-day embargo. The published artifacts include the post-processed rows, calibrated heavy-rain probabilities, adaptive 80% intervals, the release-gate result, and separate +48-hour evaluation evidence.

The current +24-hour out-of-sample release reduces amount error from **2.314 to 1.101 mm/day MAE** and from **5.180 to 3.495 mm/day RMSE** against IMD. Its calibrated probabilities have positive Brier skill for the 10 mm and 20 mm state thresholds. The release remains **Provisional**: adaptive 80% interval coverage is 86.2%, above the 75–85% target, and 34 of 36 states do not beat persistence on MAE. The dashboard presents this status and its evidence instead of calling the release validated.

The final interface keeps forecast, observation, impact, and HYDRA diagnostic layers separate. Weather regime, forecast-confidence, and forecast-bust-risk controls are intentionally absent because no calibrated operational artifacts support them. The **Events** workspace is reserved for the upcoming event-intelligence feature.

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

## WeatherGPT

WeatherGPT has two distinct entry points:

- The **taskbar WeatherGPT** chat handles general trained question families: current weather, forecast lookups, dates and date ranges, rankings, historical rainfall coverage, comparisons, conditional filters, trends, uncertainty, and HYDRA model explanations.
- **Ask WeatherGPT about [state]** in AI Insights opens a state-aware briefing. It uses the selected state's published HYDRA payload directly and formats the forecast, learned weights, disagreement, backtest metrics, 80% interval, and provider comparison without substituting a provider forecast.

The intent/slot artifact is stored at `weather_query_parser/training/models/weathergpt_intent_slot_model.joblib`. It works with deterministic location, date, and evidence resolution so every answer can state its source and coverage. The complete question set and response contract are in [docs/weathergpt_question_set.md](docs/weathergpt_question_set.md).

Set `HYDRA_NLP_API_URL` and, when needed, `HYDRA_NLP_API_KEY` only to attach an additional external response adapter through `backend/nlp.py`. It receives structured HYDRA context and must preserve units, valid times, and source limits. API credentials stay server-side.

### WeatherGPT v2 add-on

The included `weathergpt/` package adds a trained 24-intent classifier, deterministic slot extraction, data-source routing, formatted answers, and per-tab follow-up context. It handles broad state, city, date, historical, live-weather, forecast, model-accuracy, and uncertainty questions. The taskbar chat sends a session ID and renders suggested follow-up questions.

WeatherGPT v2 is enabled by default for the taskbar chat, which sends a session ID. Existing API clients without a session ID retain a configured external parser/adapter, so their current behavior remains unchanged. V2 returns `None` for grid-cell archive questions, literal coordinate questions, 2014 rainfall-grid questions, unsupported external-model questions, and one-year model-ranking questions, allowing the original implementation to answer those unchanged. Set `WEATHERGPT_V2=0` to disable the add-on without removing it. Model status, evaluation metadata, and published data coverage are available at `GET /api/weathergpt-v2/status`.

See [WeatherGPT v2 documentation](docs/WEATHERGPT_V2.md) for its supported intents, source priority, test sets, and retraining command.

## Connect the deterministic WeatherGPT query parser

`weather_query_parser/` is a separate FastAPI service that combines the trained intent/slot artifact with deterministic parsing for dates, locations, and evidence boundaries. The deterministic layer remains the fallback when a request is outside the learned model's supported coverage.

```powershell
cd weather_query_parser
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn weather_query_parser.main:app --host 127.0.0.1 --port 8010

# In the HYDRA API environment:
$env:HYDRA_QUERY_PARSER_URL = 'http://127.0.0.1:8010/parse'
```

WeatherGPT parses Indian states/UTs and mapped cities, coordinates, rainfall/temperature/wind, date points and date ranges, forecast leads, model-weight questions, and the three `nwpblend` classifier events: heavy rain, heatwave, and high wind. Requests that name multi-day windows remain date ranges rather than being silently reduced to one forecast lead.

## WeatherGPT question catalog

WeatherGPT uses one shared intent catalog rather than a separate rule or model for each sentence. It extracts the data bucket, parameter, time range, geographic scope, requested operation, and ranking direction. The machine-readable version is available at `GET /api/weathergpt-capabilities`.

| Bucket | Parameters | Supported question types | Coverage limit |
| --- | --- | --- | --- |
| 2014 historical rainfall observations | rainfall, year, India/state scope, annual or monthly aggregation, highest/lowest | location ranking, wettest/driest month, point lookup, aggregate | Daily 0.25° rainfall grid for 2014 only |
| Live weather and retained timeline | temperature, rainfall, humidity, thunderstorm potential, gusts, heat stress, soil moisture, location, retained days | current lookup, location ranking, timeline ranking | Representative provider locations; timeline begins when HYDRA collects data |
| HYDRA state neural-gating forecast | rainfall, temperature, wind, +24h/+48h, state, expert weights, disagreement, interval, backtest error | state forecast, model explanation, weight distribution, uncertainty explanation | Published 2025-12-31 cycle for 36 India states/UTs; valid 2026-01-01 and 2026-01-02 |
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

## HYDRA rainfall v3

`hydra_rain/` replaces the state-average rainfall replay model. Baseline scores of the published v2 replay, computed with the v3 metric code, are in `reports/rainfall_replay_baseline_v2.json` (Maharashtra: MAE 3.36 mm/day, RMSE 5.17, MAE 15% worse than persistence, state-mean >= 20 mm recall 15%, "80%" band coverage 90.8%, expert weight std 0.0006).

What changed:

- **Grid-cell first.** Training and prediction run on every 0.25 degree ERA5 cell; results are aggregated to states at the end, together with the wettest cell, the expected area above 20 / 64.5 mm/day and the highest cell heavy-rain probability.
- **Atmospheric predictors.** Temperature, dew point and depression, humidity, CAPE, pressure and its tendency/anomaly, wind components, cloud, radiation, boundary-layer height, plus total-column water vapour and gusts when the archives include them. Neighbourhood (+-0.5 and +-1 degree) rain, CAPE and moisture summaries.
- **Nine experts.** Climatology, persistence, recent-three-day and anomaly persistence as baselines, plus ERA5 gradient boosting, a wet/dry hurdle model, a spatial-neighbourhood model, a monsoon specialist and an 85th-percentile model. The gate learns from out-of-fold expert predictions.
- **Multi-head gate.** Expert weights, P(rain >= 1 mm), P(>= 20), P(>= 64.5), P(>= state wet-day p95), amount if wet, and lower/upper interval heads. Squared error is weighted up on heavy-rain days. No zero-initialised static anchor, lower weight decay, longer training, and weight-dynamics reporting.
- **Correct 80% intervals.** Stratified split-conformal CQR at alpha = 0.20 (state x season x lead x regime with hierarchical fallback), fitted on a pre-replay slice the experts and gate never trained on. The daily-mean state forecast script now uses a conformal 80% half-width too; its already-published cycle is flagged in the UI until retrained.
- **Rolling-origin validation** across years, seasons, states, leads, regimes and intensities with MAE, RMSE, bias, rain/no-rain skill, heavy-rain precision/recall, peak error, interval coverage and width, and skill against persistence and climatology.

Run (put one or more years of ERA5 daily-mean archives anywhere under the source folder):

```powershell
.\.venv\Scripts\python.exe scripts/build_hydra_rolling_rainfall_replay.py --source-dir data/raw/era5_daily
.\.venv\Scripts\python.exe scripts/validate_hydra_rainfall.py --source-dir data/raw/era5_daily
.\.venv\Scripts\python.exe scripts/evaluate_rainfall_replay.py   # scores whatever replay is published
```

`--synthetic-dry-run` on the first two scripts checks the pipeline on made-up weather and writes `*.dry_run.*` files only; those numbers mean nothing. Rolling validation retrains once per origin, so a multi-year run takes a while on CPU.

Published real-data result: the current v3 replay contains 184 dates for all 36 states. Across 6,624 state-days it has MAE **3.80 mm/day**, RMSE **7.47 mm/day**, and a negative bias of **0.82 mm/day**. Its central 80% intervals covered **89.9%** of held-back state means, so they are conservative. It beats climatology on MAE (17.0% skill) but is marginally worse than persistence overall (−2.7% MAE skill). It should be used for rainfall occurrence and low-to-moderate state-average rainfall, not as an accurate standalone predictor of extreme rainfall.

## Docker

`docker compose up --build` serves port 8000. Model repo, sample observations and optional runtime outputs are mounted read-only. Supply production files on the host and run the inference worker locally or in a separate container. The image is CPU-oriented and includes PyTorch.

## Known limits

This prototype is not an operationally validated forecasting service. The published HYDRA state cycle is a separate 2025 daily-mean cycle with a fixed 2025-12-31 issue date; it is not the rainfall v3 replay and is not continuously refreshed. The v3 rainfall replay substantially underestimates state-average heavy rain: its >=20 mm/day recall is 24.4%, and its largest observed state mean was predicted at 2.3% of the observed amount. It must not be used alone for extreme-rain decisions. The older three-cell archive remains limited to its supplied coverage. Other NWP experts, global weather fields, operational ingestion, calibrated confidence/bust models, impact/exposure models, and an LLM are integration pending. Historical benchmark skill is not local forecast verification. All pending modules explain the missing evidence.

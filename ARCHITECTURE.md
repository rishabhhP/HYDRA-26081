# HYDRA prototype architecture and integration plan

## Repository analysis
`nwpblend` supplies adaptive gating, static blends, seasonal LightGBM, quantile/CQR models, isotonic extreme classifiers, and benchmark CSVs. The experts are climatology, persistence, recent3, and anomaly persistence. Its original Python source and checkpoints remain unchanged. Leads are 1, 2, 3 days; targets are daily rain, mean temperature, and mean wind. Gating consumes 41 numeric columns and four categorical columns in checkpoint metadata order, applies saved median imputation and normalization, and returns four softmax weights per row. CPU inference is supported; no GPU is required.

## Data flow and model integration
1. Archived mode reads the committed nine forecast rows (three cells × three leads) with source hashes. No spatial extrapolation beyond half a grid cell.
2. ECMWF analysis mode reads the supplied September 24, 2026 zero-hour snapshot. It is explicitly separate from HYDRA forecasts.
3. Observations read supplied daily NetCDF rainfall at the nearest in-domain cell, honoring missing values. IMD state observations are exposed separately.
4. Production inference wraps `BlendingForecaster.forecast`. Validate full grid, consecutive days, required variables, valid leads, and shared artifacts before calling unchanged preprocessing. Store outputs and provenance atomically for subsequent API reads. No retraining or checkpoint modification.
5. Verification serves actual held-out benchmark tables; never presents these as verification for a selected live location.
6. Events expose only available calibrated classifier probabilities. Risk/exposure and unsupported sources return integration-pending states. Scenarios perturb deterministic weather values only and carry a simulation label.

## Backend and API
FastAPI serves typed requests, validated geographic bounds, forecast context, bounded GeoJSON layers, observation queries, verification, source registry, health, scenario and grounded evidence explanations. Heavy inference runs through a command-line worker, outside request handling. Forecast cache keys include file identity and cycle. `/api/docs` provides OpenAPI. Local files are configured server-side, never supplied through arbitrary request paths.

## Frontend
React + TypeScript + Vite, Leaflet map with local world GeoJSON fallback, dark charcoal and amber visual system from the reference. Full-screen map remains behind independently collapsible panels and reusable floating dialogs. Shared context owns location, lead, variable and source; all panels use the same response. Taskbar launches situation, models, events, risk, verification, sectors, scenario, WeatherGPT and operations. Responsive panels, loading/errors, searchable layers and command palette are included.

## Implementation order
Contracts/adapters → backend → map/state → panels/dialogs → tests/build → deployment. Run instructions live in README.md and are verified against the implemented commands.

## Limits
ERA5 history and climatology_by_doy.npz remain absent. Supplied rainfall observations cannot substitute for the trained atmospheric history. Archive covers only three grid cells. No global forecasts, operational live feeds, calibrated confidence score, cyclone/flood/landslide impact model or exposure inventory was supplied. WeatherGPT initially uses deterministic evidence-grounded responses without claiming an LLM connection. Browser shows explicit limitations. This is a runnable prototype, not an operationally validated forecasting service.

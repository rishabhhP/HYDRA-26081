# HYDRA rainfall v3 change list

New
- hydra_rain/ (config, grid, features, experts, gate, calibration, pipeline, report, metrics, validate)
- scripts/validate_hydra_rainfall.py, scripts/evaluate_rainfall_replay.py
- tests/test_hydra_rain.py, reports/rainfall_replay_baseline_v2.json

Rewritten
- scripts/build_hydra_rolling_rainfall_replay.py (v3 model; payload keeps every v2 field)
- frontend/src/RollingRainfall.tsx (evaluation scorecards, probability and wettest-cell charts, gate-weight chart, legacy warning)

Edited
- scripts/train_hydra_daily_mean_state_blend.py (conformal 80% half-width instead of p90)
- backend/data.py, backend/main.py (v3 fields, legacy interval labels, /api/hydra-rainfall-validation)
- frontend/src/Analytics.tsx, api.ts, types.ts, style.css; tests/test_api.py; README.md; .gitignore

Verified after import
- Fixed a read-only NumPy view in neural-gate training and made the configured OOF fold count apply at call time. These are compatibility/testability fixes; the v3 model design is unchanged.

Not yet verified here: gate training under torch (test_gate_training_parity_with_torch), frontend build, any run on real ERA5.

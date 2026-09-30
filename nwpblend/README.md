# nwpblend — HYDRA ML core: Model Processing & Adaptive Blending

The ML core of **HYDRA**, a hybrid AI-NWP weather-forecast blending system for India:
data processing, adaptive blending models and benchmarking. **Not** included: backend API,
dashboard, chatbot (hand-off schema in §6).

**Benchmark results: [`reports/BENCHMARK_REPORT.md`](reports/BENCHMARK_REPORT.md)** — the main deliverable.

## 1. Quick start

```bash
pip install pandas numpy scikit-learn lightgbm torch xarray netCDF4 pyarrow scipy matplotlib pytest
python scripts/run_pipeline.py all      # ingest -> features -> train -> evaluate -> report  (~1 h on 16 cores)
python -m pytest -q tests               # leakage + schema tests (inference test skips without artifacts/data)
python scripts/run_pipeline.py demo     # inference from saved artifacts
```

### Reproducing from raw data (the repo does not contain it)

Raw and processed data are git-ignored (~11 GB). To reproduce, place these in the repo root:

| path | what | source |
|---|---|---|
| `c73e70ccac6586b08e6c16143c48cab5/data_stream-oper_stepType-accum.nc` | ERA5 single levels 2025, 6-hourly (00/06/12/18 UTC): `tp, ssrd` | Copernicus CDS, *ERA5 hourly data on single levels*, area N37/W68/S7/E97, 0.25° |
| `c73e70ccac6586b08e6c16143c48cab5/data_stream-oper_stepType-instant.nc` | same request: `u10, v10, d2m, t2m, sst, sp, tcc, cape, cin, blh, msl` | as above (CDS splits the download into these two files) |
| `rainfall_statewise_daily_imd_clean.csv` | IMD statewise daily rainfall, Aug 19 – Sep 24 2026 | committed (small) |
| `export_meteostat_2025_clean.csv` | Meteostat station daily 2025 | committed (small) |
| `forecast_india_2026-09-24_0h_clean.csv` | ECMWF oper 0 h snapshot, cleaned | committed (small); raw `.grib2` not needed |
| `model_skill_summary.csv` | earlier prototype's Q4 skill (report §7) | committed (small) |

Paths are set in `nwpblend/config.py`. Then `python scripts/run_pipeline.py all` rebuilds everything:
`data/processed/` (daily truth archive + feature tables), `artifacts/`, `reports/results/` and the report.

## 2. Package layout

| module | responsibility |
|---|---|
| `nwpblend/config.py` | every threshold, split parameter and hyper-parameter (cited by the report) |
| `nwpblend/ingest.py` | raw ERA5 NetCDF → daily truth parquet; IMD / Meteostat / ECMWF loaders; pure-python GRIB inventory |
| `nwpblend/splits.py` | blocked-within-season split with embargo (primary), time-forward split (stress test) |
| `nwpblend/features.py` | experts, train-only climatology, rolling skill, regimes, static features; `FeatureBuilder` |
| `nwpblend/models/lgbm_season.py` | (a) per-season LightGBM |
| `nwpblend/models/gating.py` | (b) neural gating network (+ static blend baseline), numpy-exportable |
| `nwpblend/models/quantile.py` | (c) LightGBM quantile models + conformal (CQR) calibration |
| `nwpblend/models/extremes.py` | (d) extreme-event classifiers + isotonic calibration |
| (e) disagreement | `expert_spread` / `expert_range` features (features.py) + evaluation in `evaluation/evaluate.py` |
| `nwpblend/pipeline.py` | training + test predictions |
| `nwpblend/evaluation/` | `metrics.py`, `evaluate.py` (all benchmark tables), `external.py` (station / IMD / ECMWF), `figures.py`, `report.py` |
| `nwpblend/inference.py` | `BlendingForecaster` — the backend entry point |
| `tests/` | leakage, schema, split-integrity and export-parity tests |

## 3. Data (what is actually in the workspace)

* **Truth**: the brief's `era5_india_2025_daily_processed_csv.gz` is *not* in the workspace. Its raw source is:
  `c73e70.../data_stream-oper_stepType-{accum,instant}.nc` — ERA5 6-hourly 2025 on the same 121×117
  0.25° India grid. `ingest.build_era5_daily()` rebuilds the daily archive
  (`data/processed/era5_india_2025_daily_processed.parquet`, 5.17 M cell-days) with the brief's columns
  (`date, day_of_year, month, season, latitude, longitude, is_ocean, tp_mm, t2m_C_mean/min/max,
  wind_speed_mean/min/max, *_flag`) plus `d2m, rh, msl, sp, tcc, cape, blh, ssrd, elev_proxy_m`.
  Daily tp = mean of four hourly accumulations × 24.
* **Extreme flags** (defined here, since the pre-computed file is absent): heavy rain `tp_mm ≥ 64.5`;
  heatwave `land & t2m_C_max ≥ 40`; high wind `wind_speed_max ≥ 13.9 m/s`.
* **ECMWF snapshot**: single 0 h analysis (tp ≡ 0) with no truth at its valid time → not scorable, not a trained expert.
* **IMD statewise 2026**, **Meteostat 2025**: external validation only.
* `72d1.../data.grib`: global ERA5, February 2025 only (soil/runoff fields) → inventoried, not used.

## 4. Feature schema

One row = (grid cell, target day *d*, lead *h* ∈ {1,2,3} days). Issue time = end of day *i = d − h*.
**Contract: every feature uses truth from days ≤ i only, plus statistics fitted on TRAIN days only**
(enforced by `tests/test_features.py::test_features_never_see_truth_after_issue_day`).

| group | columns | definition |
|---|---|---|
| ids | `date, cell, lead_days, split` | `date` = target day |
| static | `latitude, longitude, is_ocean, elev_proxy_m, coastal, region_id` | `is_ocean` from ERA5 SST mask; `elev_proxy_m` hypsometric from annual-mean sp/msl/t2m (no orography field available); `coastal` = land/ocean mix in 3×3 neighbourhood; `region_id` = 3°×3° block (`r*100+c`) |
| calendar | `season_id, month, day_of_year, doy_sin, doy_cos` | of the target day; seasons IMD (winter JF, pre-monsoon MAM, monsoon JJAS, post-monsoon OND). Calendar columns *other than season* are excluded from the final gating/LightGBM inputs (they let models memorise training days — see report). |
| lead | `lead_days, lead_bucket_id` | bucket 0=`0-6h` (ECMWF 0 h only), 1=`6-24h` (lead 1 d), 2=`24-72h` (lead 2–3 d) |
| experts | `exp_climatology, exp_persistence, exp_recent3, exp_anom_persistence` | clim = train-only mean over ±30 d of the target day-of-year excluding ±5 d; persistence = truth(i); recent3 = mean truth(i-2..i); anom-persistence = clim(d) + truth(i) − clim(i) (rain clipped ≥ 0) |
| historical skill | `mae7_<expert>, rmse30_<expert>` | trailing MAE (7 target days) / RMSE (30) of each expert, ending at the issue day |
| | `regmae_<expert>` | train-split MAE of each expert per region × season × lead |
| regime | `regime_id` | rule-based on issue-day state, priority order: cyclonic (min MSLP ≤ train-mean − 6 hPa & max wind ≥ 10 m/s) > heatwave (land Tmax ≥ 40) > monsoon (JJAS & 3-day rain ≥ 5 mm/d) > wet_spell (same, other seasons) > dry_spell (land, 7-day rain < 1 mm) > normal |
| current state | `now_<var>` for tp, t2m mean/max, wind mean/max, msl mean/min, tcc, cape, rh, blh, ssrd; `now_msl_anom, now_tp3, now_tp7, now_target_tendency, now_anomaly` | issue-day values |
| disagreement (e) | `expert_spread, expert_range` | std / range across the 4 expert forecasts |
| labels | `target, heavy_rain_day_flag, heatwave_day_flag, high_wind_day_flag` | on the target day |

## 5. Saved artifacts & inference call

`artifacts/blocked/` (primary model set). **In git:** all trained models and small tables (~17 MB).
**Git-ignored (49 MB):** `climatology_by_doy.npz`, which the forecaster needs. It is a train-only
statistic, so it can be regenerated **without retraining** once the raw data is in place:

```bash
python scripts/run_pipeline.py ingest
python -c "from nwpblend.features import build_all; build_all('blocked', cellsets=())"
python -c "from nwpblend.pipeline import save_shared_artifacts; save_shared_artifacts('blocked')"
```

(The split is seeded, so this reproduces the identical file.) Alternatively, attach the file to a GitHub
Release and download it into `artifacts/blocked/`. `artifacts/time_forward/` (stress-test models) is
regenerated by `run_pipeline.py train`.

```
manifest.json                   targets, leads, experts, flags, thresholds, n_train_days
climatology_by_doy.npz          <target> -> (367, 14157) train-only climatology by day-of-year; msl_train_mean
region_season_skill.csv         train-only region x season x lead x expert MAE
grid_static.parquet             per-cell static features (cell index = row-major lat desc, lon asc)
<target>/gating.{pt,npz,json}         adaptive gating network (npz+json = torch-free, see gating.numpy_forward)
<target>/gating_static.{pt,npz,json}  static season x lead blend
<target>/lgbm_<season>.txt, lgbm_meta.json     per-season LightGBM (LightGBM text format)
<target>/quantile_qNN.txt, quantile_meta.json  quantile boosters + conformal widening per season
<target>/clf_<flag>.txt, clf_<flag>_calibration.json   classifier + isotonic knots (apply with np.interp)
```

```python
from nwpblend.inference import BlendingForecaster

fc = BlendingForecaster("artifacts/blocked")
out = fc.forecast(
    history,                 # DataFrame, daily-archive schema, FULL grid, >= 35 days ending on issue_date
    issue_date="2025-12-28",
    leads=(1, 2, 3),         # days ahead
    cells=None,              # optional subset of cell ids (default: all 14,157)
    nwp=None,                # optional {target: DataFrame[latitude, longitude, value]} raw NWP field
    nwp_weight=None,         # optional {target: float in [0,1]} - user-set, NOT learned (default 0)
)
```

Output: one row per `cell × target_date × lead_days` with, for each target
`t ∈ {tp_mm, t2m_C_mean, wind_speed_mean}`:
`pred_t` (adaptive blend — the primary forecast), `lgbm_t`, `static_t`, `exp_<expert>_t`, `w_<expert>_t`
(gating weights, sum to 1), `spread_t` (disagreement), `q05_t … q95_t`, `lo80_t / hi80_t`
(conformal 80 % interval), `nwp_weight_t`; plus `prob_heavy_rain_day_flag`, `prob_heatwave_day_flag`,
`prob_high_wind_day_flag` (calibrated) and `regime`.

## 6. Hand-off (not built here)

* **Backend** — call `BlendingForecaster.forecast`; persist the output columns above.
* **Dashboard** — weight maps `reports/results/blocked/w_cell.csv`, `w_season_region.csv`; skill tables `imp_*.csv`, `pm_*.csv`; `reliability.csv`; figures in `reports/figures/`.
* **Chatbot** — `regime`, `w_<expert>_t` and the auto-generated weight sentences in report §8.

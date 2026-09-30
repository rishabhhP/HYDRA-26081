"""Central configuration: paths, physical thresholds, split + model hyper-parameters.

Every constant that affects a benchmark number lives here so the report can cite it.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- raw inputs
ERA5_DIR = ROOT / "c73e70ccac6586b08e6c16143c48cab5"
ERA5_ACCUM_NC = ERA5_DIR / "data_stream-oper_stepType-accum.nc"
ERA5_INSTANT_NC = ERA5_DIR / "data_stream-oper_stepType-instant.nc"
ERA5_FEB_GRIB = ROOT / "72d1ab8885fcf7c2abb77942cc8283c5" / "data.grib"  # inventoried, not used
ECMWF_FC_CSV = ROOT / "forecast_india_2026-09-24_0h_clean.csv"
ECMWF_FC_GRIB = ROOT / "20260924000000-0h-oper-fc.grib2"
IMD_STATEWISE_CSV = ROOT / "rainfall_statewise_daily_imd_clean.csv"
METEOSTAT_CSV = ROOT / "export_meteostat_2025_clean.csv"
PROTOTYPE_SKILL_CSV = ROOT / "model_skill_summary.csv"
PROTOTYPE_PRED_CSV = ROOT / "blended_predictions_2025_Q4_test.csv"

# ---------------------------------------------------------------- outputs
PROCESSED_DIR = ROOT / "data" / "processed"
ERA5_DAILY_PARQUET = PROCESSED_DIR / "era5_india_2025_daily_processed.parquet"
FEATURES_DIR = PROCESSED_DIR / "features"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------- problem definition
TARGETS = ["tp_mm", "t2m_C_mean", "wind_speed_mean"]
FLAGS = ["heavy_rain_day_flag", "heatwave_day_flag", "high_wind_day_flag"]
# which per-target feature table each extreme classifier is trained on
FLAG_TABLE = {
    "heavy_rain_day_flag": "tp_mm",
    "heatwave_day_flag": "t2m_C_mean",
    "high_wind_day_flag": "wind_speed_mean",
}
LEADS = [1, 2, 3]  # forecast horizon in days (issue day = target day - lead)
EXPERTS = ["climatology", "persistence", "recent3", "anom_persistence"]

# IMD seasons
SEASON_OF_MONTH = {
    1: "winter", 2: "winter",
    3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon", 12: "post_monsoon",
}
SEASONS = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]

# ---------------------------------------------------------------- extreme-event labels
HEAVY_RAIN_MM = 64.5        # IMD "heavy rain" (24 h accumulation)
HEATWAVE_TMAX_C = 40.0      # IMD plains absolute criterion, land cells only
HIGH_WIND_MAX_MS = 13.9     # Beaufort 7 lower bound ("near gale"), on max of 6-hourly 10 m wind

# ---------------------------------------------------------------- regimes (rule-based)
REGIMES = ["cyclonic", "heatwave", "monsoon", "wet_spell", "dry_spell", "normal"]
CYCLONIC_MSL_ANOM_HPA = -6.0   # daily-min MSLP below cell's train mean by this much ...
CYCLONIC_WIND_MS = 10.0        # ... AND daily-max wind at least this
MONSOON_TP3_MM = 5.0           # 3-day mean rain for an "active" wet regime
DRY_TP7_MM = 1.0               # 7-day total below this = dry spell (land)

# ---------------------------------------------------------------- spatial
GRID_STRIDE = 2              # 0.25 deg grid -> train/benchmark on every 2nd lat & lon (0.5 deg subgrid)
REGION_DEG = 3.0             # coarse region blocks for region x season skill + weight maps
SPATIAL_HOLDOUT_CELLS = 1200  # off-subgrid cells sampled for the spatial generalisation test

# ---------------------------------------------------------------- climatology
CLIM_HALF_WINDOW = 30        # +-days around target day-of-year
CLIM_EXCLUDE = 5             # but excluding +-this many days (limits same-weather leakage)
CLIM_MIN_COUNT = 10          # widen window if fewer train days than this

# ---------------------------------------------------------------- splits
SEED = 42
BLOCK_DAYS = 7               # contiguous day blocks, assigned within season
EMBARGO_DAYS = 2             # train days within this many days of a val/test day are dropped
MIN_TEST_BLOCKS_PER_SEASON = 2
MIN_VAL_BLOCKS_PER_SEASON = 1
TEST_FRAC = 0.2
VAL_FRAC = 0.15
WARMUP_DAYS = 7              # first issue day index with full rolling-7 history

# time-forward stress-test split (reproduces the earlier prototype's Q4 hold-out)
TF_TRAIN_END = "2025-08-31"
TF_VAL_END = "2025-09-30"

# ---------------------------------------------------------------- models
LGBM_PARAMS = dict(
    objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    verbose=-1, num_threads=16, seed=SEED,
)
LGBM_ROUNDS = 1500
LGBM_EARLY_STOP = 100
LGBM_MAX_TRAIN_ROWS = 1_200_000

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QUANTILE_ROUNDS = 600
QUANTILE_MAX_TRAIN_ROWS = 500_000

CLF_ROUNDS = 1000
CLF_MAX_TRAIN_ROWS = 1_500_000

GATING_HIDDEN = [128, 64]
GATING_EPOCHS = 12
GATING_BATCH = 4096
GATING_LR = 1e-3
GATING_PATIENCE = 3
GATING_MAX_TRAIN_ROWS = 700_000
GATING_DROPOUT = 0.2
GATING_WEIGHT_DECAY = 1e-3
GATING_EXCLUDE: tuple = ()
GATING_ANCHORED = True

MIN_POSITIVES_TRUST = 50     # below this many positives in a test slice, metrics are flagged untrustworthy
MIN_POSITIVE_DAYS_TRUST = 5  # ... or fewer distinct positive days (spatial clustering)

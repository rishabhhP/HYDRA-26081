"""HYDRA rainfall v3 configuration. Every constant that changes a reported number lives here."""
from __future__ import annotations

MODEL_VERSION = "hydra-rain-v3"

# ---------------------------------------------------------------- grid
LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST = 37.0, 7.0, 68.0, 97.0
GRID_SHAPE = (121, 117)          # 0.25 deg ERA5 over the HYDRA India box

# ---------------------------------------------------------------- problem
LEADS = (1, 2)                   # days ahead
WET_MM = 1.0                     # rain occurrence threshold (cell, mm/day)
HEAVY_THRESHOLDS_MM = (20.0, 64.5)   # fixed heavy-rain heads; 64.5 = IMD "heavy rain"
STATE_PERCENTILE = 0.95          # extra heavy head: cell rain >= state's training-period wet-day p95
HEAVY_KEYS = tuple(f"{t:g}" for t in HEAVY_THRESHOLDS_MM) + ("state_p95",)

SEASON_OF_MONTH = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3}
SEASONS = ("winter", "pre_monsoon", "monsoon", "post_monsoon")
REGIMES = ("dry_spell", "normal", "wet_spell", "monsoon_active")
DRY_TP7_MM = 1.0
WET_TP3_MM = 5.0
MONSOON_TP3_MM = 10.0

# ---------------------------------------------------------------- climatology (train days only)
CLIM_WINDOWS = (15, 30, 60, 90, 183)   # circular day-of-year half windows, widened when sparse
CLIM_EXCLUDE_DAYS = 5                  # never use truth within +-5 calendar days of the target date
CLIM_MIN_COUNT = 10

# ---------------------------------------------------------------- neighbourhood
NEIGHBOURHOODS = (5, 9)          # square windows in cells: 5 = +-0.5 deg, 9 = +-1 deg

# ---------------------------------------------------------------- training data volume
TRAIN_CELL_STRIDE = 2            # every 2nd lat/lon for fitting (all cells used for prediction)
MAX_EXPERT_ROWS = 900_000
MAX_GATE_ROWS = 900_000
SEED = 42

# ---------------------------------------------------------------- chronological splits inside training data
CALIBRATION_FRAC = 0.15          # last 15% of training days: conformal calibration only
GATE_VAL_FRAC = 0.15             # days before that: gate early stopping
EXPERT_OOF_FOLDS = 3             # contiguous blocks for out-of-fold expert predictions
EMBARGO_DAYS = 2

# ---------------------------------------------------------------- trained experts (scikit-learn HistGradientBoosting)
GBM_PARAMS = dict(learning_rate=0.06, max_iter=400, max_leaf_nodes=63, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=25, random_state=SEED)
UPPER_QUANTILE = 0.85            # extreme-aware expert

# ---------------------------------------------------------------- gate
GATE_HIDDEN = (128, 64)
GATE_DROPOUT = 0.1
GATE_WEIGHT_DECAY = 1e-4         # v2 used 1e-3 plus a zero-initialised correction, which froze the weights
GATE_LR = 2e-3
GATE_EPOCHS = 40
GATE_PATIENCE = 5
GATE_BATCH = 4096
# extreme-aware loss: sample weight = 1 + EXTREME_ALPHA * min(y / EXTREME_REF_MM, EXTREME_CAP)
EXTREME_ALPHA = 3.0
EXTREME_REF_MM = 20.0
EXTREME_CAP = 4.0
LOSS_OCC = 0.5
LOSS_HEAVY = 1.0
LOSS_QUANTILE = 0.5
LOSS_WET_AMOUNT = 0.3
ENTROPY_PENALTY = 0.01           # small push towards decisive (row-specific) expert choice
INTERVAL_QUANTILES = (0.10, 0.90)   # raw heads before conformal correction -> central 80%

# ---------------------------------------------------------------- conformal calibration
ALPHA = 0.20                     # central 80% interval
MIN_STRATUM = 50                 # cell-level strata
MIN_STATE_STRATUM = 30           # state-level strata
STATE_STRATA = (("state", "season", "lead", "regime"), ("state", "season", "lead"),
                ("season", "lead", "regime"), ("season", "lead"), ())
CELL_STRATA = STATE_STRATA

# ---------------------------------------------------------------- rolling-origin validation
ROLLING_MIN_TRAIN_DAYS = 120
ROLLING_STEP_DAYS = 30
ROLLING_HORIZON_DAYS = 30

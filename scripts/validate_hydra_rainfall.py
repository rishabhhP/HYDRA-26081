"""Rolling-origin validation for HYDRA rainfall v3 across years, seasons, states and intensities.

    python scripts/validate_hydra_rainfall.py --source-dir data/raw/era5_multi_year
    python scripts/validate_hydra_rainfall.py --step-days 30 --horizon-days 30 --max-folds 12

Writes runtime/hydra_rainfall_validation.json (served at /api/hydra-rainfall-validation)
and reports/HYDRA_RAINFALL_VALIDATION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_rain import config as C  # noqa: E402
from hydra_rain import validate as V  # noqa: E402
from hydra_rain.grid import load_era5, synthetic_cube  # noqa: E402

SOURCE_DIR = ROOT / "data" / "raw" / "era5_2025_full_source"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "hydra_rainfall_validation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "HYDRA_RAINFALL_VALIDATION.md")
    parser.add_argument("--step-days", type=int, default=C.ROLLING_STEP_DAYS)
    parser.add_argument("--horizon-days", type=int, default=C.ROLLING_HORIZON_DAYS)
    parser.add_argument("--min-train-days", type=int, default=C.ROLLING_MIN_TRAIN_DAYS)
    parser.add_argument("--first-origin")
    parser.add_argument("--last-origin")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--target", choices=["era5", "imd"], default="era5")
    parser.add_argument("--imd-dir", type=Path, default=ROOT / "data" / "raw" / "imd")
    parser.add_argument("--imd-realtime")
    parser.add_argument("--imd-day-shift", default="auto")
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args()
    train_gate = True
    if args.synthetic_dry_run:
        cube = synthetic_cube(start="2023-01-01", days=900)
        try:
            import torch  # noqa: F401
        except ImportError:
            train_gate = False
        args.output = args.output.with_name("hydra_rainfall_validation.dry_run.json")
        args.report = args.report.with_name("HYDRA_RAINFALL_VALIDATION.dry_run.md")
    else:
        cube = load_era5(args.source_dir)
    truth = None
    if args.target == "imd" and not args.synthetic_dry_run:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_hydra_rolling_rainfall_replay import imd_truth
        truth = imd_truth(cube, args)
    result = V.run(cube, args.step_days, args.horizon_days, args.min_train_days, args.first_origin,
                   args.last_origin, args.max_folds, train_gate)
    result["status"] = "available" if not args.synthetic_dry_run else "dry_run"
    result["truth"] = truth or {"source": "era5"}
    result["coverage"] = {"start": str(cube.dates[0]), "end": str(cube.dates[-1]), "missing_optional": cube.missing_optional}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(V.markdown(result), encoding="utf-8")
    print(f"Wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()

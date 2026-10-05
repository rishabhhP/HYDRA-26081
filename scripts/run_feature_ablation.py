"""Measure whether the v3.1 predictors help: identical rolling-origin validation with and without them.

    python scripts/run_feature_ablation.py --source-dir data/raw/era5_all_years
    python scripts/run_feature_ablation.py --source-dir data/raw/era5_all_years --max-folds 6 --step-days 60

Requires patches/hydra_rain_features_v3_1.patch (adds the predictors and the HYDRA_EXCLUDE_FEATURES switch).
Writes runtime/ablation_baseline.json, runtime/ablation_v3_1.json and reports/FEATURE_ABLATION.md.
Each run retrains HYDRA once per fold, so this takes as long as two full validations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ["moist_flux_conv", "moist_flux_conv_nb5", "moist_flux_conv_nb9", "cape_x_rh", "cape_x_rh_nb5"]
ROWS = [("state_scale", "MAE", ("continuous", "mae"), "lower"), ("state_scale", "RMSE", ("continuous", "rmse"), "lower"),
        ("state_scale", "Bias", ("continuous", "bias"), "closer to 0"), ("state_scale", "≥20 mm recall", ("heavy", "20", "recall"), "higher"),
        ("state_scale", "≥20 mm CSI", ("heavy", "20", "csi"), "higher"), ("state_scale", "Top-5 peak ratio", ("peaks", "top5_mean_ratio"), "higher"),
        ("state_scale", "80% coverage", ("interval", "coverage"), "closer to 0.8"), ("cell_scale", "Cell ≥64.5 recall", ("heavy", "64.5", "recall"), "higher"),
        ("cell_scale", "Cell ≥64.5 Brier", ("heavy", "64.5", "brier"), "lower"), ("cell_scale", "Cell ≥20 Brier", ("heavy", "20", "brier"), "lower")]


def run(tag: str, exclude: list[str], passthrough: list[str]) -> dict:
    out = ROOT / "runtime" / f"ablation_{tag}.json"
    env = {**os.environ, "HYDRA_EXCLUDE_FEATURES": ",".join(exclude)}
    cmd = [sys.executable, str(ROOT / "scripts" / "validate_hydra_rainfall.py"), "--output", str(out),
           "--report", str(ROOT / "reports" / f"ablation_{tag}.md"), *passthrough]
    print(f"\n=== {tag}: {' '.join(cmd)}  (excluding {len(exclude)} predictors)", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    return json.loads(out.read_text())


def pick(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--max-folds", type=int)
    p.add_argument("--step-days", type=int)
    p.add_argument("--horizon-days", type=int)
    a = p.parse_args()
    passthrough = ["--source-dir", a.source_dir]
    for flag in ("max_folds", "step_days", "horizon_days"):
        if getattr(a, flag):
            passthrough += [f"--{flag.replace('_', '-')}", str(getattr(a, flag))]
    base = run("baseline", NEW, passthrough)
    new = run("v3_1", [], passthrough)
    lines = ["# Feature ablation: v3 predictors vs v3 + convergence and CAPE × humidity", "",
             f"Same rolling origins for both runs ({base['design']['folds']} folds). Pooled out-of-sample results.", "",
             "| Scale | Metric | Without new predictors | With new predictors | Better is | Change |", "|---|---|---|---|---|---|"]
    for scale, label, path, better in ROWS:
        b, n = pick(base[scale]["pooled"], path), pick(new[scale]["pooled"], path)
        if b is None or n is None:
            continue
        lines.append(f"| {scale.split('_')[0]} | {label} | {b:.3f} | {n:.3f} | {better} | {n - b:+.3f} |")
    lines += ["", "Decision rule: keep the predictors only if MAE/RMSE do not worsen and at least one heavy-rain metric improves "
              "in most folds (see by_fold in the two JSON files), not just in the pooled number."]
    report = ROOT / "reports" / "FEATURE_ABLATION.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

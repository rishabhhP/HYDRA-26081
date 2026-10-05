"""Create a read-only diagnostic from the published HYDRA v3 replay artifacts.

This script does not retrain or modify the model. It inspects the chronological
2025 replay already used by the prototype and writes a compact JSON and Markdown
report with the main failure modes and improvement priorities.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
REPORTS = ROOT / "reports"


def top(rows, key, n=5, reverse=False):
    valid = [row for row in rows if row[key] is not None]
    return sorted(valid, key=lambda row: row[key], reverse=reverse)[:n]


def fmt_rows(rows, metric, suffix=""):
    return "; ".join(f"{r['state']} ({r[metric]:.3f}{suffix})" for r in rows)


def finite_mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def main():
    evaluation = json.loads((RUNTIME / "hydra_v3_evaluation.json").read_text())
    replay = json.loads((RUNTIME / "hydra_rolling_rainfall_replay.json").read_text())
    pooled = evaluation["pooled"]

    states = []
    for state, payload in evaluation["states"].items():
        states.append({
            "state": state,
            "mae": payload["continuous"]["mae"],
            "bias": payload["continuous"]["bias"],
            "persistence_mae_skill": payload["skill"]["persistence"]["mae_skill"],
            "climatology_mae_skill": payload["skill"]["climatology"]["mae_skill"],
            "heavy20_recall": payload["heavy"]["20"]["recall"],
            "heavy20_csi": payload["heavy"]["20"]["csi"],
            "coverage": payload["interval"]["coverage"],
            "above_interval": payload["interval"]["above"],
        })

    gate = evaluation["gate_dynamics_pooled"]
    expert_weights = dict(zip(gate["experts"], gate["mean_weight"]))
    replay_state_scale = replay["metrics"]["state_scale"]
    lead = []
    for lead_name in ("lead1", "lead2"):
        reports = [values[lead_name] for values in replay_state_scale.values()]
        lead.append({
            "lead": lead_name,
            "mae": finite_mean(r["continuous"]["mae"] for r in reports),
            "rmse": finite_mean(r["continuous"]["rmse"] for r in reports),
            "bias": finite_mean(r["continuous"]["bias"] for r in reports),
            "heavy20_recall": finite_mean(r["heavy"]["20"]["recall"] for r in reports),
            "interval_coverage": finite_mean(r["interval"]["coverage"] for r in reports),
        })

    worse_than_persistence = [r for r in states if r["persistence_mae_skill"] is not None and r["persistence_mae_skill"] <= 0]
    worse_than_climatology = [r for r in states if r["climatology_mae_skill"] is not None and r["climatology_mae_skill"] <= 0]
    diagnostic = {
        "model_version": evaluation["model_version"],
        "validation": "Chronological holdout: 2025-07-01 to 2025-12-31, 36 state means, 6,624 state-days.",
        "pooled": pooled,
        "lead_time": lead,
        "gate_mean_weights": expert_weights,
        "state_summary": {
            "states_worse_than_persistence_mae": len(worse_than_persistence),
            "states_worse_than_climatology_mae": len(worse_than_climatology),
            "worst_mae": top(states, "mae", reverse=True),
            "weakest_heavy20_recall": top(states, "heavy20_recall"),
            "largest_underprediction_bias": top(states, "bias"),
            "largest_overprediction_bias": top(states, "bias", reverse=True),
            "worst_upper_interval_misses": top(states, "above_interval", reverse=True),
        },
        "findings": [
            "The point forecast improves RMSE versus persistence, but has worse MAE. It is useful for average error variance, not a clear overall replacement for persistence.",
            "Heavy-rain detection is the central weakness: state-mean >=20 mm/day recall is low and extreme peaks are severely damped.",
            "The nominal 80% interval over-covers, but misses are mostly above the upper bound. This is an asymmetric upper-tail calibration failure.",
            "The gate is dynamic, but its mean allocation is concentrated in the monsoon expert while the learned GBM, hurdle, spatial, and upper-quantile experts receive small average weights.",
            "Only one year is available and the model is evaluated on one chronological six-month holdout. Reliability across years, storm types, and operational data revisions is not established.",
        ],
        "recommendations": [
            "Train a separate extreme-rain amount head in log1p space and a threshold-specific calibrator for >=10, >=20, and >=64.5 mm/day events; choose alert thresholds from precision-recall curves rather than a fixed 0.5 cutoff.",
            "Use upper-tail asymmetric conformal intervals or quantile calibration by state, season, lead, and regime so upper bounds widen when convective or monsoon-active signals are present.",
            "Make the gate optimize against out-of-fold skill relative to persistence and climatology, with a regularizer that prevents the monsoon expert from dominating when learned experts have better recent validation loss.",
            "Add predictors that describe storm initiation and transport: CAPE, boundary-layer humidity, wind components, moisture-flux convergence, pressure tendency, and spatial rainfall context. Keep only fields available at issue time.",
            "Use multiple years of ERA5 and rolling-origin folds across complete monsoon and post-monsoon seasons. Publish state-level reliability labels; do not present sparse >=64.5 mm/day metrics as trustworthy.",
            "In the UI, distinguish a rainfall amount forecast from an event-risk forecast and display a warning when state-specific heavy-rain recall or sample count is weak.",
        ],
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hydra_v3_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")

    p = pooled
    worst_mae = fmt_rows(diagnostic["state_summary"]["worst_mae"], "mae", " mm/day")
    weak_heavy = fmt_rows(diagnostic["state_summary"]["weakest_heavy20_recall"], "heavy20_recall")
    under = fmt_rows(diagnostic["state_summary"]["largest_underprediction_bias"], "bias", " mm/day")
    upper = fmt_rows(diagnostic["state_summary"]["worst_upper_interval_misses"], "above_interval")
    lead_rows = "\n".join(
        f"| +{1 if row['lead'] == 'lead1' else 2} day | {row['mae']:.2f} | {row['rmse']:.2f} | {row['bias']:.2f} | {row['heavy20_recall']:.3f} | {row['interval_coverage']:.3f} |"
        for row in lead
    )
    weight_rows = "\n".join(f"| {name} | {weight * 100:.1f}% |" for name, weight in sorted(expert_weights.items(), key=lambda item: item[1], reverse=True))
    markdown = f"""# HYDRA v3 Diagnostic

**Validation:** {diagnostic['validation']}

## Verdict

HYDRA is useful for ordinary state-average rainfall and rain occurrence, but it is **not accurate enough to rely on for intense-rain amount or heavy-rain alerts**. The model is modestly better than persistence on RMSE, yet worse on MAE. Its largest errors occur when the impact is highest: convective and heavy rainfall.

## Pooled results

| Metric | Result |
| --- | ---: |
| MAE | {p['continuous']['mae']:.2f} mm/day |
| RMSE | {p['continuous']['rmse']:.2f} mm/day |
| Bias | {p['continuous']['bias']:.2f} mm/day |
| Rain occurrence recall (>=1 mm) | {p['occurrence']['recall']:.1%} |
| Rain occurrence precision (>=1 mm) | {p['occurrence']['precision']:.1%} |
| Heavy-rain recall (>=10 mm) | {p['heavy']['10']['recall']:.1%} |
| Heavy-rain recall (>=20 mm) | {p['heavy']['20']['recall']:.1%} |
| Heavy-rain CSI (>=20 mm) | {p['heavy']['20']['csi']:.3f} |
| Nominal / actual interval coverage | {p['interval']['nominal']:.0%} / {p['interval']['coverage']:.1%} |

The peak case was {p['peaks']['max_observed']:.1f} mm/day on {p['peaks']['date_of_max']}; HYDRA predicted {p['peaks']['predicted_at_max']:.1f} mm/day, an error of {p['peaks']['error_at_max']:.1f} mm/day. Across the five largest observed values, the model predicted only {p['peaks']['top5_mean_ratio']:.1%} of the observed total.

## Lead-time degradation

| Lead | MAE | RMSE | Bias | >=20 mm recall | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
{lead_rows}

## Main shortcomings

1. **Extreme rainfall is damped.** The model has a -{abs(p['peaks']['error_at_max']):.1f} mm/day error on its largest event, with only {p['heavy']['20']['recall']:.1%} recall at the 20 mm/day state-average threshold.
2. **It favors low-error averages over damaging events.** Dry-day bias is +{p['by_intensity']['dry_lt1']['bias']:.2f} mm/day, while 20–64.5 mm/day bias is {p['by_intensity']['rather_heavy_20_64_5']['bias']:.2f} mm/day and >=64.5 mm/day bias is {p['by_intensity']['heavy_ge64_5']['bias']:.2f} mm/day.
3. **The uncertainty interval is asymmetric.** Coverage is wider than its 80% target ({p['interval']['coverage']:.1%}), but {p['interval']['above']:.1%} of outcomes lie above the upper bound and only {p['interval']['below']:.1%} lie below the lower bound.
4. **Persistence remains hard to beat.** HYDRA MAE skill against persistence is {p['skill']['persistence']['mae_skill']:.1%}; it is worse on MAE, although RMSE skill is {p['skill']['persistence']['rmse_skill']:.1%}. {len(worse_than_persistence)} of 36 states have non-positive MAE skill against persistence; {len(worse_than_climatology)} have non-positive MAE skill against climatology.
5. **The gate is insufficiently balanced.** The monsoon expert receives {expert_weights['monsoon']:.1%} average weight. The four learned specialists together receive only {sum(expert_weights[k] for k in ('gbm_era5','hurdle','spatial','upper_q')):.1%}. The gate moves over time, so it is not frozen, but its allocation needs skill-aware constraints.
6. **Evidence remains limited.** This is a single-year, state-average, July–December holdout. The 13 >=64.5 mm/day pooled cases are too few for operationally reliable extreme-event claims.

## State patterns to inspect

- Highest MAE: {worst_mae}
- Weakest >=20 mm/day recall: {weak_heavy}
- Largest underprediction bias: {under}
- Most upper-bound misses: {upper}

## Improvement priorities

1. Train a separate log-transformed extreme-rain amount head and threshold-specific probability calibrators for 10, 20, and 64.5 mm/day.
2. Use asymmetric, state/season/lead/regime-specific upper-tail interval calibration.
3. Train the gate against rolling out-of-fold skill relative to persistence and climatology; constrain expert concentration when it loses recent validation skill.
4. Add issue-time convective and moisture predictors: CAPE, boundary-layer humidity, wind components, moisture-flux convergence, pressure tendency, and spatial rain context.
5. Refit and verify on multiple full years with rolling-origin folds; publish state-specific reliability and sample counts in the product.
6. Present heavy-rain values as risk probabilities with calibrated alert tiers, separate from the rainfall-amount forecast.

## Interpretation limit

These metrics apply to **state-average daily rainfall**, not station rainfall or every grid-cell extreme. They show real strengths in ordinary rain detection and real weaknesses in high-impact rainfall prediction.
"""
    (REPORTS / "HYDRA_V3_DIAGNOSTIC.md").write_text(markdown)
    print("Wrote reports/hydra_v3_diagnostic.json and reports/HYDRA_V3_DIAGNOSTIC.md")


if __name__ == "__main__":
    main()

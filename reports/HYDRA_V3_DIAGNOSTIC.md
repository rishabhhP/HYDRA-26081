# HYDRA v3 Diagnostic

**Validation:** Chronological holdout: 2025-07-01 to 2025-12-31, 36 state means, 6,624 state-days.

## Verdict

HYDRA is useful for ordinary state-average rainfall and rain occurrence, but it is **not accurate enough to rely on for intense-rain amount or heavy-rain alerts**. The model is modestly better than persistence on RMSE, yet worse on MAE. Its largest errors occur when the impact is highest: convective and heavy rainfall.

## Pooled results

| Metric | Result |
| --- | ---: |
| MAE | 3.80 mm/day |
| RMSE | 7.47 mm/day |
| Bias | -0.82 mm/day |
| Rain occurrence recall (>=1 mm) | 77.2% |
| Rain occurrence precision (>=1 mm) | 96.5% |
| Heavy-rain recall (>=10 mm) | 47.5% |
| Heavy-rain recall (>=20 mm) | 24.4% |
| Heavy-rain CSI (>=20 mm) | 0.195 |
| Nominal / actual interval coverage | 80% / 89.9% |

The peak case was 161.9 mm/day on 2025-09-28; HYDRA predicted 3.8 mm/day, an error of -158.1 mm/day. Across the five largest observed values, the model predicted only 18.9% of the observed total.

## Lead-time degradation

| Lead | MAE | RMSE | Bias | >=20 mm recall | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1 day | 3.80 | 7.03 | -0.82 | 0.196 | 0.899 |
| +2 day | 4.51 | 8.33 | -1.34 | 0.070 | 0.882 |

## Main shortcomings

1. **Extreme rainfall is damped.** The model has a -158.1 mm/day error on its largest event, with only 24.4% recall at the 20 mm/day state-average threshold.
2. **It favors low-error averages over damaging events.** Dry-day bias is +0.94 mm/day, while 20–64.5 mm/day bias is -15.40 mm/day and >=64.5 mm/day bias is -53.95 mm/day.
3. **The uncertainty interval is asymmetric.** Coverage is wider than its 80% target (89.9%), but 9.7% of outcomes lie above the upper bound and only 0.4% lie below the lower bound.
4. **Persistence remains hard to beat.** HYDRA MAE skill against persistence is -2.7%; it is worse on MAE, although RMSE skill is 6.2%. 25 of 36 states have non-positive MAE skill against persistence; 1 have non-positive MAE skill against climatology.
5. **The gate is insufficiently balanced.** The monsoon expert receives 44.0% average weight. The four learned specialists together receive only 9.7%. The gate moves over time, so it is not frozen, but its allocation needs skill-aware constraints.
6. **Evidence remains limited.** This is a single-year, state-average, July–December holdout. The 13 >=64.5 mm/day pooled cases are too few for operationally reliable extreme-event claims.

## State patterns to inspect

- Highest MAE: Sikkim (7.218 mm/day); Goa (7.033 mm/day); Dadra and Nagar Haveli and Daman and Diu (5.463 mm/day); Kerala (5.278 mm/day); Andaman and Nicobar Islands (4.857 mm/day)
- Weakest >=20 mm/day recall: Andhra Pradesh (0.000); Karnataka (0.000); Lakshadweep (0.000); Rajasthan (0.000); Tamil Nadu (0.000)
- Largest underprediction bias: Goa (-4.508 mm/day); Andaman and Nicobar Islands (-2.461 mm/day); Chhattisgarh (-2.417 mm/day); Dadra and Nagar Haveli and Daman and Diu (-2.314 mm/day); Odisha (-2.188 mm/day)
- Most upper-bound misses: Goa (0.179); Andaman and Nicobar Islands (0.163); Puducherry (0.163); Andhra Pradesh (0.158); Telangana (0.158)

## Improvement priorities

1. Train a separate log-transformed extreme-rain amount head and threshold-specific probability calibrators for 10, 20, and 64.5 mm/day.
2. Use asymmetric, state/season/lead/regime-specific upper-tail interval calibration.
3. Train the gate against rolling out-of-fold skill relative to persistence and climatology; constrain expert concentration when it loses recent validation skill.
4. Add issue-time convective and moisture predictors: CAPE, boundary-layer humidity, wind components, moisture-flux convergence, pressure tendency, and spatial rain context.
5. Refit and verify on multiple full years with rolling-origin folds; publish state-specific reliability and sample counts in the product.
6. Present heavy-rain values as risk probabilities with calibrated alert tiers, separate from the rainfall-amount forecast.

## Interpretation limit

These metrics apply to **state-average daily rainfall**, not station rainfall or every grid-cell extreme. They show real strengths in ordinary rain detection and real weaknesses in high-impact rainfall prediction.

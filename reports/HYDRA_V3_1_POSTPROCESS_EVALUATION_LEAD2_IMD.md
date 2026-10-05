# HYDRA v3.1 post-processing: out-of-sample evaluation

Test window 2025-08-01 to 2025-12-31, 5202 state-days. Every number is out-of-sample: each month is predicted by models fitted only on earlier months (1-day embargo). July 2025 is training-only, so the comparison window is Aug-Dec 2025 for both raw HYDRA and the fixes.

## Amount forecast

| Model | MAE | RMSE | Bias | ≥10 recall | ≥20 recall | ≥20 CSI | Bias 20-64.5 | Bias ≥64.5 | Top-5 ratio | MAE skill vs persistence | RMSE skill vs persistence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw HYDRA v3 | 4.78 | 9.74 | 1.96 | 70.0% | 40.8% | 0.213 | -10.6 | -41.0 | 0.57 | -21.9% | -0.2% |
| persistence | 3.92 | 9.72 | 0.06 | 53.9% | 34.2% | 0.205 | -13.2 | -39.0 | 0.57 | 0.0% | 0.0% |
| equal-weight experts | 3.57 | 8.55 | -1.11 | 34.1% | 8.8% | 0.081 | -20.2 | -73.0 | 0.25 | 8.9% | 12.1% |
| skill gate | 3.54 | 8.49 | -0.91 | 36.7% | 11.6% | 0.102 | -19.0 | -65.1 | 0.37 | 9.5% | 12.7% |
| log1p extreme head | 3.81 | 8.45 | -0.31 | 48.2% | 17.9% | 0.145 | -18.0 | -66.6 | 0.33 | 2.7% | 13.1% |
| Poisson head | 3.98 | 8.54 | 0.26 | 56.4% | 18.5% | 0.144 | -16.8 | -67.9 | 0.29 | -1.6% | 12.2% |
| heavy/non-heavy mixture | 3.81 | 8.45 | -0.24 | 50.3% | 10.3% | 0.094 | -18.5 | -70.7 | 0.29 | 2.8% | 13.0% |
| gate + mixture | 3.60 | 8.30 | -0.58 | 43.9% | 11.3% | 0.101 | -18.8 | -67.9 | 0.33 | 8.1% | 14.6% |
| FINAL (history-selected) | 4.12 | 8.95 | 0.67 | 60.9% | 25.1% | 0.159 | -14.3 | -56.1 | 0.46 | -5.1% | 8.0% |

95% day-block bootstrap: final − raw MAE -0.88 to -0.46 mm/day; final − raw RMSE -1.50 to -0.14; final MAE skill vs persistence -10.5% to 0.3%.

States with non-positive MAE skill vs persistence: raw 33/36, final 25/36. Vs climatology: raw 24/36, final 17/36.

## 80% interval

| Interval | Coverage | Above upper | Below lower | Mean width |
|---|---|---|---|---|
| raw HYDRA v3 (symmetric conformal) | 87.1% | 6.6% | 6.2% | 11.6 mm |
| asymmetric split conformal | 86.1% | 6.6% | 7.4% | 11.1 mm |
| adaptive conformal (ACI) | 85.9% | 7.4% | 6.7% | 10.5 mm |

Adaptive (published) coverage 95% CI 83.9% to 87.6%.

Asymmetric coverage 95% CI 84.2% to 87.9%; above-upper CI 5.6% to 7.6%.

## Heavy-rain probabilities

| Event | Events | Source | Brier | Brier skill | AUC |
|---|---|---|---|---|---|
| state_10 | 809 | raw HYDRA head (pa_20) | 0.2450 | -0.862 | 0.878 |
| state_10 | 809 | FINAL calibrated | 0.0950 | 0.274 | 0.880 |
| state_20 | 319 | raw HYDRA head (pa_20) | 0.3080 | -4.349 | 0.865 |
| state_20 | 319 | FINAL calibrated | 0.0490 | 0.148 | 0.862 |
| state_64.5 | 22 | raw HYDRA head (pa_64.5) | 0.0380 | -7.948 | 0.918 |
| state_64.5 | 22 | FINAL calibrated | 0.0040 | 0.083 | 0.909 |
| local_64.5 | 707 | raw HYDRA head (pm_64.5) | 0.1280 | -0.090 | 0.858 |
| local_64.5 | 707 | FINAL calibrated | 0.0880 | 0.251 | 0.882 |

### Alert tiers (final calibrated probabilities)

| Event | Tier | P ≥ | Issued | Hit rate | Events captured |
|---|---|---|---|---|---|
| state_10 | watch | 0.2 | 1868 | 38.0% | 87.6% |
| state_10 | alert | 0.4 | 896 | 50.7% | 56.1% |
| state_10 | warning | 0.6 | 210 | 63.8% | 16.6% |
| state_20 | watch | 0.2 | 696 | 26.0% | 56.7% |
| state_20 | alert | 0.4 | 78 | 55.1% | 13.5% |
| state_20 | warning | 0.6 | 37 | 56.8% | 6.6% |
| state_64.5 | watch | 0.2 | 5 | 60.0% | 13.6% |
| state_64.5 | alert | 0.4 | 1 | 100.0% | 4.5% |
| state_64.5 | warning | 0.6 | 1 | 100.0% | 4.5% |
| local_64.5 | watch | 0.2 | 1693 | 36.6% | 87.6% |
| local_64.5 | alert | 0.4 | 825 | 46.8% | 54.6% |
| local_64.5 | warning | 0.6 | 191 | 57.1% | 15.4% |

## Gate allocation

| Expert | v3 trained gate | Skill-aware gate |
|---|---|---|
| climatology | 37.8% | 10.2% |
| persistence | 3.2% | 12.0% |
| recent3 | 12.1% | 9.7% |
| anom_persistence | 13.5% | 11.4% |
| gbm_era5 | 0.4% | 10.9% |
| hurdle | 19.0% | 10.8% |
| spatial | 0.4% | 11.1% |
| monsoon | 13.4% | 8.3% |
| upper_q | 0.3% | 8.6% |
| hydra blend (as one input) | n/a | 6.9% |

## Model choice per month (by earlier out-of-sample record only)

- 2025-08: amount `hydra`; probabilities state_10: `iso_state_10`, state_20: `iso_state_20`, state_64.5: `iso_state_64.5`, local_64.5: `iso_local_64.5`
- 2025-09: amount `ens_gate_log`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `p_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-10: amount `ens_gate_log`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `p_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-11: amount `ens_gate_log`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `p_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-12: amount `ens_gate_log`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `p_state_64.5`, local_64.5: `stack_local_64.5`

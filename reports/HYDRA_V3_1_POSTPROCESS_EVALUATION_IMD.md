# HYDRA v3.1 post-processing: out-of-sample evaluation

Test window 2025-08-01 to 2025-12-31, 5202 state-days. Every number is out-of-sample: each month is predicted by models fitted only on earlier months (1-day embargo). July 2025 is training-only, so the comparison window is Aug-Dec 2025 for both raw HYDRA and the fixes.

## Amount forecast

| Model | MAE | RMSE | Bias | ≥10 recall | ≥20 recall | ≥20 CSI | Bias 20-64.5 | Bias ≥64.5 | Top-5 ratio | MAE skill vs persistence | RMSE skill vs persistence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw HYDRA v3 | 2.31 | 5.18 | 1.96 | 97.7% | 93.7% | 0.667 | -0.0 | -9.8 | 0.81 | -18263.1% | -5672.8% |
| persistence | 0.01 | 0.09 | 0.00 | 99.8% | 99.4% | 0.994 | -0.0 | 0.0 | 1.00 | 0.0% | 0.0% |
| equal-weight experts | 2.42 | 6.00 | -0.88 | 63.3% | 24.1% | 0.241 | -14.1 | -55.3 | 0.41 | -19106.8% | -6591.0% |
| skill gate | 0.28 | 0.71 | -0.04 | 97.3% | 92.2% | 0.922 | -1.4 | -5.4 | 0.95 | -2113.3% | -694.3% |
| log1p extreme head | 0.23 | 2.55 | -0.04 | 99.3% | 97.2% | 0.960 | 0.5 | -18.2 | 0.59 | -1765.7% | -2738.3% |
| Poisson head | 0.27 | 2.34 | -0.02 | 99.3% | 96.2% | 0.933 | 0.5 | -16.4 | 0.64 | -2025.9% | -2503.5% |
| heavy/non-heavy mixture | 0.13 | 2.08 | -0.03 | 99.5% | 97.8% | 0.975 | 0.2 | -8.9 | 0.69 | -938.1% | -2212.4% |
| gate + mixture | 0.18 | 1.21 | -0.04 | 98.0% | 95.0% | 0.950 | -0.6 | -7.1 | 0.82 | -1306.9% | -1253.3% |
| FINAL (history-selected) | 1.10 | 3.50 | 0.84 | 98.1% | 95.3% | 0.774 | -0.1 | -7.7 | 0.85 | -8638.9% | -3794.8% |

95% day-block bootstrap: final − raw MAE -1.52 to -0.94 mm/day; final − raw RMSE -2.41 to -1.02; final MAE skill vs persistence -11047.0% to -6515.3%.

States with non-positive MAE skill vs persistence: raw 34/36, final 34/36. Vs climatology: raw 1/36, final 0/36.

## 80% interval

| Interval | Coverage | Above upper | Below lower | Mean width |
|---|---|---|---|---|
| raw HYDRA v3 (symmetric conformal) | 95.2% | 0.1% | 4.7% | 10.9 mm |
| asymmetric split conformal | 88.4% | 4.8% | 6.8% | 2.3 mm |
| adaptive conformal (ACI) | 86.2% | 7.3% | 6.5% | 2.2 mm |

Adaptive (published) coverage 95% CI 84.9% to 87.5%.

Asymmetric coverage 95% CI 87.0% to 89.7%; above-upper CI 4.1% to 5.4%.

## Heavy-rain probabilities

| Event | Events | Source | Brier | Brier skill | AUC |
|---|---|---|---|---|---|
| state_10 | 809 | raw HYDRA head (pa_20) | 0.2130 | -0.622 | 0.950 |
| state_10 | 809 | FINAL calibrated | 0.0310 | 0.764 | 0.988 |
| state_20 | 319 | raw HYDRA head (pa_20) | 0.2860 | -3.973 | 0.954 |
| state_20 | 319 | FINAL calibrated | 0.0170 | 0.698 | 0.987 |
| state_64.5 | 22 | raw HYDRA head (pa_64.5) | 0.0490 | -10.629 | 0.923 |
| state_64.5 | 22 | FINAL calibrated | 0.0030 | 0.255 | 0.943 |
| local_64.5 | 707 | raw HYDRA head (pm_64.5) | 0.1300 | -0.105 | 0.926 |
| local_64.5 | 707 | FINAL calibrated | 0.0440 | 0.629 | 0.973 |

### Alert tiers (final calibrated probabilities)

| Event | Tier | P ≥ | Issued | Hit rate | Events captured |
|---|---|---|---|---|---|
| state_10 | watch | 0.2 | 1208 | 66.3% | 99.0% |
| state_10 | alert | 0.4 | 861 | 83.6% | 89.0% |
| state_10 | warning | 0.6 | 738 | 88.8% | 81.0% |
| state_20 | watch | 0.2 | 468 | 59.8% | 87.8% |
| state_20 | alert | 0.4 | 319 | 78.7% | 78.7% |
| state_20 | warning | 0.6 | 218 | 96.3% | 65.8% |
| state_64.5 | watch | 0.2 | 35 | 28.6% | 45.5% |
| state_64.5 | alert | 0.4 | 6 | 100.0% | 27.3% |
| state_64.5 | warning | 0.6 | 3 | 100.0% | 13.6% |
| local_64.5 | watch | 0.2 | 1133 | 57.2% | 91.7% |
| local_64.5 | alert | 0.4 | 842 | 70.5% | 84.0% |
| local_64.5 | warning | 0.6 | 605 | 83.1% | 71.1% |

## Gate allocation

| Expert | v3 trained gate | Skill-aware gate |
|---|---|---|
| climatology | 28.8% | 1.0% |
| persistence | 2.9% | 50.0% |
| recent3 | 11.9% | 1.0% |
| anom_persistence | 14.7% | 42.0% |
| gbm_era5 | 0.4% | 1.0% |
| hurdle | 26.4% | 1.0% |
| spatial | 0.4% | 1.0% |
| monsoon | 14.3% | 1.0% |
| upper_q | 0.3% | 1.0% |
| hydra blend (as one input) | n/a | 1.0% |

## Model choice per month (by earlier out-of-sample record only)

- 2025-08: amount `hydra`; probabilities state_10: `iso_state_10`, state_20: `iso_state_20`, state_64.5: `iso_state_64.5`, local_64.5: `iso_local_64.5`
- 2025-09: amount `ens_gate_mixture`; probabilities state_10: `p_state_10`, state_20: `p_state_20`, state_64.5: `p_state_64.5`, local_64.5: `p_local_64.5`
- 2025-10: amount `gate`; probabilities state_10: `p_state_10`, state_20: `p_state_20`, state_64.5: `p_state_64.5`, local_64.5: `p_local_64.5`
- 2025-11: amount `gate`; probabilities state_10: `p_state_10`, state_20: `p_state_20`, state_64.5: `p_state_64.5`, local_64.5: `p_local_64.5`
- 2025-12: amount `gate`; probabilities state_10: `p_state_10`, state_20: `p_state_20`, state_64.5: `p_state_64.5`, local_64.5: `p_local_64.5`

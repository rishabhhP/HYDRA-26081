# HYDRA v3.1 post-processing: out-of-sample evaluation

Test window 2025-08-01 to 2025-12-31, 5508 state-days. Every number is out-of-sample: each month is predicted by models fitted only on earlier months (1-day embargo). July 2025 is training-only, so the comparison window is Aug-Dec 2025 for both raw HYDRA and the fixes.

## Amount forecast

| Model | MAE | RMSE | Bias | ≥10 recall | ≥20 recall | ≥20 CSI | Bias 20-64.5 | Bias ≥64.5 | Top-5 ratio | MAE skill vs persistence | RMSE skill vs persistence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw HYDRA v3 | 3.33 | 6.94 | -0.54 | 44.8% | 24.8% | 0.195 | -15.8 | -64.7 | 0.16 | -4.9% | 7.3% |
| persistence | 3.18 | 7.48 | 0.05 | 61.4% | 43.6% | 0.277 | -9.7 | -52.2 | 0.29 | 0.0% | 0.0% |
| equal-weight experts | 3.03 | 6.46 | -0.59 | 51.0% | 16.9% | 0.152 | -16.6 | -66.6 | 0.18 | 4.6% | 13.6% |
| skill gate | 2.97 | 6.56 | -0.67 | 50.0% | 21.0% | 0.179 | -15.9 | -62.0 | 0.20 | 6.6% | 12.3% |
| log1p extreme head | 3.26 | 6.67 | 0.10 | 63.9% | 21.7% | 0.166 | -14.8 | -65.2 | 0.19 | -2.7% | 10.8% |
| Poisson head | 3.37 | 6.72 | 0.48 | 68.5% | 24.5% | 0.187 | -14.1 | -63.9 | 0.20 | -6.2% | 10.1% |
| heavy/non-heavy mixture | 3.28 | 6.62 | 0.18 | 66.3% | 17.8% | 0.151 | -15.2 | -65.9 | 0.18 | -3.3% | 11.5% |
| gate + mixture | 3.03 | 6.42 | -0.24 | 59.8% | 18.5% | 0.161 | -15.5 | -63.9 | 0.19 | 4.7% | 14.2% |
| FINAL (history-selected) | 3.18 | 6.65 | -0.43 | 51.7% | 20.7% | 0.176 | -16.1 | -63.8 | 0.17 | -0.0% | 11.1% |

95% day-block bootstrap: final − raw MAE -0.20 to -0.11 mm/day; final − raw RMSE -0.39 to -0.18; final MAE skill vs persistence -3.8% to 3.4%.

States with non-positive MAE skill vs persistence: raw 26/36, final 20/36. Vs climatology: raw 3/36, final 1/36.

## 80% interval

| Interval | Coverage | Above upper | Below lower | Mean width |
|---|---|---|---|---|
| raw HYDRA v3 (symmetric conformal) | 91.2% | 8.3% | 0.5% | 10.1 mm |
| asymmetric split conformal | 84.0% | 7.4% | 8.7% | 9.5 mm |

Asymmetric coverage 95% CI 82.4% to 85.5%; above-upper CI 6.3% to 8.4%.

## Heavy-rain probabilities

| Event | Events | Source | Brier | Brier skill | AUC |
|---|---|---|---|---|---|
| state_10 | 928 | raw HYDRA head (pa_20) | 0.2190 | -0.560 | 0.895 |
| state_10 | 928 | FINAL calibrated | 0.0920 | 0.343 | 0.901 |
| state_20 | 314 | raw HYDRA head (pa_20) | 0.2890 | -4.376 | 0.888 |
| state_20 | 314 | FINAL calibrated | 0.0430 | 0.206 | 0.893 |
| state_64.5 | 8 | raw HYDRA head (pa_64.5) | 0.0280 | -18.067 | 0.920 |
| state_64.5 | 8 | FINAL calibrated | 0.0010 | 0.017 | 0.938 |
| local_64.5 | 501 | raw HYDRA head (pm_64.5) | 0.1090 | -0.312 | 0.902 |
| local_64.5 | 501 | FINAL calibrated | 0.0610 | 0.257 | 0.904 |

### Alert tiers (final calibrated probabilities)

| Event | Tier | P ≥ | Issued | Hit rate | Events captured |
|---|---|---|---|---|---|
| state_10 | watch | 0.2 | 1915 | 43.2% | 89.2% |
| state_10 | alert | 0.4 | 1104 | 56.2% | 66.8% |
| state_10 | warning | 0.6 | 733 | 61.1% | 48.3% |
| state_20 | watch | 0.2 | 739 | 27.9% | 65.6% |
| state_20 | alert | 0.4 | 149 | 53.0% | 25.2% |
| state_20 | warning | 0.6 | 59 | 62.7% | 11.8% |
| state_64.5 | watch | 0.2 | 5 | 20.0% | 12.5% |
| state_64.5 | alert | 0.4 | 0 | n/a | 0.0% |
| state_64.5 | warning | 0.6 | 0 | n/a | 0.0% |
| local_64.5 | watch | 0.2 | 979 | 37.0% | 72.3% |
| local_64.5 | alert | 0.4 | 517 | 48.5% | 50.1% |
| local_64.5 | warning | 0.6 | 130 | 53.8% | 14.0% |

## Gate allocation

| Expert | v3 trained gate | Skill-aware gate |
|---|---|---|
| climatology | 9.6% | 5.4% |
| persistence | 14.3% | 12.8% |
| recent3 | 12.5% | 6.9% |
| anom_persistence | 12.6% | 12.3% |
| gbm_era5 | 2.1% | 12.6% |
| hurdle | 2.5% | 13.3% |
| spatial | 3.8% | 14.4% |
| monsoon | 40.0% | 5.6% |
| upper_q | 2.5% | 5.2% |
| hydra blend (as one input) | n/a | 11.4% |

## Model choice per month (by earlier out-of-sample record only)

- 2025-08: amount `hydra`; probabilities state_10: `iso_state_10`, state_20: `iso_state_20`, state_64.5: `iso_state_64.5`, local_64.5: `iso_local_64.5`
- 2025-09: amount `ens_gate_mixture`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `stack_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-10: amount `equal_avg`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `stack_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-11: amount `equal_avg`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `stack_state_64.5`, local_64.5: `stack_local_64.5`
- 2025-12: amount `equal_avg`; probabilities state_10: `stack_state_10`, state_20: `stack_state_20`, state_64.5: `stack_state_64.5`, local_64.5: `stack_local_64.5`

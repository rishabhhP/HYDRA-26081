# ERA5 vs IMD: is the training target damping the peaks?

6,256 state-days across 34 states. Correlation 0.56; mean ERA5 − IMD +0.19 mm/day.

| IMD bin (mm/day) | Days | IMD mean | ERA5 mean | HYDRA mean | ERA5 / IMD | HYDRA / IMD |
|---|---|---|---|---|---|---|
| 0-1 | 3044 | 0.1 | 1.1 | 1.4 | 11.49 | 15.49 |
| 1-10 | 2003 | 4.6 | 7.7 | 6.3 | 1.66 | 1.35 |
| 10-20 | 730 | 14.1 | 13.2 | 11.0 | 0.94 | 0.78 |
| 20-64.5 | 446 | 30.3 | 18.5 | 15.7 | 0.61 | 0.52 |
| >=64.5 | 33 | 100.3 | 39.5 | 32.2 | 0.39 | 0.32 |

20 wettest IMD state-days: ERA5 captured 32% of the rain, HYDRA 28%.
Events ≥20 mm: IMD 479, ERA5 463; ≥64.5 mm: IMD 33, ERA5 13.
On 20-64.5 mm days, about 81% of HYDRA's shortfall against IMD is already present in ERA5 itself.

HYDRA scored against each truth:

| Truth | MAE | RMSE | Bias | ≥20 recall | ≥20 CSI | Top-5 ratio |
|---|---|---|---|---|---|---|
| IMD | 3.80 | 8.57 | -0.59 | 30.3% | 0.256 | 0.22 |
| ERA5 | 3.80 | 7.52 | -0.78 | 25.1% | 0.200 | 0.19 |

Reading: if ERA5 / IMD is well below 1 on heavy bins, retrain and verify HYDRA on IMD truth. If it is near 1, the damping is in the model.

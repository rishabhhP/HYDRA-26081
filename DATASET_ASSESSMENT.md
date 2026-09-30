# Supplied dataset assessment

## crop-yield-analysis

Source: https://github.com/eric157/crop-yield-analysis/tree/main/rainfall_data

Repository inventory contains annual rainfall NetCDF files for 2014–2023. Inspected the actual 2014 file and all three root NetCDF files using xarray/netCDF4; the other annual files have not yet been validated.

| File | Verified contents | Use in HYDRA |
| --- | --- | --- |
| RF25_ind2014_rfp25.nc | 365 daily timestamps in 2014; RAINFALL in mm; missing value -999; 129 × 135 grid; latitude 6.5–38.5 N, longitude 66.5–100 E, 0.25° spacing | Historical rainfall observations; requires explicit time-window definition and spatial alignment before verification |
| data_stream-oper_stepType-accum.nc | 132 monthly timestamps, Jan 2014–Dec 2024; 5 × 5 grid at 22–23 N, 75–76 E; tp, cp, lsp in metres | Local historical precipitation; not the nwpblend input file despite matching filename |
| data_stream-oper_stepType-instant.nc | Same monthly timestamps and local grid; t2m in K and categorical slt | Local historical temperature and soil type |
| data_stream-oper_stepType-max.nc | Same monthly timestamps and local grid; mx2t, mn2t in K | Local temperature analysis; aggregation semantics need checking before use |

The README's description of the accumulated file as temperature data is inconsistent with its actual precipitation variables. File metadata is the integration source of truth. Do not infer daily accumulation windows merely from midnight timestamps.

## User ZIP

`5fc2aabe94917a75e88241b0ce8bf6ae-001.zip` contains one 7,590,579,152-byte GRIB1 file. A full streaming header scan found 1,240 records and reference dates from 2025-02-28 through 2025-03-31. Table 128 parameter IDs: 8, 39, 40, 41, 139, 169, 170, 205, 228, 235, each with 124 records. The first field has a global 0.1° grid. Reference dates must not be confused with valid times for accumulated fields. This ZIP is not established as originating from the crop-yield repository; its format, dates, and fields differ from that repository's files.

## Remaining requirements for the supplied trained nwpblend forecaster

- Full 14,157-cell grid history (7–37 N, 68–97 E, 0.25°), at least 35 consecutive days through the issue day, with all required atmospheric variables.
- Saved train-only `artifacts/blocked/climatology_by_doy.npz`, or the original 2025 ERA5 inputs needed to regenerate it without retraining.
- Crop-yield rainfall can be integrated as a separate historical observation source. It cannot replace the multivariable ERA5 history or the fitted training statistics.
- No model weights, feature order, normalization, or scientific source code have been changed.

Metadata inspection utility: `scripts/inspect_netcdf.py`.

## 2025 ERA5 Drive import (September 2026)

The shared Drive folder is being staged under `data/raw/era5_2025_full_source/` and is checked by
`scripts/preflight_hydra_era5_2025.py`. Its current machine-readable result is
`data/processed/hydra_era5_2025_import_preflight.json`.

The January, February, March, April, July, August, September, October, November, and December archives
each contain the ten selected daily-mean fields: 2 m temperature and dew point, 10 m u/v wind, mean
sea-level pressure, total precipitation, radiation, cloud cover, CAPE, and boundary-layer height. The
best available May archive is missing CAPE and boundary-layer height. June is also missing those two
fields and contains neutral v-wind (`v10n`) rather than the required standard 10 m v-wind (`v10`).

Even a complete set of those ten *daily-mean* fields is not sufficient to publish a fresh forecast with
the saved HYDRA checkpoint. Its feature contract additionally needs daily maximum temperature and peak
wind to form the heatwave/high-wind state and labels, plus the original train-only
`climatology_by_doy.npz` artifact. The importer records these gaps and never substitutes daily means or
live-provider values for them.

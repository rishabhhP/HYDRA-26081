"""Parallel, resumable ERA5 download for HYDRA v3 training (CDS API).

    pip install cdsapi            # and put your key in ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api)
    python scripts/download_era5_parallel.py --years 2020-2024 --dry-run
    python scripts/download_era5_parallel.py --years 2020-2024 --workers 4
    python scripts/download_era5_parallel.py --years 2020-2024 --mode hourly --workers 4   # if daily statistics queue slowly

Why it is faster than one big request: every (variable, year) or (variable, year, month) is its own request,
they are all queued at once and run side by side, finished pieces are never re-downloaded, and failures retry.

What can and cannot be trimmed (checked against hydra_rain/grid.py):
  - all 10 REQUIRED variables are mandatory; total_column_water_vapour is added because the v3.1 convergence
    predictor needs it. Wind gust is optional (--with-gust).
  - days must be consecutive (the loader rejects gaps), so trim YEARS, not months.

Output: <out>/<year>/<variable>_daily-mean_<year>[_<MM>].nc (or .zip), which hydra_rain.grid.load_era5 reads directly:
    python scripts/build_hydra_rolling_rainfall_replay.py --source-dir <out>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

AREA = [37, 68, 7, 97]                     # N, W, S, E: the HYDRA India grid (121 x 117 at 0.25 degree)
REQUIRED = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "total_precipitation", "surface_solar_radiation_downwards", "total_cloud_cover",
            "convective_available_potential_energy", "boundary_layer_height"]
OPTIONAL = ["total_column_water_vapour"]
GUST = ["instantaneous_10m_wind_gust"]
ACCUMULATED = {"total_precipitation", "surface_solar_radiation_downwards"}
DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
HOURLY_DATASET = "reanalysis-era5-single-levels"


def years_arg(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in text.split(",")]


def plan(years, variables, chunk: str, mode: str, out: Path) -> list[dict]:
    jobs = []
    for year in years:
        for var in variables:
            months = [None] if chunk == "year" else list(range(1, 13))
            for month in months:
                stem = f"{var}_daily-mean_{year}" + (f"_{month:02d}" if month else "")
                req = {"product_type": "reanalysis", "variable": [var], "year": str(year),
                       "month": [f"{m:02d}" for m in ([month] if month else range(1, 13))],
                       "day": [f"{d:02d}" for d in range(1, 32)], "area": AREA}
                if mode == "daily":
                    req.update({"daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": "1_hourly"})
                    dataset = DAILY_DATASET
                else:
                    hours = range(24) if var in ACCUMULATED else (0, 6, 12, 18)
                    req.update({"time": [f"{h:02d}:00" for h in hours], "data_format": "netcdf", "download_format": "unarchived"})
                    dataset = HOURLY_DATASET
                jobs.append({"dataset": dataset, "request": req, "target": out / str(year) / stem, "var": var, "mode": mode})
    return jobs


def _finalise(raw: Path, target_stem: Path, mode: str) -> Path:
    head = raw.read_bytes()[:4]
    suffix = ".zip" if head[:2] == b"PK" else ".nc"
    if mode == "daily":
        final = target_stem.with_suffix(suffix)
        raw.replace(final)
        return final
    # hourly -> daily mean, same convention as the daily-statistics product (mean of hourly values)
    import xarray as xr
    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(raw) as z:
            member = next(n for n in z.namelist() if n.endswith(".nc"))
            z.extract(member, raw.parent)
            raw.unlink()
            raw = raw.parent / member
    with xr.open_dataset(raw) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        daily = ds.resample({tname: "1D"}).mean().load()
    final = target_stem.with_suffix(".nc")
    daily.to_netcdf(final)
    raw.unlink()
    return final


def run_job(job: dict, retries: int) -> tuple[str, str]:
    import cdsapi
    stem = job["target"]
    for existing in (stem.with_suffix(".nc"), stem.with_suffix(".zip")):
        if existing.exists() and existing.stat().st_size > 10_000:
            return stem.name, "already present"
    stem.parent.mkdir(parents=True, exist_ok=True)
    raw = stem.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve(job["dataset"], job["request"], str(raw))
            final = _finalise(raw, stem, job["mode"])
            return stem.name, f"ok -> {final.name}"
        except Exception as exc:
            msg = str(exc)
            if "too large" in msg.lower() or "cost limits" in msg.lower():
                return stem.name, "REJECTED: request too large; rerun with --chunk month"
            if attempt == retries:
                return stem.name, f"FAILED: {msg[:200]}"
            time.sleep(min(600, 30 * 2 ** attempt + random.uniform(0, 15)))
    return stem.name, "FAILED"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=years_arg, required=True, help="e.g. 2020-2024 (2025 you already have)")
    p.add_argument("--out", type=Path, default=Path("data/raw/era5_multi_year"))
    p.add_argument("--mode", choices=["daily", "hourly"], default="daily",
                   help="daily = CDS daily-statistics product; hourly = raw hours averaged locally (needs xarray)")
    p.add_argument("--chunk", choices=["year", "month"], default="year", help="request size; use month if CDS rejects a year")
    p.add_argument("--workers", type=int, default=4, help="parallel requests (CDS also limits per-user concurrency)")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--with-gust", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    variables = REQUIRED + OPTIONAL + (GUST if a.with_gust else [])
    jobs = plan(a.years, variables, a.chunk, a.mode, a.out)
    size_mb = len(a.years) * len(variables) * 365 * 121 * 117 * 4 / 1e6
    print(f"{len(jobs)} requests for {len(a.years)} year(s) x {len(variables)} variables, about {size_mb:,.0f} MB of daily data, "
          f"{a.workers} in parallel -> {a.out}")
    if a.dry_run:
        for j in jobs[:3]:
            print(json.dumps({"dataset": j["dataset"], "target": str(j["target"]), **j["request"]})[:400])
        return
    log_path = a.out / "download_log.json"
    a.out.mkdir(parents=True, exist_ok=True)
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_job, j, a.retries): j for j in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            name, status = fut.result()
            log[name] = {"status": status, "at": datetime.now().isoformat(timespec="seconds")}
            log_path.write_text(json.dumps(log, indent=1))
            print(f"[{done}/{len(jobs)}] {name}: {status}", flush=True)
    failed = [k for k, v in log.items() if v["status"].startswith(("FAILED", "REJECTED"))]
    print("all done" if not failed else f"{len(failed)} request(s) need a rerun (same command resumes): {failed[:5]}")


if __name__ == "__main__":
    main()

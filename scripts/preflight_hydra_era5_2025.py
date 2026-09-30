"""Verify CDS ERA5 daily-statistics archives before they are offered to HYDRA.

The saved HYDRA blend is deliberately strict: it accepts only daily history with
the exact feature schema on the trained 0.25-degree India grid.  This command is
read-only with respect to source archives.  It inventories the ZIP files and,
when requested, opens one NetCDF member per month to verify dates and grid.

It does *not* fabricate daily maximum temperature, peak wind, extreme-event
flags, or the train-only climatology artifact.  A daily-mean CDS product cannot
recover those inputs faithfully, so it is never published as a HYDRA forecast.

Example
-------
python scripts/preflight_hydra_era5_2025.py --verify-netcdf
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "era5_2025_full_source"
DEFAULT_REPORT = ROOT / "data" / "processed" / "hydra_era5_2025_import_preflight.json"

# These are the daily-mean files selected in the shared 2025 archives.
SOURCE_FIELDS = {
    "2m_temperature_0_daily-mean.nc": "t2m",
    "2m_dewpoint_temperature_0_daily-mean.nc": "d2m",
    "10m_u_component_of_wind_stream-oper_daily-mean.nc": "u10",
    "10m_v_component_of_wind_0_daily-mean.nc": "v10",
    "mean_sea_level_pressure_0_daily-mean.nc": "msl",
    "total_precipitation_0_daily-mean.nc": "tp",
    "surface_solar_radiation_downwards_0_daily-mean.nc": "ssrd",
    "total_cloud_cover_0_daily-mean.nc": "tcc",
    "convective_available_potential_energy_0_daily-mean.nc": "cape",
    "boundary_layer_height_0_daily-mean.nc": "blh",
}

# The trained feature builder uses these daily state values.  The static grid
# supplies the land/ocean mask and elevation proxy.
AVAILABLE_STATE_FIELDS = [
    "tp_mm", "t2m_C_mean", "wind_speed_mean", "msl_hPa_mean", "msl_hPa_min",
    "tcc_mean", "cape_max", "rh_pct_mean", "blh_mean", "ssrd_MJ",
]
UNRECOVERABLE_DAILY_PEAK_FIELDS = [
    "t2m_C_max", "wind_speed_max", "heatwave_day_flag", "high_wind_day_flag",
]


@dataclass
class ArchiveCheck:
    archive: str
    month: int | None
    zip_ok: bool
    source_fields: list[str]
    missing_source_fields: list[str]
    dates: list[str] | None = None
    grid: dict | None = None
    error: str | None = None


def month_from_name(path: Path) -> int | None:
    key = path.name.casefold()
    aliases = {
        "jan": 1, "feb": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "sep": 9, "october": 10, "nov": 11, "dec": 12,
    }
    return next((value for token, value in aliases.items() if token in key), None)


def inspect_archive(path: Path, verify_netcdf: bool) -> ArchiveCheck:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            names = {Path(member.filename).name for member in archive.infolist() if not member.is_dir()}
            fields = sorted(SOURCE_FIELDS[name] for name in names if name in SOURCE_FIELDS)
            missing = sorted(set(SOURCE_FIELDS.values()) - set(fields))
            result = ArchiveCheck(
                archive=path.name,
                month=month_from_name(path),
                zip_ok=bad_member is None,
                source_fields=fields,
                missing_source_fields=missing,
                error=f"Corrupt ZIP member: {bad_member}" if bad_member else None,
            )
            if verify_netcdf and result.error is None:
                member = "2m_temperature_0_daily-mean.nc"
                if member not in names:
                    result.error = f"Missing {member}; cannot verify dates and grid"
                else:
                    with tempfile.TemporaryDirectory(prefix="hydra-era5-") as temporary:
                        sample = Path(temporary) / member
                        with archive.open(member) as source, sample.open("wb") as destination:
                            while chunk := source.read(1024 * 1024):
                                destination.write(chunk)
                        import pandas as pd
                        import xarray as xr
                        with xr.open_dataset(sample) as dataset:
                            if "valid_time" not in dataset.coords or "latitude" not in dataset.coords or "longitude" not in dataset.coords:
                                result.error = "NetCDF must contain valid_time, latitude, and longitude coordinates"
                            else:
                                dates = pd.DatetimeIndex(dataset.valid_time.values)
                                result.dates = [str(d.date()) for d in dates]
                                result.grid = {
                                    "latitude_count": int(dataset.latitude.size),
                                    "longitude_count": int(dataset.longitude.size),
                                    "latitude_first": float(dataset.latitude.values[0]),
                                    "latitude_last": float(dataset.latitude.values[-1]),
                                    "longitude_first": float(dataset.longitude.values[0]),
                                    "longitude_last": float(dataset.longitude.values[-1]),
                                }
            return result
    except (OSError, zipfile.BadZipFile) as exc:
        return ArchiveCheck(path.name, month_from_name(path), False, [], sorted(SOURCE_FIELDS.values()), error=str(exc))


def summary(checks: list[ArchiveCheck]) -> dict:
    valid_checks = [check for check in checks if check.month is not None and check.zip_ok]
    valid_zip_months = sorted({check.month for check in valid_checks})
    # A folder can retain older or refresh copies.  Select the archive with the
    # most compatible fields for each calendar month, while preserving every
    # archive-level result above for audit.
    selected: dict[int, ArchiveCheck] = {}
    for check in valid_checks:
        current = selected.get(check.month)
        if current is None or len(check.source_fields) > len(current.source_fields) or (
            len(check.source_fields) == len(current.source_fields) and check.archive > current.archive
        ):
            selected[check.month] = check
    months = sorted(month for month, check in selected.items() if not check.missing_source_fields)
    expected_months = list(range(1, 13))
    missing_months = sorted(set(expected_months) - set(months))
    archive_fields = set.intersection(*(set(check.source_fields) for check in selected.values())) if selected else set()
    source_complete = archive_fields == set(SOURCE_FIELDS.values()) and not missing_months
    return {
        "archives_found": len(checks),
        "months_with_valid_zip": valid_zip_months,
        "months_found": months,
        "missing_months": missing_months,
        "selected_archive_by_month": {str(month): selected[month].archive for month in sorted(selected)},
        "missing_source_fields_by_month": {
            str(month): selected[month].missing_source_fields
            for month in sorted(selected) if selected[month].missing_source_fields
        },
        "source_fields_common_to_every_valid_archive": sorted(archive_fields),
        "source_archive_set_complete": source_complete,
        "saved_hydra_inference_ready": False,
        "available_daily_state_fields": AVAILABLE_STATE_FIELDS if source_complete else [],
        "blocking_inputs_for_saved_hydra_model": [
            *([] if source_complete else ["complete set of 12 monthly archives with all 10 selected ERA5 daily-mean fields"]),
            *UNRECOVERABLE_DAILY_PEAK_FIELDS,
            "climatology_by_doy.npz (train-only artifact from the original HYDRA training run)",
        ],
        "reason": (
            "The shared archives are daily means. They can support observed mean-state analysis, "
            "but they cannot reconstruct daily maximum temperature or peak wind used by the saved "
            "HYDRA feature contract. They are therefore not published as trained-model forecasts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-netcdf", action="store_true", help="Read each month's temperature file to verify calendar coverage and grid")
    args = parser.parse_args()

    archives = sorted(path for path in args.source_dir.glob("*2025*.zip*") if path.is_file())
    checks = [inspect_archive(path, args.verify_netcdf) for path in archives]
    report = {"checks": [asdict(check) for check in checks], "summary": summary(checks)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

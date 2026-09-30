"""Download the exact 2025 ERA5 inputs required by nwpblend ingestion.

Usage (after accepting the CDS licence and configuring ``~/.cdsapirc``):
    .venv\\Scripts\\python.exe scripts/download_era5_2025.py

The downloads are intentionally split because nwpblend's existing ingestion
expects an accumulation file (precipitation/radiation) and an instantaneous
state file. No daily aggregate product is used: the feature builder requires
the 00/06/12/18 UTC grid values.
"""
from __future__ import annotations

from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "nwpblend" / "c73e70ccac6586b08e6c16143c48cab5"
DATASET = "reanalysis-era5-single-levels"

COMMON = {
    "product_type": ["reanalysis"],
    "year": ["2025"],
    "month": [f"{month:02d}" for month in range(1, 13)],
    "day": [f"{day:02d}" for day in range(1, 32)],
    "time": ["00:00", "06:00", "12:00", "18:00"],
    # North, West, South, East: matches HYDRA's 7-37 N / 68-97 E grid.
    "area": [37, 68, 7, 97],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

ACCUMULATION_VARIABLES = [
    "total_precipitation",
    "surface_solar_radiation_downwards",
]

INSTANT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_cloud_cover",
    "convective_available_potential_energy",
    "boundary_layer_height",
    "sea_surface_temperature",
]


def retrieve(client: cdsapi.Client, variables: list[str], output: Path) -> None:
    if output.exists():
        print(f"Keeping existing download: {output}")
        return
    request = {**COMMON, "variable": variables}
    print(f"Requesting {output.name}; this may stay queued at CDS before downloading.")
    client.retrieve(DATASET, request).download(str(output))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    retrieve(client, ACCUMULATION_VARIABLES, OUTPUT / "data_stream-oper_stepType-accum.nc")
    retrieve(client, INSTANT_VARIABLES, OUTPUT / "data_stream-oper_stepType-instant.nc")


if __name__ == "__main__":
    main()

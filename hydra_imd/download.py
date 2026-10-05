"""Download IMD 0.25 degree daily gridded rainfall from IMD Pune via imdlib.

    pip install imdlib
    python -m hydra_imd.download --years 2015-2024 --dir data/raw/imd
    python -m hydra_imd.download --realtime 2025-07-01:2025-12-31 --dir data/raw/imd

Archive (final, quality-controlled) data come as one file per year. Recent months may only exist in IMD's
real-time gridded product, which imdlib fetches day by day. Source: https://imdpune.gov.in/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def years_arg(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in text.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=years_arg, help="archive years, e.g. 2015-2024 or 2019,2020")
    p.add_argument("--realtime", help="real-time range START:END (YYYY-MM-DD:YYYY-MM-DD)")
    p.add_argument("--dir", type=Path, default=Path("data/raw/imd"))
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()
    try:
        import imdlib as imd
    except ImportError:
        raise SystemExit("Install imdlib first:  pip install imdlib")
    a.dir.mkdir(parents=True, exist_ok=True)
    for year in a.years or []:
        target = a.dir / "rain" / f"{year}.grd"
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"{year}: already downloaded")
            continue
        for attempt in range(1, a.retries + 1):
            try:
                imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(a.dir))
                print(f"{year}: done")
                break
            except Exception as exc:  # IMD server is occasionally slow or down
                print(f"{year}: attempt {attempt} failed ({exc}); retrying in {20 * attempt}s")
                time.sleep(20 * attempt)
    if a.realtime:
        start, end = a.realtime.split(":")
        imd.get_real_data("rain", start, end, file_dir=str(a.dir))
        print(f"real-time {start} to {end}: done")


if __name__ == "__main__":
    main()

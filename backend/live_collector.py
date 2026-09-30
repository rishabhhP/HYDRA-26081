"""Persist daily live-layer samples on a fixed schedule."""
from __future__ import annotations

import argparse
import time

from . import live_weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval-minutes', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_minutes < 5:
        parser.error('interval must be at least 5 minutes')
    while True:
        report = live_weather.station_reports()
        print(f"{report['status']}: {len(report.get('stations', []))} live locations", flush=True)
        if args.once:
            return
        time.sleep(args.interval_minutes * 60)


if __name__ == '__main__':
    main()

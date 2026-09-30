"""Build a long-term HYDRA summary from repeated unchanged-model forecasts.

This worker is intentionally offline. It never runs in a WeatherGPT request:
an annual India grid requires one forecast pass per target day and a complete
35-day full-grid history before every issue date.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import data as D
from .inference import validate_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--start', required=True, help='First target date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='Last target date (YYYY-MM-DD)')
    parser.add_argument('--lead', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()

    if not (D.ART / 'climatology_by_doy.npz').exists():
        parser.error('Missing original artifacts/blocked/climatology_by_doy.npz.')
    import pandas as pd
    from .experts import NwpblendExpert

    targets = pd.date_range(args.start, args.end, freq='D')
    if targets.empty:
        parser.error('The requested target-date window is empty.')
    history = pd.read_parquet(args.history)
    grid = pd.read_parquet(D.ART / 'grid_static.parquet')
    model = NwpblendExpert(D.ART)
    summary = grid[['cell', 'latitude', 'longitude']].copy().set_index('cell')
    totals = {name: pd.Series(0.0, index=summary.index) for name in (
        'rainfall_total_mm', 'temperature_sum_C', 'wind_sum_ms',
        'heavy_rain_expected_days', 'heatwave_expected_days', 'high_wind_expected_days',
    )}

    for target in targets:
        issue = target - pd.Timedelta(days=args.lead)
        valid = validate_history(history, grid, issue)
        output = model.infer(valid, str(issue.date()), leads=[args.lead]).set_index('cell')
        totals['rainfall_total_mm'] += output['pred_tp_mm']
        totals['temperature_sum_C'] += output['pred_t2m_C_mean']
        totals['wind_sum_ms'] += output['pred_wind_speed_mean']
        totals['heavy_rain_expected_days'] += output['prob_heavy_rain_day_flag']
        totals['heatwave_expected_days'] += output['prob_heatwave_day_flag']
        totals['high_wind_expected_days'] += output['prob_high_wind_day_flag']
        print(f'Processed target {target.date()}', flush=True)

    days = len(targets)
    for name, values in totals.items():
        summary[name] = values
    summary['temperature_mean_C'] = summary.pop('temperature_sum_C') / days
    summary['wind_mean_ms'] = summary.pop('wind_sum_ms') / days
    summary['heavy_rain_probability_mean'] = summary['heavy_rain_expected_days'] / days
    summary['heatwave_probability_mean'] = summary['heatwave_expected_days'] / days
    summary['high_wind_probability_mean'] = summary['high_wind_expected_days'] / days
    summary = summary.reset_index()

    D.RUNTIME.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    output_path = D.RUNTIME / f'historical-summary-{stamp}.csv'
    summary.to_csv(output_path, index=False)
    metadata = {
        'file': output_path.name, 'method':'Repeated BlendingForecaster.forecast with original preprocessing',
        'history':str(args.history), 'window_start':str(targets[0].date()), 'window_end':str(targets[-1].date()),
        'window_days':days, 'lead_days':args.lead, 'generated_at':datetime.now(timezone.utc).isoformat(),
    }
    temporary = D.RUNTIME / 'long_term_latest.tmp'
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, D.RUNTIME / 'long_term_latest.json')
    print(f'Published {len(summary)} grid-cell summaries for {days} target days to {output_path}')


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.getenv('HYDRA_MODEL_REPO', str(ROOT / 'nwpblend')))
ART = REPO / 'artifacts' / 'blocked'
MANIFEST = json.loads((ART / 'manifest.json').read_text())
TARGETS = {'tp_mm': ('Rainfall', 'mm/day'), 't2m_C_mean': ('Temperature', '°C'), 'wind_speed_mean': ('Wind speed', 'm/s')}
ARCHIVE = REPO / 'reports' / 'inference_demo_2025-12-28.csv'
SNAPSHOT = REPO / 'forecast_india_2026-09-24_0h_clean.csv'
RUNTIME = ROOT / 'runtime'
ERA5_IMPORT_PREFLIGHT = ROOT / 'data' / 'processed' / 'hydra_era5_2025_import_preflight.json'
HYDRA_DAILY_MEAN_BLEND = RUNTIME / 'hydra_daily_mean_state_blend.json'
INDIA_STATES = ROOT / 'frontend' / 'public' / 'india-states.geojson'
MAP_CITIES = (
    ('Srinagar', 34.0837, 74.7973), ('Shimla', 31.1048, 77.1734),
    ('New Delhi', 28.6139, 77.2090), ('Jaipur', 26.9124, 75.7873),
    ('Lucknow', 26.8467, 80.9462), ('Guwahati', 26.1445, 91.7362),
    ('Patna', 25.5941, 85.1376), ('Kolkata', 22.5726, 88.3639),
    ('Ahmedabad', 23.0225, 72.5714), ('Bhopal', 23.2599, 77.4126),
    ('Bhubaneswar', 20.2961, 85.8245), ('Mumbai', 19.0760, 72.8777),
    ('Hyderabad', 17.3850, 78.4867), ('Bengaluru', 12.9716, 77.5946),
    ('Chennai', 13.0827, 80.2707), ('Thiruvananthapuram', 8.5241, 76.9366),
)


@lru_cache(maxsize=16)
def read_csv_cached(path: str, modified: int):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def rows(path: Path):
    return read_csv_cached(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def digest(path: str, modified: int):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(path: Path):
    head = (REPO / '.git' / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = REPO / '.git' / head[5:]
        head = ref.read_text().strip() if ref.exists() else head
    return {'repository': 'nidhiiiii0707/nwpblend', 'commit': head,
            'file': path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
            'sha256': digest(str(path), path.stat().st_mtime_ns), 'checkpoint': 'blocked',
            'method': 'Archived repository output' if path == ARCHIVE else 'Supplied dataset'}


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def production_path():
    marker = RUNTIME / 'latest.json'
    if not marker.exists():
        return None
    name = json.loads(marker.read_text())['file']
    p = (RUNTIME / name).resolve()
    if p.parent != RUNTIME.resolve() or not p.exists():
        return None
    return p


def daily_mean_hydra_state_outlook(name):
    """Read a published, separately retrained HYDRA neural-gating state cycle."""
    if not HYDRA_DAILY_MEAN_BLEND.exists():
        return None
    try:
        payload = json.loads(HYDRA_DAILY_MEAN_BLEND.read_text(encoding='utf-8'))
        state = payload.get('states', {}).get(name)
        if payload.get('kind') != 'hydra_daily_mean_neural_gating' or not isinstance(state, dict):
            return None
        outlooks = state.get('outlooks')
        if not isinstance(outlooks, list) or {item.get('lead_hours') for item in outlooks} != {24, 48}:
            return None
        return {
            'status': 'available', 'state': name, 'outlooks': outlooks,
            'source': 'HYDRA daily-mean neural-gating adaptive blend',
            'issue_date': payload.get('issue_date'), 'message': payload.get('message'),
            'nearest_grid_fallback': bool(state.get('nearest_grid_fallback')),
        }
    except (OSError, ValueError, TypeError):
        return None


def era5_import_preflight():
    """Return the latest read-only ERA5 import check, if one has been run.

    The report records source coverage only.  It never makes a daily-mean
    archive eligible for fresh HYDRA inference by itself.
    """
    if not ERA5_IMPORT_PREFLIGHT.exists():
        return None
    try:
        payload = json.loads(ERA5_IMPORT_PREFLIGHT.read_text(encoding='utf-8'))
        return payload.get('summary') if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def long_term_path():
    """Return the safely published historical-model summary, if one exists."""
    marker = RUNTIME / 'long_term_latest.json'
    if not marker.exists():
        return None, None
    metadata = json.loads(marker.read_text())
    name = metadata.get('file')
    if not isinstance(name, str):
        return None, None
    path = (RUNTIME / name).resolve()
    if path.parent != RUNTIME.resolve() or not path.exists() or path.suffix != '.csv':
        return None, None
    return path, metadata


def forecast_path(source):
    return ARCHIVE if source == 'archive' else production_path()


def pending(message):
    return {'status': 'integration_pending', 'message': message}


@lru_cache(maxsize=1)
def state_features():
    return json.loads(INDIA_STATES.read_text(encoding='utf-8'))['features']


def _rings(geometry):
    if geometry['type'] == 'Polygon':
        return [geometry['coordinates']]
    if geometry['type'] == 'MultiPolygon':
        return geometry['coordinates']
    return []


def _in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i, (xi, yi, *_) in enumerate(ring):
        xj, yj = ring[j][:2]
        if (yi > lat) != (yj > lat):
            crossing = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < crossing:
                inside = not inside
        j = i
    return inside


def point_in_geometry(lon, lat, geometry):
    for polygon in _rings(geometry):
        if polygon and _in_ring(lon, lat, polygon[0]) and not any(_in_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


@lru_cache(maxsize=20000)
def point_in_india(lon, lat):
    return any(west <= lon <= east and south <= lat <= north and point_in_geometry(lon, lat, geometry)
               for west, south, east, north, geometry in state_bounds())


def state_feature(name):
    wanted = name.casefold().strip()
    return next((f for f in state_features() if f['properties']['name'].casefold() == wanted), None)


@lru_cache(maxsize=1)
def state_index():
    """State geometries with bounds, used for repeated grid-cell lookups."""
    indexed = []
    for feature in state_features():
        geometry = feature['geometry']
        coordinates = [point for polygon in _rings(geometry) for ring in polygon for point in ring]
        longitudes = [point[0] for point in coordinates]
        latitudes = [point[1] for point in coordinates]
        indexed.append((min(longitudes), min(latitudes), max(longitudes), max(latitudes),
                        feature['properties']['name'], geometry))
    return tuple(indexed)


def state_at(latitude, longitude):
    """Return the mapped Indian state/UT at a supplied grid point, if any."""
    return next((name for west, south, east, north, name, geometry in state_index()
                 if west <= longitude <= east and south <= latitude <= north
                 and point_in_geometry(longitude, latitude, geometry)), None)


def nearby_map_city(latitude, longitude, max_distance_degrees=1.5):
    """Return a nearby displayed-city label without pretending it is the cell."""
    city, city_latitude, city_longitude = min(MAP_CITIES, key=lambda item:(item[1]-latitude)**2+(item[2]-longitude)**2)
    distance = math.hypot(city_latitude-latitude, city_longitude-longitude)
    return city if distance <= max_distance_degrees else None


def geometry_center(geometry):
    coordinates = []
    for polygon in _rings(geometry):
        if polygon:
            coordinates.extend(polygon[0])
    return {'latitude': sum(p[1] for p in coordinates) / len(coordinates),
            'longitude': sum(p[0] for p in coordinates) / len(coordinates)}


@lru_cache(maxsize=1)
def state_bounds():
    indexed = []
    for feature in state_features():
        coordinates = [point for polygon in _rings(feature['geometry']) for ring in polygon for point in ring]
        longitudes = [point[0] for point in coordinates]
        latitudes = [point[1] for point in coordinates]
        indexed.append((min(longitudes), min(latitudes), max(longitudes), max(latitudes), feature['geometry']))
    return indexed


def _imd_state_name(name):
    aliases = {
        'andaman and nicobar islands': 'ANDAMAN & NICOBAR (UT)',
        'dadra and nagar haveli and daman and diu': 'DADRA & NAGAR HAVELI AND DAMAN & DIU (UT)',
        'delhi': 'DELHI (UT)', 'jammu and kashmir': 'JAMMU & KASHMIR (UT)'
    }
    if name.casefold() in aliases:
        return aliases[name.casefold()]
    available = {r['state'] for r in rows(REPO / 'rainfall_statewise_daily_imd_clean.csv')}
    normalized = re.sub(r'[^a-z]', '', name.casefold())
    return next((v for v in available if re.sub(r'[^a-z]', '', v.casefold().replace('ut', '')) == normalized), name.upper())


def imd_latest_by_state():
    data = rows(REPO / 'rainfall_statewise_daily_imd_clean.csv')
    result = {}
    for feature in state_features():
        name = feature['properties']['name']
        matched = [r for r in data if r['state'] == _imd_state_name(name)]
        if matched:
            row = matched[-1]
            result[name] = {'date': row['date'], 'daily_actual_mm': number(row['daily_actual_mm']),
                            'daily_normal_mm': number(row['daily_normal_mm']), 'daily_category': row['daily_category']}
    return result


def state_context(name, lead, source):
    feature = state_feature(name)
    if not feature:
        return {'status': 'unavailable', 'state': name, 'message': 'Unknown India state or union territory.'}
    path = SNAPSHOT if source == 'ecmwf' else forecast_path(source)
    result = {'status': 'unavailable', 'scope': 'state', 'state': feature['properties']['name'],
              'source': source, 'lead_hours': lead, 'forecast': {}, 'events': [], 'confidence': None,
              'regime': None, 'risk': pending('No state exposure or vulnerability model has been supplied.'),
              'message': '', 'location': {**geometry_center(feature['geometry']), 'state': feature['properties']['name']}}
    imd_name = _imd_state_name(name)
    imd_rows = [r for r in rows(REPO / 'rainfall_statewise_daily_imd_clean.csv') if r['state'] == imd_name]
    result['observations'] = ({'status': 'available', 'source': 'IMD supplied statewise CSV', **imd_rows[-1]}
                              if imd_rows else pending('No matching IMD state observation in the supplied CSV.'))
    if path is None:
        result['message'] = 'Fresh HYDRA state coverage is unavailable until a full-grid inference cycle is published.'
        return result
    candidates = rows(path)
    if source != 'ecmwf':
        candidates = [r for r in candidates if int(r['lead_days']) * 24 == lead]
    elif lead != 0:
        result['message'] = 'The supplied ECMWF field is a zero-hour analysis only.'
        return result
    selected = [r for r in candidates if point_in_geometry(float(r['longitude']), float(r['latitude']), feature['geometry'])]
    if not selected:
        result['message'] = ('The three-cell HYDRA archive has no cell in this state.' if source == 'archive'
                             else 'No supplied grid cells fall inside this state boundary.')
        return result
    result.update(status='available', grid_cell_count=len(selected), provenance=provenance(path))
    if source == 'ecmwf':
        result.update(valid_date=selected[0]['valid_time'], issue_date=selected[0]['init_time'],
                      message='State summary from supplied ECMWF zero-hour grid cells. IMD rainfall is a separate observation date.')
        mapping = {'t2m_C_mean': 't2m_C', 'wind_speed_mean': 'wind_speed'}
        for target, column in mapping.items():
            values = [number(r[column]) for r in selected if number(r[column]) is not None]
            result['forecast'][target] = {'value': statistics.fmean(values), 'minimum': min(values), 'maximum': max(values),
                                          'unit': TARGETS[target][1], 'experts': [], 'interval80': None, 'spread': None}
        result['conditions'] = {column: statistics.fmean([float(r[column]) for r in selected])
                                for column in ['rh_pct', 'msl_hPa', 'tcc']}
    else:
        valid = date.fromisoformat(selected[0]['target_date'][:10])
        result.update(valid_date=str(valid), issue_date=str(valid - timedelta(days=lead // 24)),
                      message=f"State summary from {len(selected)} supplied HYDRA grid cell(s); this is sparse archive coverage.")
        regimes = [r['regime'] for r in selected]
        result['regime'] = max(set(regimes), key=regimes.count)
        for target, (_, unit) in TARGETS.items():
            values = [float(r[f'pred_{target}']) for r in selected]
            experts = []
            for expert in MANIFEST['experts']:
                weights = [float(r[f'w_{expert}_{target}']) for r in selected]
                expert_values = [float(r[f'exp_{expert}_{target}']) for r in selected]
                experts.append({'name': expert, 'weight': statistics.fmean(weights), 'value': statistics.fmean(expert_values)})
            result['forecast'][target] = {'value': statistics.fmean(values), 'minimum': min(values), 'maximum': max(values),
                                          'unit': unit, 'experts': experts,
                                          'spread': statistics.fmean(float(r[f'spread_{target}']) for r in selected),
                                          'interval80': [statistics.fmean(float(r[f'lo80_{target}']) for r in selected),
                                                         statistics.fmean(float(r[f'hi80_{target}']) for r in selected)]}
        for flag, label, threshold in [
            ('heavy_rain_day_flag', 'Heavy rainfall', 'Daily rain ≥64.5 mm'),
            ('heatwave_day_flag', 'Heatwave', 'Land daily maximum temperature ≥40°C'),
            ('high_wind_day_flag', 'High wind', 'Daily maximum sampled wind ≥13.9 m/s'),
        ]:
            probabilities = [number(r.get(f'prob_{flag}')) for r in selected]
            probabilities = [value for value in probabilities if value is not None]
            result['events'].append({'id': f"{name}-{lead}-{flag}", 'name': label,
                                     'probability': statistics.fmean(probabilities) if probabilities else None,
                                     'threshold': threshold, 'valid_date': str(valid)})
    return result


def hydra_state_outlook(name):
    """Aggregate a published fresh HYDRA cycle into +24h and +48h state results.

    This deliberately reads only the locally published output from the trained
    ``BlendingForecaster``. Provider forecasts are kept outside this path so a
    caller can compare them without treating them as an HYDRA expert.
    """
    if state_feature(name) is None:
        return {'status': 'unavailable', 'state': name, 'message': 'Unknown Indian state or union territory.', 'outlooks': []}
    output = production_path()
    if output is None:
        retrained = daily_mean_hydra_state_outlook(name)
        if retrained:
            return retrained
        history = Path(os.getenv('HYDRA_HISTORY', str(REPO/'data/processed/era5_india_2025_daily_processed.parquet')))
        climatology = ART / 'climatology_by_doy.npz'
        import_summary = era5_import_preflight()
        staged_later_grib = REPO / 'e2354a3ec798c3287a0da9e8e6075baf' / 'data.grib'
        rainfall_history = REPO / 'data/processed/era5_india_2025_precipitation_only.parquet'
        missing = []
        if not history.exists():
            if import_summary:
                missing.extend(import_summary.get('blocking_inputs_for_saved_hydra_model', []))
            elif rainfall_history.exists():
                missing.append('2025 trained-feature fields beyond rainfall: temperature, wind, pressure, cloud, CAPE, humidity, boundary-layer height, radiation, and surface mask')
            else:
                missing.append('2025-11-27 to 2025-12-31 full-grid ERA5 daily history')
        if not climatology.exists() and not any('climatology_by_doy.npz' in item for item in missing):
            missing.append('climatology_by_doy.npz')
        missing = list(dict.fromkeys(missing))
        staged_message = (' The later-date 0.25° GRIB is staged for January-August 2026, but it begins on '
                          '2026-01-01 and cannot provide the 35 days before the first issue date.'
                          if staged_later_grib.exists() else '')
        import_message = ''
        if import_summary:
            found = len(import_summary.get('months_found', []))
            missing_months = import_summary.get('missing_months', [])
            import_message = (
                f' The ERA5 daily-mean import check has validated {found}/12 monthly archives'
                + (f' (months still unavailable: {", ".join(str(month) for month in missing_months)}).' if missing_months else '.')
                + ' These source files are retained for observed analysis and are not silently substituted into the trained forecast.'
            )
        return {
            'status': 'awaiting_inference', 'state': name, 'outlooks': [],
            'message': ('HYDRA has not been initialised for 2026 yet.' + import_message +
                        (' The supplied 2025 precipitation history is processed on the 14,157-cell HYDRA India grid; it covers the rainfall input only.' if rainfall_history.exists() and not import_summary else '') + staged_message +
                        ' The trained model is not being replaced by a provider forecast.'),
            'missing_inputs': missing,
            'run_command': 'python -m backend.inference --history <full_grid_history.parquet> --issue <YYYY-MM-DD>',
        }
    outlooks = []
    for lead in (24, 48):
        context = state_context(name, lead, 'production')
        if context['status'] != 'available':
            continue
        outlooks.append({
            'lead_hours': lead, 'valid_date': context.get('valid_date'), 'regime': context.get('regime'),
            'grid_cell_count': context.get('grid_cell_count'), 'forecast': context['forecast'], 'events': context['events'],
        })
    if not outlooks:
        return {'status': 'unavailable', 'state': name, 'outlooks': [],
                'message': 'The published HYDRA cycle contains no grid cells inside this state for +24h or +48h.'}
    return {'status': 'available', 'state': name, 'outlooks': outlooks,
            'source': 'HYDRA adaptive blend · published fresh inference',
            'message': 'State values aggregate only the trained HYDRA grid-cell outputs. Expert weights are averaged across the state grid; intervals and spread remain uncertainty diagnostics.'}


def context(lat, lon, lead, source, state=None):
    if state:
        return state_context(state, lead, source)
    result = {'status': 'unavailable', 'source': source, 'requested': {'latitude': lat, 'longitude': lon},
              'lead_hours': lead, 'forecast': {}, 'events': [], 'confidence': None,
              'risk': pending('Exposure, vulnerability and validated impact models have not been supplied.'),
              'observations': pending('No observations matched to this forecast valid day.'),
              'message': ''}
    path = SNAPSHOT if source == 'ecmwf' else forecast_path(source)
    if path is None:
        result['message'] = 'Fresh inference pending: supply full-grid ERA5 history and climatology_by_doy.npz, then run the inference worker.'
        return result
    data = rows(path)
    if source != 'ecmwf':
        data = [r for r in data if int(r['lead_days']) * 24 == lead]
    elif lead != 0:
        result['message'] = 'The ECMWF dataset contains a zero-hour analysis only.'
        return result
    if not data:
        result['message'] = 'No forecast for this lead.'
        return result
    row = min(data, key=lambda r: (float(r['latitude'])-lat)**2 + (float(r['longitude'])-lon)**2)
    y, x = float(row['latitude']), float(row['longitude'])
    if abs(y-lat) > .12501 or abs(x-lon) > .12501:
        result['message'] = 'No supplied forecast at this location. Select a marked archive cell, or choose ECMWF analysis within India.'
        return result
    result.update(status='available', location={'latitude': y, 'longitude': x, 'cell': row.get('cell')}, provenance=provenance(path))
    if source == 'ecmwf':
        result.update(valid_date=row['valid_time'], issue_date=row['init_time'], regime=None,
                      message='Archived ECMWF zero-hour analysis · 24 Sep 2026. Rain accumulation at step zero is not a daily rainfall forecast.')
        for target, col in [('t2m_C_mean','t2m_C'),('wind_speed_mean','wind_speed')]:
            result['forecast'][target] = {'value': number(row[col]), 'unit': TARGETS[target][1], 'experts': [], 'interval80': None, 'spread': None}
        result['conditions'] = {k: number(row[k]) for k in ['rh_pct','msl_hPa','wind_dir','tcc','d2m_C']}
    else:
        valid = date.fromisoformat(row['target_date'][:10])
        result.update(valid_date=str(valid), issue_date=str(valid-timedelta(days=lead//24)), regime=row['regime'],
                      message='Historical model output · issue time is end of issue day; values describe the target UTC day.')
        if source == 'production':
            result['provenance'].update(json.loads(path.with_suffix('.json').read_text()))
        for target, (_, unit) in TARGETS.items():
            result['forecast'][target] = {
                'value': number(row[f'pred_{target}']), 'unit': unit,
                'spread': number(row.get(f'spread_{target}')),
                'interval80': [number(row.get(f'lo80_{target}')), number(row.get(f'hi80_{target}'))],
                'quantiles': {str(q): number(row.get(f'q{round(q*100):02d}_{target}')) for q in MANIFEST['quantiles']},
                'experts': [{'name': e, 'weight': number(row[f'w_{e}_{target}']), 'value': number(row[f'exp_{e}_{target}'])} for e in MANIFEST['experts']],
                'lgbm': number(row.get(f'lgbm_{target}')), 'static': number(row.get(f'static_{target}'))}
        for flag, label, threshold in [('heavy_rain_day_flag','Heavy rainfall','Daily rain ≥64.5 mm'),('heatwave_day_flag','Heatwave','Land daily maximum temperature ≥40°C'),('high_wind_day_flag','High wind','Daily maximum sampled wind ≥13.9 m/s')]:
            result['events'].append({'id': f"{row['cell']}-{lead}-{flag}", 'name': label, 'probability': number(row.get('prob_'+flag)), 'threshold': threshold,
                                     'valid_date': str(valid), 'severity': None, 'peak_time': None})
    return result


def cell_location(cell: str, lead: int, source: str) -> dict | None:
    """Resolve a supplied forecast cell ID without fabricating coordinates."""
    path = SNAPSHOT if source == 'ecmwf' else forecast_path(source)
    if path is None:
        return None
    for row in rows(path):
        if str(row.get('cell')) != str(cell):
            continue
        if source != 'ecmwf' and int(row.get('lead_days', 0)) * 24 != lead:
            continue
        return {'cell': str(cell), 'latitude': float(row['latitude']), 'longitude': float(row['longitude'])}
    return None


def forecast_rankings(source, lead, variable='tp_mm', event=None, state=None, limit=5):
    """Rank only supplied HYDRA forecast cells for a broad WeatherGPT query.

    This is a forecast scan, not a climatology or a spatial interpolation. Its
    coverage is exactly the cells present in the selected output file.
    """
    path = SNAPSHOT if source == 'ecmwf' else forecast_path(source)
    if path is None or not path.exists():
        return {'status':'unavailable', 'message':'No supplied forecast output is available for this source.', 'items':[]}
    if event and source == 'ecmwf':
        return {'status':'unavailable', 'message':'The ECMWF analysis has no trained extreme-event classifier probabilities.', 'items':[]}
    if source == 'ecmwf' and lead != 0:
        return {'status':'unavailable', 'message':'The ECMWF dataset contains a zero-hour analysis only.', 'items':[]}
    boundary = state_feature(state) if state else None
    if state and boundary is None:
        return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
    event_columns = {'heavy_rain':'prob_heavy_rain_day_flag', 'heatwave':'prob_heatwave_day_flag', 'high_wind':'prob_high_wind_day_flag'}
    archive_columns = {'tp_mm':'pred_tp_mm', 't2m_C_mean':'pred_t2m_C_mean', 'wind_speed_mean':'pred_wind_speed_mean'}
    ecmwf_columns = {'t2m_C_mean':'t2m_C', 'wind_speed_mean':'wind_speed'}
    column = event_columns.get(event) if event else (ecmwf_columns if source == 'ecmwf' else archive_columns).get(variable)
    if column is None:
        return {'status':'unavailable', 'message':'This variable is unavailable for the selected source.', 'items':[]}
    selected = []
    for row in rows(path):
        if source != 'ecmwf' and int(row['lead_days']) * 24 != lead:
            continue
        latitude, longitude = float(row['latitude']), float(row['longitude'])
        if source == 'ecmwf' and not point_in_india(longitude, latitude):
            continue
        if boundary and not point_in_geometry(longitude, latitude, boundary['geometry']):
            continue
        score = number(row.get(column))
        if score is None:
            continue
        evidence_variable = {'heavy_rain':'tp_mm', 'heatwave':'t2m_C_mean', 'high_wind':'wind_speed_mean'}.get(event, variable)
        experts = ([] if source == 'ecmwf' else [
            {'name':expert, 'weight':number(row.get(f'w_{expert}_{evidence_variable}')),
             'value':number(row.get(f'exp_{expert}_{evidence_variable}'))}
            for expert in MANIFEST['experts']
        ])
        experts = [expert for expert in experts if expert['weight'] is not None]
        dominant = max(experts, key=lambda expert:expert['weight']) if experts else None
        selected.append({'cell':row.get('cell'), 'latitude':latitude, 'longitude':longitude,
                         'state':state_at(latitude, longitude), 'nearby_city':nearby_map_city(latitude, longitude), 'score':score,
                         'value':number(row.get(archive_columns.get(variable, ''))),
                         'valid_date':row.get('valid_time') if source == 'ecmwf' else row.get('target_date'),
                         'regime':None if source == 'ecmwf' else row.get('regime'), 'experts':experts,
                         'dominant_expert':dominant, 'spread':None if source == 'ecmwf' else number(row.get(f'spread_{evidence_variable}')),
                         'interval80':None if source == 'ecmwf' else [number(row.get(f'lo80_{evidence_variable}')), number(row.get(f'hi80_{evidence_variable}'))],
                         'evidence_variable':evidence_variable})
    selected.sort(key=lambda item:item['score'], reverse=True)
    return {'status':'available', 'items':selected[:limit], 'total_cells':len(selected), 'lead_hours':lead,
            'event':event, 'variable':variable, 'unit':'%' if event else TARGETS[variable][1],
            'valid_date':selected[0]['valid_date'] if selected else None, 'provenance':provenance(path)}


def long_term_rankings(variable='tp_mm', event=None, window_days=None, state=None, limit=5):
    """Rank an explicitly published repeated-inference summary.

    A summary is created only by the offline historical inference worker. This
    prevents a handful of archived forecast rows from being passed off as a
    seasonal or annual climatology.
    """
    path, metadata = long_term_path()
    if path is None or metadata is None:
        return {'status':'unavailable', 'message':'HYDRA cannot provide a reliable answer to this historical question because the available data do not cover the requested period. The supplied historical rainfall dataset covers 2014 only, and the HYDRA forecast archive contains just three target dates; neither is a complete one-year record. Provide full-India daily observations for the requested period, or the original full-grid ERA5 history and climatology artifact to generate a model-based historical summary.', 'items':[]}
    available_days = metadata.get('window_days')
    if not isinstance(available_days, int) or available_days < 1:
        return {'status':'unavailable', 'message':'The published historical summary has invalid window metadata.', 'items':[]}
    if window_days is not None and window_days != available_days:
        return {'status':'unavailable', 'message':f'The published model summary covers {available_days} days, not the requested {window_days} days. Publish a matching window before comparing long-term risk.', 'items':[]}
    boundary = state_feature(state) if state else None
    if state and boundary is None:
        return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
    event_columns = {'heavy_rain':'heavy_rain_expected_days', 'heatwave':'heatwave_expected_days', 'high_wind':'high_wind_expected_days'}
    event_means = {'heavy_rain':'heavy_rain_probability_mean', 'heatwave':'heatwave_probability_mean', 'high_wind':'high_wind_probability_mean'}
    value_columns = {'tp_mm':'rainfall_total_mm', 't2m_C_mean':'temperature_mean_C', 'wind_speed_mean':'wind_mean_ms'}
    score_column = event_columns.get(event) if event else value_columns.get(variable)
    if score_column is None:
        return {'status':'unavailable', 'message':'This long-term variable is unavailable.', 'items':[]}
    selected = []
    for row in rows(path):
        latitude, longitude = float(row['latitude']), float(row['longitude'])
        if boundary and not point_in_geometry(longitude, latitude, boundary['geometry']):
            continue
        score = number(row.get(score_column))
        if score is None:
            continue
        selected.append({'cell':row.get('cell'), 'latitude':latitude, 'longitude':longitude,
                         'state':state_at(latitude, longitude), 'nearby_city':nearby_map_city(latitude, longitude),
                         'score':score, 'rainfall_total_mm':number(row.get('rainfall_total_mm')),
                         'mean_probability':number(row.get(event_means[event])) if event else None})
    selected.sort(key=lambda item:item['score'], reverse=True)
    return {'status':'available', 'items':selected[:limit], 'total_cells':len(selected), 'window_days':available_days,
            'window_start':metadata.get('window_start'), 'window_end':metadata.get('window_end'), 'event':event,
            'variable':variable, 'provenance':provenance(path), 'metadata':metadata}


@lru_cache(maxsize=4)
def historical_rainfall_rankings(year: int, state: str | None = None, limit: int = 5):
    """Rank observed annual rainfall from a supplied complete yearly grid.

    This is deliberately separate from model inference: it sums observations
    for the requested calendar year and does not claim a forecast or a farm
    suitability score.
    """
    partial_era5 = REPO / f'data/processed/era5_india_{year}_precipitation_only.parquet'
    if partial_era5.exists():
        import pandas as pd
        boundary = state_feature(state) if state else None
        if state and boundary is None:
            return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
        source = pd.read_parquet(partial_era5, columns=['date', 'cell', 'latitude', 'longitude', 'tp_mm'])
        dates = pd.DatetimeIndex(source['date']).normalize()
        expected_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(dates.unique()) != expected_days or dates.min() != pd.Timestamp(f'{year}-01-01') or dates.max() != pd.Timestamp(f'{year}-12-31'):
            return {'status':'unavailable', 'message':f'The supplied {year} ERA5 precipitation history does not contain a complete calendar year.', 'items':[]}
        totals = source.groupby(['cell', 'latitude', 'longitude'], sort=False)['tp_mm'].agg(['sum', 'count']).reset_index()
        candidates = []
        for _cell, latitude_raw, longitude_raw, total_raw, count_raw in totals.itertuples(index=False, name=None):
            if count_raw != expected_days:
                continue
            latitude, longitude = float(latitude_raw), float(longitude_raw)
            if boundary:
                if not point_in_geometry(longitude, latitude, boundary['geometry']):
                    continue
                mapped_state = boundary['properties']['name']
            else:
                mapped_state = state_at(latitude, longitude)
            if mapped_state is not None:
                candidates.append((float(total_raw), latitude, longitude, mapped_state))
        candidates.sort(reverse=True)
        items = [{'state': mapped_state, 'nearby_city': nearby_map_city(latitude, longitude),
                  'latitude': latitude, 'longitude': longitude, 'rainfall_total_mm': total}
                 for total, latitude, longitude, mapped_state in candidates[:limit]]
        if not items:
            return {'status':'unavailable', 'message':f'No complete daily rainfall observations could be matched to {state or "India"} for {year}.', 'items':[]}
        return {'status':'available', 'items':items, 'total_cells':len(candidates), 'year':year,
                'window_start':f'{year}-01-01', 'window_end':f'{year}-12-31', 'observation_days':expected_days,
                'unit':'mm', 'source':partial_era5.name,
                'method':'ERA5 daily-mean precipitation at 6-hour sampling converted to mm/day'}
    path = ROOT / f'crop-yield-analysis/rainfall_data/RF25_ind{year}_rfp25.nc'
    if not path.exists():
        return {'status':'unavailable', 'message':f'No complete supplied rainfall observation grid is available for {year}. The local historical rainfall dataset covers 2014 only.', 'items':[]}
    boundary = state_feature(state) if state else None
    if state and boundary is None:
        return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
    import xarray as xr
    with xr.open_dataset(path) as dataset:
        rainfall = dataset['RAINFALL']
        times = [str(item)[:10] for item in dataset['TIME'].values]
        expected_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(times) != expected_days or times[0] != f'{year}-01-01' or times[-1] != f'{year}-12-31':
            return {'status':'unavailable', 'message':f'The supplied {year} rainfall grid does not contain a complete calendar year.', 'items':[]}
        totals = rainfall.sum(dim='TIME', skipna=True).values
        counts = rainfall.count(dim='TIME').values
        latitudes = dataset['LATITUDE'].values
        longitudes = dataset['LONGITUDE'].values
    candidates = []
    for latitude_index, latitude_raw in enumerate(latitudes):
        latitude = float(latitude_raw)
        for longitude_index, longitude_raw in enumerate(longitudes):
            longitude = float(longitude_raw)
            total = number(totals[latitude_index, longitude_index])
            if total is None or int(counts[latitude_index, longitude_index]) != expected_days:
                continue
            candidates.append((total, latitude, longitude))
    candidates.sort(reverse=True)
    items = []
    for total, latitude, longitude in candidates:
        if boundary:
            if not point_in_geometry(longitude, latitude, boundary['geometry']):
                continue
            mapped_state = boundary['properties']['name']
        else:
            mapped_state = state_at(latitude, longitude)
        if mapped_state is None:
            continue
        items.append({'state':mapped_state, 'nearby_city':nearby_map_city(latitude, longitude),
                      'latitude':latitude, 'longitude':longitude, 'rainfall_total_mm':total})
        if len(items) == limit:
            break
    if not items:
        return {'status':'unavailable', 'message':f'No complete daily rainfall observations could be matched to {state or "India"} for {year}.', 'items':[]}
    return {'status':'available', 'items':items, 'total_cells':len(candidates), 'year':year,
            'window_start':f'{year}-01-01', 'window_end':f'{year}-12-31', 'observation_days':expected_days,
            'unit':'mm', 'source':path.name}


@lru_cache(maxsize=4)
def historical_rainfall_month_rankings(year: int, state: str | None = None, direction: str = 'descending', limit: int = 12):
    """Rank calendar months by mean accumulated observed rainfall over a region."""
    partial_era5 = REPO / f'data/processed/era5_india_{year}_precipitation_only.parquet'
    if partial_era5.exists():
        import pandas as pd
        boundary = state_feature(state) if state else None
        if state and boundary is None:
            return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
        source = pd.read_parquet(partial_era5, columns=['date', 'cell', 'latitude', 'longitude', 'tp_mm'])
        dates = pd.DatetimeIndex(source['date']).normalize()
        expected_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(dates.unique()) != expected_days or dates.min() != pd.Timestamp(f'{year}-01-01') or dates.max() != pd.Timestamp(f'{year}-12-31'):
            return {'status':'unavailable', 'message':f'The supplied {year} ERA5 precipitation history does not contain a complete calendar year.', 'items':[]}
        cells = source[['cell', 'latitude', 'longitude']].drop_duplicates()
        allowed = []
        for row in cells.itertuples(index=False):
            if boundary:
                include = point_in_geometry(float(row.longitude), float(row.latitude), boundary['geometry'])
            else:
                include = state_at(float(row.latitude), float(row.longitude)) is not None
            if include:
                allowed.append(row.cell)
        filtered = source[source['cell'].isin(allowed)].copy()
        filtered['month_number'] = pd.DatetimeIndex(filtered['date']).month
        monthly = filtered.groupby(['cell', 'month_number'], sort=False)['tp_mm'].sum().groupby('month_number').mean()
        names = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
        items = [{'month':names[index - 1], 'month_number':index, 'rainfall_mean_mm':float(total), 'grid_cells':len(allowed)}
                 for index, total in monthly.items()]
        items.sort(key=lambda item:item['rainfall_mean_mm'], reverse=direction != 'ascending')
        return {'status':'available', 'items':items[:limit], 'year':year, 'window_start':f'{year}-01-01',
                'window_end':f'{year}-12-31', 'observation_days':expected_days, 'grid_cells':len(allowed),
                'unit':'mm', 'source':partial_era5.name,
                'statistic':'Mean monthly accumulated rainfall across matched 0.25° ERA5 grid cells'}
    path = ROOT / f'crop-yield-analysis/rainfall_data/RF25_ind{year}_rfp25.nc'
    if not path.exists():
        return {'status':'unavailable', 'message':f'No complete supplied rainfall observation grid is available for {year}. The local historical rainfall dataset covers 2014 only.', 'items':[]}
    boundary = state_feature(state) if state else None
    if state and boundary is None:
        return {'status':'unavailable', 'message':'Unknown Indian state or union territory.', 'items':[]}
    import xarray as xr
    with xr.open_dataset(path) as dataset:
        rainfall = dataset['RAINFALL']
        times = [str(item)[:10] for item in dataset['TIME'].values]
        expected_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(times) != expected_days or times[0] != f'{year}-01-01' or times[-1] != f'{year}-12-31':
            return {'status':'unavailable', 'message':f'The supplied {year} rainfall grid does not contain a complete calendar year.', 'items':[]}
        monthly = rainfall.groupby('TIME.month').sum(dim='TIME', skipna=True).values
        latitudes = dataset['LATITUDE'].values
        longitudes = dataset['LONGITUDE'].values
    totals = [0.0] * 12
    grid_counts = [0] * 12
    matched_cells = 0
    for latitude_index, latitude_raw in enumerate(latitudes):
        latitude = float(latitude_raw)
        for longitude_index, longitude_raw in enumerate(longitudes):
            longitude = float(longitude_raw)
            if boundary:
                if not point_in_geometry(longitude, latitude, boundary['geometry']):
                    continue
            elif state_at(latitude, longitude) is None:
                continue
            matched_cells += 1
            for month_index in range(12):
                rainfall_total = number(monthly[month_index, latitude_index, longitude_index])
                if rainfall_total is not None:
                    totals[month_index] += rainfall_total
                    grid_counts[month_index] += 1
    if not matched_cells:
        return {'status':'unavailable', 'message':f'No rainfall grid cells could be matched to {state or "India"}.', 'items':[]}
    names = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
    items = [{'month':names[index], 'month_number':index + 1,
              'rainfall_mean_mm':totals[index] / grid_counts[index], 'grid_cells':grid_counts[index]}
             for index in range(12) if grid_counts[index]]
    items.sort(key=lambda item:item['rainfall_mean_mm'], reverse=direction != 'ascending')
    return {'status':'available', 'items':items[:limit], 'year':year, 'window_start':f'{year}-01-01',
            'window_end':f'{year}-12-31', 'observation_days':expected_days, 'grid_cells':matched_cells,
            'unit':'mm', 'source':path.name, 'statistic':'Mean monthly accumulated rainfall across matched 0.25° grid cells'}


def coverage(lat: float, lon: float) -> dict:
    """Return supplied sources that have a cell within half a native grid step."""
    sources: list[str] = []
    for source, path in (("archive", ARCHIVE), ("ecmwf", SNAPSHOT), ("production", forecast_path("production"))):
        if path is None or not path.exists():
            continue
        candidates = rows(path)
        if source == "ecmwf":
            candidates = [row for row in candidates if point_in_india(float(row["longitude"]), float(row["latitude"]))]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda row: (float(row["latitude"]) - lat) ** 2 + (float(row["longitude"]) - lon) ** 2)
        if abs(float(nearest["latitude"]) - lat) <= 0.12501 and abs(float(nearest["longitude"]) - lon) <= 0.12501:
            sources.append(source)
    return {"has_data": bool(sources), "sources": sources}


def locations():
    seen = {}
    for row in rows(ARCHIVE):
        seen[row['cell']] = {'name': f"Archive cell {row['cell']}", 'latitude': float(row['latitude']), 'longitude': float(row['longitude'])}
    return list(seen.values())


def health():
    history = Path(os.getenv('HYDRA_HISTORY', str(REPO/'data/processed/era5_india_2025_daily_processed.parquet')))
    import_summary = era5_import_preflight()
    import_status = 'not_checked'
    if import_summary:
        import_status = 'complete_source_set' if import_summary.get('source_archive_set_complete') else 'incomplete_source_set'
    missing = [name for name, ok in [('ERA5 daily full-grid history', history.exists()), ('climatology_by_doy.npz', (ART/'climatology_by_doy.npz').exists())] if not ok]
    return {'status': 'degraded' if missing else 'ready_for_inference', 'mode': os.getenv('HYDRA_MODE','demo'),
            'api': 'healthy', 'archive': 'available', 'production_output': production_path() is not None,
            'missing': missing, 'era5_import': import_summary, 'stages': [{'name': n, 'status': s, 'last_success': None, 'latency_ms': None} for n,s in [
                ('Repository datasets','available'),('Model checkpoints','available'),('ERA5 history','missing' if not history.exists() else 'available'),
                ('2025 ERA5 import',import_status),
                ('Shared climatology','missing' if not (ART/'climatology_by_doy.npz').exists() else 'available'),('Fresh inference','pending'),('Risk models','pending')]],
            'sources': [{'name': n, 'status': s} for n,s in [('HYDRA','archived'),('ECMWF IFS','archived zero-hour snapshot'),('IMD','archived'),('Rainfall NetCDF','historical 2014 sample'),('AIFS','integration pending'),('GFS','integration pending'),('GEFS','integration pending'),('Radar / IMERG / AWS','integration pending')]]}

from __future__ import annotations

from datetime import date, timedelta
from enum import IntEnum
from typing import Literal
import logging
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import data as D
from . import live_weather
from . import imagery
from weather_query_parser.nlp.query_catalog import QUERY_CATALOG

app = FastAPI(title='HYDRA Intelligence API', version='0.1.0', docs_url='/api/docs')
log = logging.getLogger('hydra')
Source = Literal['archive','ecmwf','production']


class Lead(IntEnum):
    analysis = 0
    day1 = 24
    day2 = 48
    day3 = 72


class Selection(BaseModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    lead: Literal[0,24,48,72] = 24
    source: Source = 'archive'
    state: str | None = Field(default=None, max_length=100)


class Scenario(Selection):
    rain_percent: float = Field(default=0, ge=-100, le=100)
    wind_percent: float = Field(default=0, ge=-100, le=100)
    temperature_delta: float = Field(default=0, ge=-10, le=10)


class Question(Selection):
    question: str = Field(min_length=1, max_length=2000)


def methodology_answer(topic: str) -> str:
    """Explain documented HYDRA behavior without inventing model internals."""
    experts = ', '.join(D.MANIFEST['experts'])
    if topic == 'architecture':
        return (f"HYDRA uses the supplied nwpblend trained blend for rainfall, temperature, and wind. Its saved forecast output combines these trained experts: {experts}. "
                "For each grid cell and lead, the learned gate allocates weights across the experts and produces a blended forecast. This prototype reads the saved outputs; fresh inference requires the original full-grid ERA5 history and shared climatology artifact.")
    if topic == 'weights':
        return ("Expert weights are the trained blend's allocation across its available experts for a particular forecast. They sum to one; the largest weight identifies the most influential saved expert for that forecast. "
                "Weights do not prove that an expert caused the weather outcome and are not a confidence percentage.")
    if topic == 'uncertainty':
        return ("HYDRA reports expert spread as disagreement among the saved expert forecasts and includes the saved 80% interval when the selected output provides one. "
                "Neither value is presented as calibrated forecast confidence. The prototype does not publish a separate calibrated confidence or bust-risk model.")
    if topic == 'events':
        return ("The supplied trained event classifiers cover heavy rain, heatwave, and high wind. Their documented daily labels are rainfall at least 64.5 mm, land daily maximum temperature at least 40 °C, and daily maximum sampled wind at least 13.9 m/s. "
                "A classifier probability is a short-range forecast probability for its label, not a long-term statement that a place is inherently prone to that event.")
    if topic == 'data_coverage':
        missing = D.health()['missing']
        return ("HYDRA has the supplied archive, checkpoints, a 2014 daily rainfall grid, and an ECMWF zero-hour analysis snapshot. "
                f"Fresh trained inference is unavailable because these required inputs are missing: {', '.join(missing)}. "
                "The live timeline begins only when HYDRA saves provider samples, and it cannot reconstruct earlier days.")
    if topic == 'provenance':
        return ("Each trained-output answer identifies the source file, repository commit, checkpoint label, and SHA-256 digest when that provenance is available. "
                "This makes it possible to distinguish archived model output, historical observations, provider layers, and imagery metadata.")
    return "Ask about a forecast value, expert weights, uncertainty, trained event classifiers, data coverage, or provenance."


def weight_distribution_answer(context: dict, requested_location: str | None = None) -> str:
    """Render saved adaptive-blend allocations for one actual forecast context."""
    location = context.get('location') or {}
    location_text = requested_location or (
        f"grid cell {location.get('cell')} at {location.get('latitude'):.2f} N, {location.get('longitude'):.2f} E"
        if location.get('cell') is not None else "the selected forecast"
    )
    sections = []
    for target, forecast in context.get('forecast', {}).items():
        experts = [expert for expert in forecast.get('experts', []) if expert.get('weight') is not None]
        if not experts:
            continue
        total = sum(float(expert['weight']) for expert in experts)
        allocation = ", ".join(
            f"{expert['name']} {float(expert['weight']) * 100:.1f}%"
            + (f" ({float(expert['value']):.2f} {forecast['unit']})" if expert.get('value') is not None else "")
            for expert in sorted(experts, key=lambda item: float(item['weight']), reverse=True)
        )
        dominant = max(experts, key=lambda item: float(item['weight']))
        interval = forecast.get('interval80')
        uncertainty = (
            f"; expert spread {forecast['spread']:.2f} {forecast['unit']}; saved 80% interval {interval[0]:.2f}-{interval[1]:.2f} {forecast['unit']}"
            + (f" ({forecast['interval_note']})" if forecast.get('interval_note') else "")
            if forecast.get('spread') is not None and interval and None not in interval else ""
        )
        sections.append(
            f"{D.TARGETS[target][0]}: {allocation}. Total allocation {total * 100:.1f}%; "
            f"dominant expert {dominant['name']} ({float(dominant['weight']) * 100:.1f}%){uncertainty}."
        )
    if not sections:
        return (
            f"No saved HYDRA adaptive-blend weights are available for {location_text} in this source. "
            "Provider and ECMWF analysis values are not presented as HYDRA expert weights."
        )
    return (
        f"HYDRA adaptive-blend weight distribution for {location_text}, valid {context.get('valid_date')} at +{context.get('lead_hours')}h:\n"
        + "\n".join(sections)
        + "\nWeights are learned blend allocations for this saved forecast. They sum to about 100% per target, are not calibrated confidence, and do not establish causal attribution."
    )


@app.get('/api/health')
def health():
    return D.health()


@app.get('/api/catalog')
def catalog():
    return {'manifest': D.MANIFEST, 'locations': D.locations(), 'variables': D.TARGETS,
            'states': [{**f['properties'], **D.geometry_center(f['geometry'])} for f in D.state_features()],
            'state_observations': D.imd_latest_by_state(),
            'default_source': 'production' if D.health()['mode']=='production' else 'archive',
            'layers': [{'id': k, 'name': v, 'group': group, 'available': available} for k,v,group,available in [
                ('forecast','HYDRA optimized points','Forecast',True),('heatmap','Grid-value heatmap','Forecast',True),('temperature','Temperature grid','Forecast',True),('wind','Wind notation','Forecast',True),
                ('live_weather','Live weather locations','Observations',True),('open_meteo_rainfall','Open-Meteo current rainfall','Observations',True),('open_meteo_rainfall_24h','Open-Meteo 24h rain outlook','Observations',True),('open_meteo_humidity','Open-Meteo humidity','Observations',True),('open_meteo_lightning','Lightning watch (Open-Meteo)','Observations',True),('open_meteo_thunderstorm','Thunderstorm potential','Observations',True),('open_meteo_wind_gust','Wind-gust risk','Observations',True),('open_meteo_heat_stress','Heat stress','Impact',True),('open_meteo_soil_moisture','Soil moisture / crop stress','Impact',True),
                ('spread','Model disagreement','Intelligence',True),('weights','Expert weights','Intelligence',True),('regime','Weather regime','Intelligence',True),
                ('events','Extreme probabilities','Events',True),('confidence','Forecast confidence','Intelligence',False),('bust','Forecast bust risk','Intelligence',False),
                ('rainfall_obs','Current city weather','Observations',True),('radar','Radar precipitation','Observations',True),('satellite','IMD satellite IR','Observations',True),('flood','Flood risk','Impact',False),('exposure','Population exposure','Impact',False)]]}


@app.get('/api/weathergpt-capabilities')
def weathergpt_capabilities():
    """Machine-readable question buckets exposed by WeatherGPT."""
    return {'buckets': QUERY_CATALOG}


@app.get('/api/live-weather')
def live_weather_reports():
    """Cached IndianAPI reports enriched by Open-Meteo current weather."""
    return live_weather.station_reports()


@app.get('/api/live-weather-history')
def live_weather_history(layer: str=Query('live_weather', max_length=100), days: int=Query(7, ge=1, le=365), state: str | None=Query(default=None, max_length=100)):
    """Retained daily provider samples for the live map layers."""
    return live_weather.historical_rank_layer(layer, days, state)


@app.get('/api/current-observation')
def current_observation(latitude: float=Query(ge=-90,le=90), longitude: float=Query(ge=-180,le=180), state: str | None=Query(default=None,max_length=100)):
    return live_weather.current_report(state, latitude, longitude)


@app.get('/api/state-outlook')
def state_outlook(state: str=Query(min_length=2, max_length=100), latitude: float=Query(ge=-90,le=90), longitude: float=Query(ge=-180,le=180)):
    """HYDRA-first state outlook with a separately labelled provider comparison."""
    if D.state_feature(state) is None:
        raise HTTPException(404, 'Unknown Indian state or union territory')
    return {
        'state': state,
        'hydra': D.hydra_state_outlook(state),
        'provider_comparison': live_weather.state_outlook(state, latitude, longitude),
    }


@app.get('/api/hydra-rolling-rainfall')
def hydra_rolling_rainfall(state: str=Query(min_length=2, max_length=100), start: date | None=None, end: date | None=None):
    """Six-month, one-day HYDRA rainfall replay for the Overview chart."""
    feature = D.state_feature(state)
    if feature is None:
        raise HTTPException(404, 'Unknown India state or union territory.')
    if start and end and start > end:
        raise HTTPException(422, 'start must be on or before end')
    return D.hydra_rolling_rainfall_replay(feature['properties']['name'], start, end)


@app.get('/api/hydra-rainfall-validation')
def hydra_rainfall_validation():
    """Rolling-origin validation of the HYDRA v3 rainfall model."""
    return D.hydra_rainfall_validation()


@app.get('/api/radar-frames')
def radar_frames():
    try:
        return imagery.radar()
    except httpx.HTTPError:
        return {'status':'unavailable','message':'Radar frames unavailable','radar':[]}


@app.get('/api/imd-satellite')
def imd_satellite():
    try:
        content, headers = imagery.satellite()
        return Response(content=content, headers=headers, media_type=headers['Content-Type'])
    except httpx.HTTPError:
        return Response(status_code=503, content='IMD satellite image unavailable')


@app.get('/api/imd-satellite-meta')
def imd_satellite_meta():
    return imagery.satellite_bounds()


@app.get('/api/forecast')
def forecast(latitude: float=Query(ge=-90,le=90,allow_inf_nan=False), longitude: float=Query(ge=-180,le=180,allow_inf_nan=False), lead: Lead=Lead.day1, source: Source='archive', state: str | None=Query(default=None,max_length=100)):
    return D.context(latitude,longitude,lead,source,state)


@app.get('/api/coverage')
def coverage(latitude: float=Query(ge=-90,le=90,allow_inf_nan=False), longitude: float=Query(ge=-180,le=180,allow_inf_nan=False)):
    return D.coverage(latitude, longitude)


@app.get('/api/field')
def field(source: Source='archive', lead: Lead=Lead.day1, variable: Literal['tp_mm','t2m_C_mean','wind_speed_mean']='tp_mm', state: str | None=Query(default=None,max_length=100), west: float=Query(-180,ge=-180,le=180), south: float=Query(-90,ge=-90,le=90), east: float=Query(180,ge=-180,le=180), north: float=Query(90,ge=-90,le=90)):
    if west>east or south>north:
        raise HTTPException(422,'Invalid bounding box')
    path = D.SNAPSHOT if source=='ecmwf' else D.forecast_path(source)
    data = D.rows(path) if path else []
    col = {'tp_mm':None,'t2m_C_mean':'t2m_C','wind_speed_mean':'wind_speed'}[variable] if source=='ecmwf' else 'pred_'+variable
    boundary = D.state_feature(state) if state else None
    if state and boundary is None:
        raise HTTPException(404, 'Unknown Indian state or union territory')
    selected = [r for r in data if west<=float(r['longitude'])<=east and south<=float(r['latitude'])<=north and (lead==0 if source=='ecmwf' else int(r['lead_days'])*24==lead) and (source!='ecmwf' or D.point_in_india(float(r['longitude']),float(r['latitude']))) and (boundary is None or D.point_in_geometry(float(r['longitude']),float(r['latitude']),boundary['geometry']))]
    # The supplied ECMWF snapshot is only ~14k India cells, which is practical
    # to render as its native 0.25° grid. Archive data stays bounded for safety.
    max_points = 15000 if source == 'ecmwf' else 1200
    stride = max(1, (len(selected)+max_points-1)//max_points)
    features = []
    for r in selected[::stride]:
        experts = D.MANIFEST['experts'] if source!='ecmwf' else []
        weights = {e:D.number(r.get(f'w_{e}_{variable}')) for e in experts}
        dominant = max(weights,key=lambda e:weights[e] if weights[e] is not None else -1) if weights else None
        features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[float(r['longitude']),float(r['latitude'])]},'properties':{'value': D.number(r.get(col)) if col else None,'spread':D.number(r.get('spread_'+variable)), 'regime':r.get('regime'),'cell':r.get('cell'), 'rain_probability':D.number(r.get('prob_heavy_rain_day_flag')), 'wind_direction':D.number(r.get('wind_dir')), 'dominant_expert':dominant, 'dominant_weight':weights.get(dominant) if dominant else None}})
    return {'type':'FeatureCollection','features':features,'sample_stride':stride,'total_cells':len(selected),'unit':D.TARGETS[variable][1],'state':boundary['properties']['name'] if boundary else None}


@app.get('/api/verification')
def verification(variable: Literal['tp_mm','t2m_C_mean','wind_speed_mean']='tp_mm', lead: Lead=Lead.day1, season: Literal['winter','pre_monsoon','monsoon','post_monsoon']='post_monsoon'):
    if lead==0:
        raise HTTPException(422,'Benchmark leads are 24, 48 and 72 hours')
    path = D.REPO/'reports/results/blocked/pm_season_lead.csv'
    return {'scope':'2025 held-out day blocks across India; aggregate benchmark, not selected-location verification',
            'rows':[r for r in D.rows(path) if r['target']==variable and int(r['lead_days'])==lead//24 and r['season']==season], 'provenance':D.provenance(path)}


@app.get('/api/observations')
def observations(latitude: float=Query(ge=-90,le=90), longitude: float=Query(ge=-180,le=180), day: date=date(2014,7,1)):
    path = D.ROOT/f'crop-yield-analysis/rainfall_data/RF25_ind{day.year}_rfp25.nc'
    if not path.exists():
        return D.pending(f'Rainfall file for {day.year} is not downloaded. The local sample covers 2014.')
    import xarray as xr
    with xr.open_dataset(path) as ds:
        if not (float(ds.LATITUDE.min())<=latitude<=float(ds.LATITUDE.max()) and float(ds.LONGITUDE.min())<=longitude<=float(ds.LONGITUDE.max())):
            return D.pending('Selected location is outside the rainfall observation grid.')
        cell = ds.sel(LATITUDE=latitude,LONGITUDE=longitude,method='nearest')
        selected = cell.sel(TIME=str(day))
        value = D.number(selected.RAINFALL.item())
        return {'status':'available' if value is not None else 'missing','date':str(day),'value':value,'unit':'mm','latitude':float(cell.LATITUDE),'longitude':float(cell.LONGITUDE),'source':path.name,'message':'Historical daily rainfall. Accumulation window must be confirmed before forecast verification.'}


@app.get('/api/imd')
def imd(state: str='KERALA'):
    return {'source':'IMD supplied statewise CSV','rows':[r for r in D.rows(D.REPO/'rainfall_statewise_daily_imd_clean.csv') if r['state']==state.upper()][-40:]}


@app.post('/api/scenario')
def scenario(q: Scenario):
    ctx = D.context(q.latitude,q.longitude,q.lead,q.source,q.state)
    if ctx['status']!='available':
        raise HTTPException(409,ctx['message'])
    factors = {'tp_mm':1+q.rain_percent/100,'wind_speed_mean':1+q.wind_percent/100,'t2m_C_mean':1}
    values = {k: {'baseline':v['value'],'simulated':v['value']*factors[k]+(q.temperature_delta if k=='t2m_C_mean' else 0),'unit':v['unit']} for k,v in ctx['forecast'].items() if v['value'] is not None}
    return {'label':'SIMULATED SCENARIO — deterministic perturbation only','values':values,'risk':D.pending('Impact cannot be recomputed without an impact model. Forecast probabilities and weights are not recalculated.'),'provenance':ctx['provenance']}


@app.post('/api/weathergpt')
def explain(q: Question):
    from . import nlp
    parsed_query = None
    resolved_location = None
    latitude, longitude, state, lead = q.latitude, q.longitude, q.state, q.lead
    if nlp.query_parser_configured():
        try:
            parser_result = nlp.parse_query(q.question)
            parsed_query = parser_result['parsed']
            parsed_location = parsed_query.get('location')
            if isinstance(parsed_location, str) and D.state_feature(parsed_location):
                state = parsed_location
                center = D.geometry_center(D.state_feature(parsed_location)['geometry'])
                latitude, longitude = center['latitude'], center['longitude']
            coordinates = parsed_query.get('coordinates')
            if isinstance(coordinates, dict) and isinstance(coordinates.get('lat'), (int, float)) and isinstance(coordinates.get('lon'), (int, float)):
                latitude, longitude = coordinates['lat'], coordinates['lon']
                # City and literal coordinates use their nearest grid. A state
                # remains an intentional state aggregation.
                if not (isinstance(parsed_location, str) and D.state_feature(parsed_location)):
                    state = None
            time_range = parsed_query.get('time_range', {})
            parsed_lead = time_range.get('lead_hours') if isinstance(time_range, dict) else None
            if parsed_lead in (0, 24, 48, 72):
                lead = parsed_lead
            grid_cell = parsed_query.get('grid_cell')
            if grid_cell:
                grid = D.cell_location(str(grid_cell), lead, q.source)
                if grid:
                    latitude, longitude, state = grid['latitude'], grid['longitude'], None
                    parsed_query['resolved_grid'] = grid
        except Exception:
            log.exception('Weather-query parser failed; continuing with map selection')

    if parsed_query and parsed_query.get('query_bucket') != 'hydra_model_methodology' and not parsed_query.get('coordinates') and parsed_query.get('location_candidate'):
        try:
            resolved_location = live_weather.resolve_indian_location(parsed_query['location_candidate'])
            if resolved_location:
                latitude, longitude = resolved_location['latitude'], resolved_location['longitude']
                state = None
                parsed_query['resolved_location'] = resolved_location
        except httpx.HTTPError:
            log.info('India-only live geocoder unavailable for %s', parsed_query['location_candidate'])

    if parsed_query and parsed_query.get('query_bucket') == 'hydra_model_methodology':
        return {'answer':methodology_answer(parsed_query.get('explanation_topic', 'forecast_value')),
                'engine':'HYDRA model methodology', 'evidence':[], 'parsed_query':parsed_query}

    if parsed_query and parsed_query.get('question_operation') == 'capability':
        labels = '; '.join(f"{bucket['label']}: {', '.join(bucket['parameters'])}" for bucket in QUERY_CATALOG)
        return {'answer':f"WeatherGPT can answer these supported question groups: {labels}. Ask a question with a location or India/state scope, weather parameter, and time period. Results always state the source and its coverage limits.",
                'engine':'WeatherGPT capability catalog', 'evidence':[], 'parsed_query':parsed_query, 'buckets':QUERY_CATALOG}

    if parsed_query and parsed_query.get('data_domain') == 'historical_observations' and parsed_query.get('weather_variable') == 'rainfall':
        time_range = parsed_query.get('time_range') or {}
        start_date = time_range.get('start_date') if isinstance(time_range, dict) else None
        try:
            year = int(str(start_date)[:4])
        except (TypeError, ValueError):
            return {'answer':'Please specify the calendar year for the historical rainfall observation query.', 'engine':'Historical rainfall observations', 'evidence':[], 'parsed_query':parsed_query}
        observation_state = None if 'india' in q.question.casefold() else state
        if parsed_query.get('question_operation') == 'time_ranking':
            ranking = D.historical_rainfall_month_rankings(year, observation_state, parsed_query.get('ranking_direction', 'descending'))
            if ranking['status'] != 'available':
                return {'answer':ranking['message'], 'engine':'Historical rainfall observations', 'evidence':[], 'parsed_query':parsed_query}
            leading = ranking['items'][0]
            qualifier = ('This reports rainfall only; it cannot establish farming suitability because the supplied data contain no crop, soil, irrigation, terrain, yield, market, or crop-calendar information. '
                         if parsed_query.get('use_case') == 'agriculture' else '')
            adjective = 'highest' if parsed_query.get('ranking_direction', 'descending') != 'ascending' else 'lowest'
            answer = (qualifier + f"{leading['month']} had the {adjective} mean monthly accumulated rainfall across {ranking['grid_cells']} matched 0.25° grid cells in {observation_state or 'India'} during {year}: {leading['rainfall_mean_mm']:.1f} mm per grid cell. " +
                      f"This ranks the monthly mean of each grid cell's accumulated rainfall, using {ranking['observation_days']} supplied daily observations; it is not a countrywide rainfall volume.")
            return {'answer':answer, 'engine':f'{year} rainfall observation analysis', 'evidence':[{'file':ranking['source'], 'method':ranking.get('method', 'Supplied daily rainfall observations')}],
                    'parsed_query':parsed_query, 'ranking':ranking['items'], 'coverage_cells':ranking['grid_cells'],
                    'window_start':ranking['window_start'], 'window_end':ranking['window_end'], 'observation_days':ranking['observation_days']}
        ranking = D.historical_rainfall_rankings(year, observation_state)
        if ranking['status'] != 'available':
            return {'answer':ranking['message'], 'engine':'Historical rainfall observations', 'evidence':[], 'parsed_query':parsed_query}
        entries = []
        for index, item in enumerate(ranking['items'], start=1):
            location = f"{item['state']} grid cell"
            if item['nearby_city']:
                location += f" near {item['nearby_city']}"
            entries.append(f"{index}. {location} ({item['latitude']:.2f}° N, {item['longitude']:.2f}° E) — {item['rainfall_total_mm']:.1f} mm")
        qualification = ('This identifies annual rainfall accumulation only; it cannot establish the best farming location because the supplied data contain no crop, soil, irrigation, terrain, yield, market, or crop-calendar information. '
                         if parsed_query.get('use_case') == 'agriculture' else '')
        answer = (qualification + f"Across the supplied {ranking['observation_days']} daily rainfall observations from {ranking['window_start']} to {ranking['window_end']}, the highest annual accumulated rainfall was:\n" +
                  "\n".join(entries) + " These are 0.25° rainfall grid cells, not city-station measurements.")
        return {'answer':answer, 'engine':f'{year} rainfall observation analysis', 'evidence':[{'file':ranking['source'], 'method':ranking.get('method', 'Supplied daily rainfall observations')}],
                'parsed_query':parsed_query, 'ranking':ranking['items'], 'coverage_cells':ranking['total_cells'],
                'window_start':ranking['window_start'], 'window_end':ranking['window_end'], 'observation_days':ranking['observation_days']}

    if parsed_query and parsed_query.get('use_case') == 'agriculture' and parsed_query.get('query_scope') == 'spatial_ranking':
        rainfall = live_weather.rank_layer('open_meteo_rainfall_24h', state)
        soil = live_weather.rank_layer('open_meteo_soil_moisture', state)
        evidence = []
        if rainfall.get('items'):
            item = rainfall['items'][0]
            evidence.append(f"highest sampled next-24-hour rainfall: {item['city']}, {item['state']} ({item['value']:.2f} mm)")
        if soil.get('items'):
            item = soil['items'][0]
            evidence.append(f"highest sampled surface soil moisture: {item['city']}, {item['state']} ({item['value']:.3f} m³/m³)")
        answer = ("This is an agriculture suitability question. HYDRA can provide weather screening, but it cannot name a best farm location from precipitation alone: no crop, soil-type, irrigation, terrain, yield, market, or crop-calendar dataset was supplied. "
                  + ("Current weather screening: " + "; ".join(evidence) + "." if evidence else "Live rainfall and soil-moisture samples are currently unavailable."))
        return {'answer':answer, 'engine':'Agriculture weather screening', 'evidence':[], 'parsed_query':parsed_query,
                'coverage_locations':max(rainfall.get('sample_count', 0), soil.get('sample_count', 0)),
                'provider':rainfall.get('source') or soil.get('source')}

    if parsed_query and parsed_query.get('data_domain') == 'live_layers' and parsed_query.get('analysis_window_days'):
        layer = parsed_query.get('data_layer') or 'live_weather'
        ranking = live_weather.historical_rank_layer(layer, parsed_query['analysis_window_days'], state)
        if ranking['status'] != 'available':
            return {'answer':ranking['message'], 'engine':'Retained live-layer timeline query', 'evidence':[], 'parsed_query':parsed_query}
        entries = [f"{index}. {item['city']}, {item['state']} — {item['value']:.2f} {ranking['unit']} average across {item['sample_days']} recorded day(s)"
                   for index, item in enumerate(ranking['items'], start=1)]
        answer = (f"{ranking['label'].capitalize()} over retained daily live samples from {ranking['window_start']} to {ranking['window_end']}:\n" +
                  "\n".join(entries) + " Values use only days HYDRA actually recorded; missing dates are not estimated.")
        return {'answer':answer, 'engine':'Retained live-layer timeline query', 'evidence':[], 'parsed_query':parsed_query,
                'ranking':ranking['items'], 'coverage_locations':ranking['sample_count'], 'window_days':ranking['window_days'],
                'window_start':ranking['window_start'], 'window_end':ranking['window_end'], 'provider':ranking['source']}

    if parsed_query and parsed_query.get('analysis_window_days'):
        requested_variable = parsed_query.get('weather_variable')
        target = {'temperature':'t2m_C_mean','wind':'wind_speed_mean','rainfall':'tp_mm'}.get(requested_variable, 'tp_mm')
        ranking = D.long_term_rankings(target, parsed_query.get('extreme_event_type'), parsed_query['analysis_window_days'], state)
        if ranking['status'] != 'available':
            return {'answer':ranking['message'], 'engine':'HYDRA historical-model query', 'evidence':[], 'parsed_query':parsed_query}
        event = ranking['event']
        label = {'heavy_rain':'heavy-rain', 'heatwave':'heatwave', 'high_wind':'high-wind'}.get(event)
        entries = []
        for index, item in enumerate(ranking['items'], start=1):
            area = item['state'] or item['nearby_city'] or f"grid cell {item['cell']}"
            coordinates = f"{item['latitude']:.2f}° N, {item['longitude']:.2f}° E"
            if event:
                entries.append(f"{index}. {area} ({coordinates}) — {item['score']:.2f} expected {label} days; mean daily probability {item['mean_probability'] * 100:.2f}%")
            elif target == 'tp_mm':
                entries.append(f"{index}. {area} ({coordinates}) — {item['score']:.1f} mm predicted rainfall total")
            else:
                unit = '°C' if target == 't2m_C_mean' else 'm/s'
                entries.append(f"{index}. {area} ({coordinates}) — {item['score']:.2f} {unit} window mean")
        qualifier = (' The trained repository has a heavy-rain classifier (daily rainfall ≥64.5 mm); it does not provide a separate very-heavy-rain classifier.'
                     if 'very heavy' in q.question.casefold() else '')
        subject = f"expected {label} days" if event else ("predicted rainfall total" if target == 'tp_mm' else D.TARGETS[target][0].lower())
        answer = (f"Across {ranking['window_days']} model target days ({ranking['window_start']} to {ranking['window_end']}), the repeated HYDRA forecasts rank {ranking['total_cells']} supplied cells by {subject}:\n" +
                  ("\n".join(entries) if entries else 'No scored cells are available for this state/window.') + qualifier)
        return {'answer':answer, 'engine':'HYDRA historical-model query', 'evidence':[ranking['provenance']],
                'parsed_query':parsed_query, 'ranking':ranking['items'], 'coverage_cells':ranking['total_cells'],
                'window_days':ranking['window_days'], 'window_start':ranking['window_start'], 'window_end':ranking['window_end']}

    if parsed_query and parsed_query.get('data_domain') == 'imagery':
        layer = parsed_query.get('data_layer')
        try:
            if layer == 'radar':
                metadata = imagery.radar()
                frames = metadata.get('radar', [])
                message = (f"The live radar precipitation layer has {len(frames)} available frames. It is a visual precipitation mosaic, "
                           "so WeatherGPT can report its availability but does not convert pixels into a city rainfall value.")
                return {'answer':message, 'engine':'Live radar layer metadata', 'evidence':[], 'parsed_query':parsed_query,
                        'updated_at':metadata.get('generated_at'), 'frame_count':len(frames)}
            metadata = imagery.satellite_bounds()
            return {'answer':'The IMD satellite IR layer is a current visual cloud-top image over its published bounds. WeatherGPT does not infer rainfall, lightning, or a city measurement from image pixels.',
                    'engine':'IMD satellite layer metadata', 'evidence':[], 'parsed_query':parsed_query, 'bounds':metadata.get('bounds')}
        except httpx.HTTPError:
            return {'answer':'The requested live imagery layer is currently unavailable.', 'engine':'Live imagery unavailable', 'evidence':[], 'parsed_query':parsed_query}

    if parsed_query and parsed_query.get('weather_summary_request'):
        """Answer generic current-weather requests from one labelled live report.

        HYDRA forecast outputs remain available only for explicit HYDRA/model
        requests.  A request like "weather today in Kerala" has no model
        variable or forecast lead and is therefore a live conditions summary.
        """
        location_name = (resolved_location or {}).get('name') or (
            parsed_query.get('location') if isinstance(parsed_query.get('location'), str) else None
        )
        report = live_weather.current_report(state or location_name, latitude, longitude)
        if report['status'] != 'available':
            return {'answer':report['message'], 'engine':'Live weather summary', 'evidence':[], 'parsed_query':parsed_query}
        observation = report['observation']
        if resolved_location and resolved_location.get('state'):
            observation = {**observation, 'state':resolved_location['state']}
        time_range = parsed_query.get('time_range') or {}
        lead_hours = time_range.get('lead_hours') if isinstance(time_range, dict) else None
        place = f"{observation['city']}, {observation['state']}"
        if lead_hours in (24, 48):
            outlook = observation.get(f'outlook_{lead_hours}h')
            if not isinstance(outlook, dict):
                return {'answer':'The live provider did not return a complete weather outlook for that period.',
                        'engine':'Live weather summary', 'evidence':[], 'parsed_query':parsed_query}
            values = []
            if outlook.get('temperature_mean_C') is not None:
                values.append(f"mean temperature {outlook['temperature_mean_C']:.1f} °C")
            if outlook.get('rainfall_total_mm') is not None:
                values.append(f"rainfall {outlook['rainfall_total_mm']:.2f} mm")
            if outlook.get('humidity_mean_percent') is not None:
                values.append(f"mean humidity {outlook['humidity_mean_percent']:.0f}%")
            if outlook.get('precipitation_probability_max_percent') is not None:
                values.append(f"maximum rain probability {outlook['precipitation_probability_max_percent']:.0f}%")
            if outlook.get('wind_gust_max_kmh') is not None:
                values.append(f"peak wind gust {outlook['wind_gust_max_kmh']:.1f} km/h")
            answer = (f"Forecast weather for {place} for hours {lead_hours - 24}–{lead_hours}: " +
                      ("; ".join(values) if values else 'the provider returned no summary values') +
                      ".\nSource: live Open-Meteo provider outlook at the disclosed representative coordinate; this is not a HYDRA forecast.")
        else:
            values = []
            fields = (
                ('temperature_max_C', 'temperature', '°C', 1),
                ('rainfall_mm', 'rainfall', 'mm', 2),
                ('humidity_percent', 'relative humidity', '%', 0),
                ('wind_speed_kmh', 'wind speed', 'km/h', 1),
                ('cloud_cover_percent', 'cloud cover', '%', 0),
                ('pressure_hpa', 'surface pressure', 'hPa', 0),
            )
            for field, label, unit, precision in fields:
                value = observation.get(field)
                if isinstance(value, (int, float)):
                    values.append(f"{label} {value:.{precision}f} {unit}")
            description = observation.get('description')
            description_text = f" Conditions: {description}." if description else ''
            answer = (f"Current weather for {place}: " +
                      ("; ".join(values) if values else 'no numeric conditions were returned') + "." + description_text +
                      "\nSource: live provider value at the disclosed representative coordinate; it is not a state-wide average or a HYDRA forecast.")
        return {'answer':answer, 'engine':'Live IndianAPI / Open-Meteo weather summary', 'evidence':[], 'parsed_query':parsed_query,
                'updated_at':report.get('updated_at'), 'provider':report.get('source'), 'observation':observation}

    if parsed_query and parsed_query.get('data_domain') == 'live_layers':
        layer = parsed_query.get('data_layer') or 'live_weather'
        time_range = parsed_query.get('time_range') or {}
        window_start = window_end = None
        if isinstance(time_range, dict) and time_range.get('kind') == 'date_range':
            try:
                window_start = date.fromisoformat(str(time_range.get('start_date')))
                window_end = date.fromisoformat(str(time_range.get('end_date')))
                if window_start > window_end or window_end >= date.today():
                    window_start = window_end = None
            except ValueError:
                window_start = window_end = None
        if parsed_query.get('query_scope') == 'spatial_ranking':
            state_ranking = parsed_query.get('ranking_scope') == 'states'
            past_date = window_start if window_start and window_start == window_end else None
            # An explicit state comparison must not inherit the map's selected
            # state as a filter.
            ranking = (
                live_weather.rank_state_layer_for_day(layer, past_date)
                if state_ranking and past_date is not None and time_range.get('text') == 'yesterday'
                else live_weather.rank_state_layer_for_window(layer, window_start, window_end)
                if state_ranking and window_start and window_end
                else live_weather.rank_state_layer(layer)
                if state_ranking
                else live_weather.rank_location_layer_for_window(layer, window_start, window_end, state)
                if window_start and window_end
                else live_weather.rank_layer(layer, state)
            )
            if ranking['status'] != 'available':
                return {'answer':ranking['message'], 'engine':'Live layer query', 'evidence':[], 'parsed_query':parsed_query}
            if not ranking['items']:
                answer = f"No live {ranking['label']} samples are available for this query."
            elif state_ranking:
                winner = ranking['items'][0]
                runners_up = ranking['items'][1:3]
                day_phrase = (
                    f"On {ranking['date']}" if ranking.get('date') else
                    f"From {ranking['window_start']} to {ranking['window_end']}" if ranking.get('window_start') else
                    "Today"
                )
                answer = (
                    f"Result: {day_phrase}, {winner['state']} has the highest {ranking['label']} among the "
                    f"{ranking['sample_count']} India state/UT representative readings: {winner['value']:.2f} "
                    f"{ranking['unit']} at {winner['latitude']:.2f} N, {winner['longitude']:.2f} E."
                )
                if runners_up:
                    answer += "\nNext readings: " + "; ".join(
                        f"{item['state']} ({item['value']:.2f} {ranking['unit']})" for item in runners_up
                    ) + "."
                if ranking.get('calculation'):
                    answer += f"\nCalculation: {ranking['calculation']}."
                answer += ("\nCoverage: one disclosed Open-Meteo representative coordinate per state/UT; "
                           "this is not a statewide spatial average.")
            else:
                entries = [f"{index}. {item['city']}, {item['state']} ({item['latitude']:.2f}° N, {item['longitude']:.2f}° E) — {item['value']:.2f} {ranking['unit']}"
                           for index, item in enumerate(ranking['items'], start=1)]
                answer = (f"{ranking['label'].capitalize()} across {ranking['sample_count']} available Indian map locations:\n" + "\n".join(entries) +
                          " These are provider locations, not an interpolated India-wide field.")
            return {'answer':answer, 'engine':'Live IndianAPI / Open-Meteo layer query', 'evidence':[], 'parsed_query':parsed_query,
                    'ranking':ranking['items'], 'coverage_locations':ranking['sample_count'], 'updated_at':ranking.get('updated_at'),
                    'provider':ranking.get('source')}
        location_name = (resolved_location or {}).get('name') or (parsed_query.get('location') if isinstance(parsed_query.get('location'), str) else None)
        current = (
            live_weather.layer_value_for_window(layer, latitude, longitude, state or location_name, window_start, window_end)
            if window_start and window_end
            else live_weather.layer_value(layer, state, latitude, longitude, location_name)
        )
        if current['status'] != 'available':
            return {'answer':current['message'], 'engine':'Live layer query', 'evidence':[], 'parsed_query':parsed_query}
        observation = current['observation']
        if resolved_location and resolved_location.get('state'):
            observation = {**observation, 'state':resolved_location['state']}
        period = (f" from {current['window_start']} to {current['window_end']}" if current.get('window_start') else "")
        calculation = (f"\nCalculation: {current['calculation']}." if current.get('calculation') else "")
        answer = (f"Result: {current['label'].title()}{period} at {observation['city']}, {observation['state']} "
                  f"is {current['value']:.2f} {current['unit']}." + calculation +
                  "\nCoverage: this is a provider value at the disclosed location, not an interpolated grid estimate.")
        return {'answer':answer, 'engine':'Live IndianAPI / Open-Meteo layer query', 'evidence':[], 'parsed_query':parsed_query,
                'updated_at':current.get('updated_at'), 'provider':current.get('source')}

    if parsed_query and parsed_query.get('query_scope') == 'spatial_ranking':
        requested_variable = parsed_query.get('weather_variable')
        target = {'temperature':'t2m_C_mean','wind':'wind_speed_mean','rainfall':'tp_mm'}.get(requested_variable, 'tp_mm')
        ranking = D.forecast_rankings(q.source, lead, target, parsed_query.get('extreme_event_type'), state)
        if ranking['status'] != 'available':
            return {'answer':ranking['message'], 'engine':'HYDRA trained-output query', 'evidence':[], 'parsed_query':parsed_query}
        event = ranking['event']
        display = {'heavy_rain':'heavy-rain', 'heatwave':'heatwave', 'high_wind':'high-wind'}.get(event)
        if not ranking['items']:
            answer = f"The supplied {q.source} output has no scored cells for this query and lead."
        else:
            entries = []
            for index, item in enumerate(ranking['items'], start=1):
                area = item['state'] or item['nearby_city']
                place = f"{area} · grid cell {item['cell']}" if area else f"grid cell {item['cell']} (outside supplied India state/UT boundaries)"
                coordinates = f"{item['latitude']:.2f}° N, {item['longitude']:.2f}° E"
                if event:
                    forecast = f"; predicted rainfall {item['value']:.2f} mm/day" if target == 'tp_mm' and item['value'] is not None else ''
                    entries.append(f"{index}. {place} ({coordinates}) — {item['score'] * 100:.2f}% {display} probability{forecast}")
                else:
                    entries.append(f"{index}. {place} ({coordinates}) — {item['score']:.2f} {ranking['unit']}")
            for entry_index, item in enumerate(ranking['items']):
                dominant = item.get('dominant_expert')
                if dominant:
                    evidence_unit = D.TARGETS[item['evidence_variable']][1]
                    expert_inputs = ', '.join(
                        f"{expert['name']} {expert['value']:.2f} {evidence_unit} at {expert['weight']:.1%}"
                        for expert in item['experts'] if expert['value'] is not None
                    )
                    interval = item.get('interval80')
                    interval_text = (f"; saved 80% interval {interval[0]:.2f}-{interval[1]:.2f} {evidence_unit}"
                                     if interval and None not in interval else '')
                    spread_text = (f"expert spread {item['spread']:.2f} {evidence_unit}" if item['spread'] is not None
                                   else "expert spread unavailable")
                    entries[entry_index] += (f"\n   Model evidence: dominant expert {dominant['name']} ({dominant['weight']:.1%}); "
                                             f"expert inputs/weights: {expert_inputs}; regime {item['regime']}; {spread_text}{interval_text}.")
                else:
                    entries[entry_index] += "\n   Model evidence is unavailable because this source is an analysis field, not saved HYDRA blend output."
            framing = (f"For {ranking['valid_date']}, the trained HYDRA classifier ranks the {ranking['total_cells']} supplied forecast cells by {display} probability:\n"
                       if event else f"For {ranking['valid_date']}, the trained HYDRA blend ranks the {ranking['total_cells']} supplied forecast cells by {D.TARGETS[target][0].lower()}:\n")
            qualifier = ("\nCoverage: this is a rank within the supplied forecast cells, not an all-India result. "
                         "Expert spread is disagreement, not calibrated confidence; learned weights are blend allocation, not causal explanation.")
            answer = framing + "\n".join(entries) + qualifier
        return {'answer':answer, 'engine':'HYDRA trained-output query', 'evidence':[ranking['provenance']],
                'valid_date':ranking['valid_date'], 'parsed_query':parsed_query, 'ranking':ranking['items'],
                'coverage_cells':ranking['total_cells']}

    ctx = D.context(latitude,longitude,lead,q.source,state)
    if ctx['status']!='available':
        if parsed_query and parsed_query.get('model_evidence_request'):
            requested = parsed_query.get('location') or (f"grid cell {parsed_query['grid_cell']}" if parsed_query.get('grid_cell') else None)
            return {'answer':f"No saved HYDRA weight distribution is available for {requested or 'this request'} at +{lead}h. {ctx['message']} "
                            "Live-provider and ECMWF analysis values are not HYDRA expert weights.",
                    'engine':'HYDRA adaptive-blend weight evidence unavailable', 'evidence':[], 'parsed_query':parsed_query}
        return {'answer':ctx['message'],'engine':'Structured evidence explainer','evidence':[], 'parsed_query':parsed_query}
    if parsed_query and parsed_query.get('model_evidence_request'):
        return {'answer':weight_distribution_answer(ctx, parsed_query.get('location')),
                'engine':'HYDRA adaptive-blend weight evidence', 'evidence':[ctx['provenance']],
                'valid_date':ctx['valid_date'], 'parsed_query':parsed_query}
    if nlp.configured():
        try:
            context = {**ctx, 'query_intent': parsed_query} if parsed_query else ctx
            response = nlp.answer(q.question, context)
            return {**response, 'evidence':[ctx['provenance']], 'valid_date':ctx['valid_date'], 'parsed_query':parsed_query}
        except Exception as exc:
            log.exception('Configured NLP service failed')
            return {'answer':'The configured NLP service is unavailable. '+str(exc), 'engine':'NLP adapter error',
                    'evidence':[ctx['provenance']], 'valid_date':ctx['valid_date'], 'parsed_query':parsed_query}
    text = q.question.lower()
    if any(w in text for w in ['flood','risk','impact']):
        answer = ctx['risk']['message']+' Rainfall alone does not establish flood risk.'
    elif any(w in text for w in ['aifs','gfs','gefs']):
        answer = 'These models are not trained experts in the supplied checkpoint. The trained experts are '+', '.join(D.MANIFEST['experts'])+'.'
    elif any(w in text for w in ['change','ago','6 hour']):
        answer = 'No comparable prior issue cycle was supplied, so a change since the last cycle cannot be established.'
    else:
        requested_variable = parsed_query.get('weather_variable') if parsed_query else None
        target = {'temperature':'t2m_C_mean','wind':'wind_speed_mean','rainfall':'tp_mm'}.get(requested_variable)
        target = target or ('t2m_C_mean' if 'temperature' in text else 'wind_speed_mean' if 'wind' in text else 'tp_mm')
        f = ctx['forecast'].get(target)
        if not f:
            answer = 'This variable is unavailable for the selected source.'
        else:
            answer = f"Forecast\n{ctx['valid_date']} · {D.TARGETS[target][0]}: {f['value']:.2f} {f['unit']}.\n\n"
            if f['experts']:
                winner=max(f['experts'],key=lambda e:e['weight'])
                expert_inputs = ', '.join(f"{expert['name']} {expert['value']:.2f} {f['unit']} at {expert['weight']:.1%}"
                                          for expert in f['experts'] if expert['value'] is not None)
                answer += f"Model evidence\nSaved expert inputs and learned allocations: {expert_inputs}.\n"
                answer += f"Dominant expert: {winner['name']} ({winner['weight']:.1%}).\n"
                answer += f"Rule-based regime: {ctx['regime']}.\n"
                answer += f"Expert spread: {f['spread']:.2f} {f['unit']}.\n"
                answer += f"Saved 80% interval: {f['interval80'][0]:.2f}–{f['interval80'][1]:.2f} {f['unit']}.\n\n"
                answer += "How to interpret this\nThe weights show blend allocation. Expert spread measures disagreement, not calibrated confidence, and the weights do not establish a causal explanation of the regime gate."
    return {'answer':answer,'engine':'Structured evidence explainer · no LLM configured','evidence':[ctx['provenance']], 'valid_date':ctx['valid_date'], 'parsed_query':parsed_query}


dist = D.ROOT/'frontend/dist'
if dist.exists():
    app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')

import json
import math
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend import data as D

client=TestClient(app)


def selection():
    p=D.locations()[0]
    return {'latitude':p['latitude'],'longitude':p['longitude'],'lead':24,'source':'archive'}


def test_archive_weights_and_blend_are_actual_csv():
    response=client.get('/api/forecast',params=selection())
    assert response.status_code==200
    data=response.json()
    assert data['status']=='available'
    assert data['issue_date']=='2025-12-28'
    assert data['valid_date']=='2025-12-29'
    for target,value in data['forecast'].items():
        assert sum(e['weight'] for e in value['experts'])==pytest.approx(1,abs=1e-5)
        assert sum(e['weight']*e['value'] for e in value['experts'])==pytest.approx(value['value'],abs=1e-4)
        assert value['interval80'][0]<=value['interval80'][1]
    assert len(data['provenance']['sha256'])==64
    assert data['confidence'] is None


def test_no_spatial_fabrication():
    data=client.get('/api/forecast',params={**selection(),'latitude':9.93,'longitude':76.27}).json()
    assert data['status']=='unavailable'
    assert data['forecast']=={}


def test_coverage_never_claims_ocean_and_finds_archive_cell():
    ocean=client.get('/api/coverage',params={'latitude':12.21,'longitude':67.48}).json()
    assert ocean=={'has_data':False,'sources':[]}
    point=D.locations()[0]
    covered=client.get('/api/coverage',params=point).json()
    assert covered['has_data'] is True
    assert 'archive' in covered['sources']


def test_time_context_and_units():
    a=client.get('/api/forecast',params=selection()).json()
    b=client.get('/api/forecast',params={**selection(),'lead':48}).json()
    assert a['valid_date']!=b['valid_date']
    assert a['forecast']['wind_speed_mean']['unit']=='m/s'
    assert client.get('/api/forecast',params={**selection(),'lead':6}).status_code==422
    assert client.get('/api/forecast',params={**selection(),'latitude':91}).status_code==422


def test_production_does_not_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(D,'RUNTIME',tmp_path)
    data=client.get('/api/forecast',params={**selection(),'source':'production'}).json()
    assert data['status']=='unavailable'
    assert data['forecast']=={}


def test_ecmwf_analysis_semantics_and_field_limit():
    data=client.get('/api/forecast',params={'latitude':10,'longitude':76.25,'source':'ecmwf','lead':0}).json()
    assert data['status']=='available'
    assert 'tp_mm' not in data['forecast']
    assert data['forecast']['t2m_C_mean']['experts']==[]
    field=client.get('/api/field',params={'source':'ecmwf','lead':0,'variable':'t2m_C_mean'}).json()
    assert len(field['features'])==field['total_cells']
    assert 1000 < field['total_cells'] < 14157


def test_live_weather_endpoint_keeps_key_server_side_and_caches_reports(monkeypatch, tmp_path):
    from backend import live_weather
    monkeypatch.setattr(live_weather, 'DAILY_HISTORY_PATH', tmp_path / 'live_weather_daily.csv')
    def open_meteo(stations):
        return [{'city':city,'state':state,'latitude':latitude,'longitude':longitude,
                 'temperature_max_C':29.0,'temperature_min_C':29.0,'rainfall_mm':4.0,
                 'humidity_percent':72.0,'description':'Cloudy','forecast_date':'2026-09-29T12:00',
                 'wind_speed_kmh':18.0,'wind_direction_deg':180.0,'cloud_cover_percent':70.0,
                 'pressure_hpa':1004.0,'open_meteo_rainfall_mm':4.0,
                 'open_meteo_humidity_percent':72.0,'open_meteo_rainfall_24h_mm':18.5,
                 'open_meteo_precipitation_probability_max':80.0,'open_meteo_cape_max_Jkg':1200.0,
                 'open_meteo_lightning_watch_percent':61.4,'open_meteo_wind_gust_max_kmh':44.0,
                 'open_meteo_wet_bulb_max_C':29.0,'open_meteo_soil_moisture_0_to_1cm':0.23,
                 'provider':'Open-Meteo weather model'}
                for city,state,latitude,longitude in stations]
    live_weather.clear_cache()
    monkeypatch.delenv('INDIAN_WEATHER_API_KEY', raising=False)
    monkeypatch.setattr(live_weather,'fetch_open_meteo_stations',open_meteo)
    fallback=client.get('/api/live-weather').json()
    assert fallback['status']=='available'
    assert fallback['source']=='Open-Meteo current weather model'
    assert fallback['stations'][0]['humidity_percent']==72.0
    assert fallback['stations'][0]['open_meteo_rainfall_mm']==4.0
    assert fallback['stations'][0]['open_meteo_humidity_percent']==72.0
    assert fallback['stations'][0]['open_meteo_rainfall_24h_mm']==18.5
    assert fallback['stations'][0]['open_meteo_cape_max_Jkg']==1200.0
    assert fallback['stations'][0]['open_meteo_lightning_watch_percent']==61.4
    assert fallback['stations'][0]['open_meteo_wet_bulb_max_C']==29.0

    monkeypatch.setenv('INDIAN_WEATHER_API_KEY','test-key')
    monkeypatch.setattr(live_weather,'STATIONS',(('Test city','Test State',20.0,80.0),))
    calls=[]
    def fake_station(_, station):
        calls.append(station)
        return {'city':'Test city','state':'Test State','latitude':20.0,'longitude':80.0,
                'temperature_max_C':31.0,'temperature_min_C':21.0,'rainfall_mm':4.0,
                'humidity_percent':72.0,'description':'Cloudy','forecast_date':'01-Jan-2026',
                'wind_speed_kmh':None,'wind_direction_deg':None,'cloud_cover_percent':None,
                'pressure_hpa':None,'provider':'IndianAPI / IMD city report'}
    monkeypatch.setattr(live_weather,'fetch_station',fake_station)
    live_weather.clear_cache()
    first=client.get('/api/live-weather').json()
    second=client.get('/api/live-weather').json()
    assert first['status']=='available'
    assert first['stations'][0]['city']=='Test city'
    assert first['stations'][0]['wind_speed_kmh']==18.0
    assert 'Open-Meteo enrichment' in first['stations'][0]['provider']
    assert first['cached'] is False and second['cached'] is True
    assert len(calls)==1
    live_weather.clear_cache()


def test_current_observation_uses_open_meteo_when_imd_is_not_configured(monkeypatch):
    from backend import live_weather
    monkeypatch.delenv('INDIAN_WEATHER_API_KEY', raising=False)
    monkeypatch.setattr(live_weather,'fetch_open_meteo_stations',lambda stations:[{
        'city':city,'state':state,'latitude':latitude,'longitude':longitude,
        'temperature_max_C':28.0,'temperature_min_C':28.0,'rainfall_mm':0.2,
        'humidity_percent':68.0,'description':'Partly cloudy','forecast_date':'2026-09-29T12:00',
        'wind_speed_kmh':12.0,'wind_direction_deg':160.0,'cloud_cover_percent':35.0,
        'pressure_hpa':1008.0,'provider':'Open-Meteo weather model'}
        for city,state,latitude,longitude in stations])
    live_weather.clear_cache()
    data=client.get('/api/current-observation',params={'latitude':27.2,'longitude':80.9,'state':'Uttar Pradesh'}).json()
    assert data['status']=='available'
    assert data['source']=='Open-Meteo weather model'
    assert data['observation']['pressure_hpa']==1008.0


def test_state_outlook_exposes_two_provider_windows_and_hydra_limit(monkeypatch):
    from backend import live_weather
    monkeypatch.setattr(D, 'hydra_state_outlook', lambda state: {
        'status':'awaiting_inference', 'state':state, 'outlooks':[],
        'message':'No fresh HYDRA forecast cycle has been published.',
        'missing_inputs':['full-grid ERA5 daily history', 'climatology_by_doy.npz'],
    })
    monkeypatch.setattr(live_weather, 'current_report', lambda state, latitude, longitude: {
        'status':'available', 'source':'Open-Meteo weather model', 'updated_at':'2026-09-29T00:00:00Z',
        'observation':{'city':'Thiruvananthapuram', 'state':state, 'latitude':latitude, 'longitude':longitude,
                       'outlook_24h':{'lead_hours':24, 'temperature_mean_C':29.0, 'rainfall_total_mm':5.0, 'humidity_mean_percent':78.0, 'precipitation_probability_max_percent':80.0, 'wind_gust_max_kmh':24.0},
                       'outlook_48h':{'lead_hours':48, 'temperature_mean_C':28.0, 'rainfall_total_mm':8.0, 'humidity_mean_percent':81.0, 'precipitation_probability_max_percent':85.0, 'wind_gust_max_kmh':28.0}}
    })
    data = client.get('/api/state-outlook', params={'state':'Kerala','latitude':10.5,'longitude':76.5}).json()
    assert data['hydra']['status'] == 'awaiting_inference'
    assert data['hydra']['missing_inputs'] == ['full-grid ERA5 daily history', 'climatology_by_doy.npz']
    assert [item['lead_hours'] for item in data['provider_comparison']['outlooks']] == [24, 48]


def test_retrained_hydra_daily_mean_cycle_is_available_to_state_insights(monkeypatch, tmp_path):
    artifact = tmp_path / 'hydra_daily_mean_state_blend.json'
    artifact.write_text(json.dumps({
        'kind':'hydra_daily_mean_neural_gating', 'issue_date':'2025-12-31', 'message':'Neural gate.',
        'states':{'Kerala':{'nearest_grid_fallback':False, 'outlooks':[{'lead_hours':24}, {'lead_hours':48}]}}
    }))
    monkeypatch.setattr(D, 'HYDRA_DAILY_MEAN_BLEND', artifact)
    result = D.daily_mean_hydra_state_outlook('Kerala')
    assert result['status'] == 'available'
    assert result['source'] == 'HYDRA daily-mean neural-gating adaptive blend'
    assert [item['lead_hours'] for item in result['outlooks']] == [24, 48]


def test_every_india_state_has_a_selectable_boundary_and_state_summary():
    catalog=client.get('/api/catalog').json()
    assert len(catalog['states'])==36
    data=client.get('/api/forecast',params={'latitude':10.5,'longitude':76.5,'state':'Kerala','source':'ecmwf','lead':0}).json()
    assert data['status']=='available'
    assert data['scope']=='state'
    assert data['grid_cell_count']>1
    assert data['forecast']['t2m_C_mean']['minimum']<=data['forecast']['t2m_C_mean']['value']<=data['forecast']['t2m_C_mean']['maximum']
    assert data['observations']['source']=='IMD supplied statewise CSV'


def test_ecmwf_field_can_be_limited_to_selected_state_geometry():
    all_india=client.get('/api/field',params={'source':'ecmwf','lead':0,'variable':'t2m_C_mean'}).json()
    kerala=client.get('/api/field',params={'source':'ecmwf','lead':0,'variable':'t2m_C_mean','state':'Kerala'}).json()
    assert kerala['state']=='Kerala'
    assert 0 < kerala['total_cells'] < all_india['total_cells']
    assert all(D.point_in_geometry(*feature['geometry']['coordinates'],D.state_feature('Kerala')['geometry']) for feature in kerala['features'])
    assert client.get('/api/field',params={'source':'ecmwf','lead':0,'state':'Not a state'}).status_code==404


def test_scenario_is_labelled_and_does_not_invent_risk():
    base=client.get('/api/forecast',params=selection()).json()
    data=client.post('/api/scenario',json={**selection(),'rain_percent':20}).json()
    assert data['label'].startswith('SIMULATED')
    assert data['values']['tp_mm']['simulated']==pytest.approx(base['forecast']['tp_mm']['value']*1.2)
    assert data['risk']['status']=='integration_pending'


def test_weather_explanation_rejects_unsupported_experts():
    data=client.post('/api/weathergpt',json={**selection(),'question':'Why trust AIFS?'}).json()
    assert 'not trained experts' in data['answer']
    assert data['evidence'][0]['sha256']


def test_weather_explanation_accepts_state_context():
    data=client.post('/api/weathergpt',json={'latitude':10.5,'longitude':76.5,'state':'Kerala','source':'ecmwf','lead':0,'question':'Explain temperature'}).json()
    assert data['valid_date']=='2026-09-24'
    assert data['evidence'][0]['file'].endswith('forecast_india_2026-09-24_0h_clean.csv')


def test_weight_distribution_question_uses_saved_hydra_experts():
    data = client.post('/api/weathergpt', json={**selection(), 'question':'Show HYDRA weight distribution for grid cell 12000 tomorrow'}).json()
    assert data['engine'] == 'HYDRA adaptive-blend weight evidence'
    assert 'HYDRA adaptive-blend weight distribution for grid cell 12000' in data['answer']
    assert 'Rainfall:' in data['answer']
    assert 'Total allocation 100.0%' in data['answer']
    assert data['parsed_query']['resolved_grid']['cell'] == '12000'


def test_weight_distribution_does_not_substitute_provider_values_for_a_state():
    data = client.post('/api/weathergpt', json={**selection(), 'question':'Show HYDRA weight distribution for Maharashtra tomorrow'}).json()
    assert data['engine'] == 'HYDRA adaptive-blend weight evidence unavailable'
    assert 'Live-provider and ECMWF analysis values are not HYDRA expert weights.' in data['answer']


def test_configured_nlp_adapter_receives_grounded_context(monkeypatch):
    from backend import nlp
    captured={}
    monkeypatch.setattr(nlp,'configured',lambda:True)
    def answer(question,context):
        captured.update(question=question,context=context)
        return {'answer':'Grounded service response','engine':'test-nlp'}
    monkeypatch.setattr(nlp,'answer',answer)
    response=client.post('/api/weathergpt',json={**selection(),'question':'Explain this forecast'}).json()
    assert response['answer']=='Grounded service response'
    assert captured['context']['valid_date']=='2025-12-29'
    assert captured['context']['forecast']['tp_mm']['experts']


def test_model_methodology_question_uses_documented_logic(monkeypatch):
    from backend import nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'query_bucket':'hydra_model_methodology', 'question_operation':'explanation', 'explanation_topic':'weights',
        'time_range':{'kind':'unspecified'},
    }})
    data = client.post('/api/weathergpt', json={**selection(), 'question':'What do expert weights mean?'}).json()
    assert data['engine'] == 'HYDRA model methodology'
    assert 'allocation' in data['answer']


def test_weather_query_parser_guides_weathergpt_context(monkeypatch):
    from backend import nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location': None, 'location_suggestion': None,
        'coordinates': None, 'weather_variable': 'wind',
        'time_range': {'kind': 'lead_hours', 'lead_hours': 48, 'text': 'in 2 days'},
        'extreme_event_type': 'high_wind',
    }})
    data = client.post('/api/weathergpt', json={**selection(), 'question': 'high winds at 9, 76 in 2 days'}).json()
    assert data['parsed_query']['weather_variable'] == 'wind'
    assert data['valid_date'] == '2025-12-30'
    assert 'wind speed' in data['answer'].lower()


def test_trained_output_ranking_answers_where_questions(monkeypatch):
    from backend import nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location': None, 'location_suggestion': None, 'coordinates': None,
        'weather_variable': 'rainfall', 'time_range': {'kind': 'lead_hours', 'lead_hours': 24},
        'extreme_event_type': 'heavy_rain', 'query_scope': 'spatial_ranking',
    }})
    data = client.post('/api/weathergpt', json={**selection(), 'question': 'Name places with high prone to heavy rains'}).json()
    assert data['engine'] == 'HYDRA trained-output query'
    assert data['coverage_cells'] == 3
    assert len(data['ranking']) == 3
    assert data['ranking'][0]['score'] >= data['ranking'][1]['score']
    assert 'state' in data['ranking'][0]
    assert 'nearby_city' in data['ranking'][0]
    assert 'classifier ranks' in data['answer']
    assert data['ranking'][0]['dominant_expert']['weight'] is not None
    assert 'Model evidence: dominant expert' in data['answer']


def test_live_layer_ranking_answers_today_questions(monkeypatch):
    from backend import nlp, live_weather
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location': None, 'location_suggestion': None, 'coordinates': None,
        'weather_variable': 'temperature', 'data_layer': 'live_weather', 'data_domain': 'live_layers',
        'time_range': {'kind': 'lead_hours', 'lead_hours': 0, 'text': 'today'}, 'extreme_event_type': None,
        'query_scope': 'spatial_ranking',
    }})
    monkeypatch.setattr(live_weather, 'rank_layer', lambda layer, state: {
        'status':'available', 'items':[{'city':'Chennai','state':'Tamil Nadu','latitude':13.0827,'longitude':80.2707,'value':33.2}],
        'sample_count':1, 'label':'current temperature', 'unit':'°C', 'source':'Open-Meteo weather model', 'updated_at':'2026-09-29T00:00:00Z',
    })
    data = client.post('/api/weathergpt', json={**selection(), 'question': 'where is the most temperature today'}).json()
    assert data['engine'] == 'Live IndianAPI / Open-Meteo layer query'
    assert 'Chennai, Tamil Nadu' in data['answer']
    assert data['coverage_locations'] == 1


def test_state_ranking_does_not_inherit_the_selected_map_state(monkeypatch):
    from backend import nlp, live_weather
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location': None, 'location_suggestion': None, 'coordinates': None,
        'weather_variable': 'temperature', 'data_layer': 'live_weather', 'data_domain': 'live_layers',
        'time_range': {'kind': 'lead_hours', 'lead_hours': 0, 'text': 'today'}, 'extreme_event_type': None,
        'query_scope': 'spatial_ranking', 'ranking_scope': 'states',
    }})
    monkeypatch.setattr(live_weather, 'rank_state_layer', lambda layer: {
        'status':'available',
        'items':[{'city':'state representative point','state':'Rajasthan','latitude':26.8,'longitude':73.1,'value':39.2},
                 {'city':'state representative point','state':'Gujarat','latitude':22.3,'longitude':71.2,'value':37.8}],
        'sample_count':36, 'label':'current temperature', 'unit':'°C',
        'source':'Open-Meteo weather model', 'updated_at':'2026-09-30T00:00:00Z',
    })
    data = client.post('/api/weathergpt', json={**selection(), 'state':'Maharashtra',
                                                  'question':'Which state had the highest temperature today?'}).json()
    assert data['engine'] == 'Live IndianAPI / Open-Meteo layer query'
    assert 'Today, Rajasthan has the highest current temperature' in data['answer']
    assert '36 India state/UT representative readings' in data['answer']


def test_yesterday_state_ranking_uses_past_day_provider_values(monkeypatch):
    from backend import nlp, live_weather
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'weather_variable': 'temperature', 'data_layer': 'live_weather', 'data_domain': 'live_layers',
        'time_range': {'kind': 'date_range', 'start_date': '2026-09-29', 'end_date': '2026-09-29', 'text': 'yesterday'},
        'query_scope': 'spatial_ranking', 'ranking_scope': 'states',
    }})
    monkeypatch.setattr(live_weather, 'rank_state_layer_for_day', lambda layer, day: {
        'status':'available', 'items':[{'state':'Gujarat','latitude':22.3,'longitude':71.2,'value':37.8}],
        'sample_count':36, 'label':'daily maximum temperature', 'unit':'°C', 'date':'2026-09-29',
        'source':'Open-Meteo past-day weather model series',
    })
    data = client.post('/api/weathergpt', json={**selection(), 'question':'Which state had the highest temperature yesterday?'}).json()
    assert data['engine'] == 'Live IndianAPI / Open-Meteo layer query'
    assert 'On 2026-09-29, Gujarat has the highest daily maximum temperature' in data['answer']
    assert data['provider'] == 'Open-Meteo past-day weather model series'


def test_state_duration_ranking_uses_exact_window_and_calculation(monkeypatch):
    from backend import nlp, live_weather
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'weather_variable': 'rainfall', 'data_layer': 'open_meteo_rainfall', 'data_domain': 'live_layers',
        'time_range': {'kind': 'date_range', 'start_date': '2026-09-10', 'end_date': '2026-09-15'},
        'query_scope': 'spatial_ranking', 'ranking_scope': 'states',
    }})
    captured = {}
    def rank(layer, start, end):
        captured.update(layer=layer, start=start, end=end)
        return {
            'status':'available', 'items':[{'state':'Kerala','latitude':10.4,'longitude':76.4,'value':112.4}],
            'sample_count':36, 'label':'accumulated rainfall', 'unit':'mm',
            'window_start':'2026-09-10', 'window_end':'2026-09-15',
            'calculation':'sum of daily precipitation', 'source':'Open-Meteo Historical Weather API',
        }
    monkeypatch.setattr(live_weather, 'rank_state_layer_for_window', rank)
    data = client.post('/api/weathergpt', json={**selection(), 'question':'Which state had the highest rainfall from 2026-09-10 to 2026-09-15?'}).json()
    assert captured['layer'] == 'open_meteo_rainfall'
    assert captured['start'].isoformat() == '2026-09-10'
    assert 'From 2026-09-10 to 2026-09-15, Kerala has the highest accumulated rainfall' in data['answer']
    assert 'Calculation: sum of daily precipitation.' in data['answer']


def test_dated_state_point_query_uses_the_parsed_state_location(monkeypatch):
    from backend import nlp, live_weather
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location':'Kerala', 'weather_variable':'rainfall', 'data_layer':'open_meteo_rainfall', 'data_domain':'live_layers',
        'time_range': {'kind':'date_range', 'start_date':'2026-09-20', 'end_date':'2026-09-20'},
        'query_scope':'selected_location',
    }})
    captured = {}
    def value(layer, latitude, longitude, state, start, end):
        captured.update(layer=layer, latitude=latitude, longitude=longitude, state=state, start=start, end=end)
        return {
            'status':'available', 'value':12.4, 'label':'accumulated rainfall', 'unit':'mm',
            'window_start':'2026-09-20', 'window_end':'2026-09-20', 'calculation':'sum of daily precipitation',
            'observation':{'city':'Kerala','state':'Kerala'}, 'source':'Open-Meteo Historical Weather API',
        }
    monkeypatch.setattr(live_weather, 'layer_value_for_window', value)
    data = client.post('/api/weathergpt', json={**selection(), 'latitude':28.6, 'longitude':77.2,
                                                  'question':'What was rainfall in Kerala on 2026-09-20?'}).json()
    assert captured['layer'] == 'open_meteo_rainfall'
    assert captured['state'] == 'Kerala'
    assert captured['latitude'] != pytest.approx(28.6)
    assert 'Accumulated Rainfall from 2026-09-20 to 2026-09-20 at Kerala, Kerala is 12.40 mm.' in data['answer']


def test_imagery_query_does_not_infer_weather_from_pixels(monkeypatch):
    from backend import imagery, nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'data_domain':'imagery', 'data_layer':'radar', 'query_scope':'selected_location',
        'time_range':{'kind':'unspecified'},
    }})
    monkeypatch.setattr(imagery, 'radar', lambda: {'radar':[{'time':1}], 'generated_at':'2026-09-29T00:00:00Z'})
    data = client.post('/api/weathergpt', json={**selection(), 'question': 'show radar'}).json()
    assert data['engine'] == 'Live radar layer metadata'
    assert 'does not convert pixels' in data['answer']


def test_unlisted_city_resolves_before_live_layer_query(monkeypatch):
    from backend import live_weather, nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location':None, 'location_candidate':'Kottayam', 'coordinates':None,
        'weather_variable':'temperature', 'data_layer':'live_weather', 'data_domain':'live_layers',
        'time_range':{'kind':'lead_hours', 'lead_hours':0}, 'query_scope':'selected_location', 'use_case':'general',
    }})
    monkeypatch.setattr(live_weather, 'resolve_indian_location', lambda query: {'name':'Kottayam','state':'Kerala','latitude':9.5916,'longitude':76.5222})
    captured = {}
    def layer_value(layer, state, latitude, longitude, location_name):
        captured.update(layer=layer, state=state, latitude=latitude, longitude=longitude, location_name=location_name)
        return {'status':'available','label':'current temperature','unit':'°C','value':29.4,
                'observation':{'city':'Kottayam','state':'Kerala'},'source':'Open-Meteo weather model'}
    monkeypatch.setattr(live_weather, 'layer_value', layer_value)
    data = client.post('/api/weathergpt', json={**selection(), 'question': "today's temperature in kottayam"}).json()
    assert 'Kottayam, Kerala is 29.40 °C' in data['answer']
    assert captured['location_name'] == 'Kottayam'
    assert data['parsed_query']['resolved_location']['latitude'] == 9.5916


def test_generic_current_weather_uses_live_summary_instead_of_hydra_archive(monkeypatch):
    from backend import live_weather, nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'location':'Kerala', 'coordinates':None, 'weather_variable':None,
        'weather_summary_request':True, 'data_layer':'weather_summary', 'data_domain':'live_layers',
        'time_range':{'kind':'lead_hours', 'lead_hours':0, 'text':'today'}, 'query_scope':'selected_location',
    }})
    monkeypatch.setattr(live_weather, 'current_report', lambda state, latitude, longitude: {
        'status':'available', 'source':'Open-Meteo weather model', 'updated_at':'2026-09-30T00:00:00Z',
        'observation':{'city':'Thiruvananthapuram', 'state':'Kerala', 'temperature_max_C':29.4,
                       'rainfall_mm':1.2, 'humidity_percent':81, 'wind_speed_kmh':12.0,
                       'cloud_cover_percent':70, 'pressure_hpa':1008, 'description':'Partly cloudy'},
    })
    data = client.post('/api/weathergpt', json={**selection(), 'question':'what is the weather today in kerala'}).json()
    assert data['engine'] == 'Live IndianAPI / Open-Meteo weather summary'
    assert 'Current weather for Thiruvananthapuram, Kerala' in data['answer']
    assert 'temperature 29.4 °C' in data['answer']
    assert 'not a state-wide average or a HYDRA forecast' in data['answer']


def test_agriculture_suitability_uses_weather_screening_without_claiming_best_farm(monkeypatch):
    from backend import live_weather, nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'data_domain':'hydra_model', 'query_scope':'spatial_ranking', 'use_case':'agriculture',
        'weather_variable':'rainfall', 'time_range':{'kind':'unspecified'},
    }})
    def rank_layer(layer, state):
        if layer == 'open_meteo_rainfall_24h':
            return {'status':'available','items':[{'city':'Kochi','state':'Kerala','value':12.5}], 'sample_count':1, 'source':'Open-Meteo'}
        return {'status':'available','items':[{'city':'Kottayam','state':'Kerala','value':0.31}], 'sample_count':1, 'source':'Open-Meteo'}
    monkeypatch.setattr(live_weather, 'rank_layer', rank_layer)
    data = client.post('/api/weathergpt', json={**selection(), 'question': 'best farming location with perfect precipitation'}).json()
    assert data['engine'] == 'Agriculture weather screening'
    assert 'cannot name a best farm location' in data['answer']
    assert 'Kottayam, Kerala' in data['answer']


def test_historical_window_uses_published_repeated_model_summary(monkeypatch, tmp_path):
    from backend import data as data_module, nlp
    summary = tmp_path / 'historical-summary-test.csv'
    summary.write_text(
        'cell,latitude,longitude,rainfall_total_mm,temperature_mean_C,wind_mean_ms,heavy_rain_expected_days,heatwave_expected_days,high_wind_expected_days,heavy_rain_probability_mean,heatwave_probability_mean,high_wind_probability_mean\n'
        '1,10,76,1200,27,4,4.5,0.1,0.2,0.0123,0.0002,0.0005\n'
        '2,11,77,900,28,5,2.5,0.2,0.3,0.0068,0.0005,0.0008\n'
    )
    (tmp_path / 'long_term_latest.json').write_text('{"file":"historical-summary-test.csv","window_days":365,"window_start":"2025-01-01","window_end":"2025-12-31"}')
    monkeypatch.setattr(data_module, 'RUNTIME', tmp_path)
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'weather_variable':'rainfall', 'extreme_event_type':'heavy_rain', 'analysis_window_days':365,
        'query_scope':'spatial_ranking', 'use_case':'general', 'data_domain':'hydra_model', 'time_range':{'kind':'unspecified'},
    }})
    data = client.post('/api/weathergpt', json={**selection(), 'question':'where is prone to heavy rain based on 1 year of data'}).json()
    assert data['engine'] == 'HYDRA historical-model query'
    assert data['window_days'] == 365
    assert data['ranking'][0]['cell'] == '1'
    assert '4.50 expected heavy-rain days' in data['answer']


def test_live_daily_samples_are_retained_and_ranked(monkeypatch, tmp_path):
    from backend import live_weather
    from datetime import datetime, timezone
    monkeypatch.setattr(live_weather, 'DAILY_HISTORY_PATH', tmp_path / 'live_weather_daily.csv')
    sample = {'city':'Kottayam','state':'Kerala','latitude':9.59,'longitude':76.52,'temperature_max_C':29.0,
              'open_meteo_rainfall_mm':1.0,'open_meteo_rainfall_24h_mm':4.0,'open_meteo_humidity_percent':80.0,
              'open_meteo_cape_max_Jkg':100.0,'open_meteo_wind_gust_max_kmh':20.0,'open_meteo_wet_bulb_max_C':25.0,
              'open_meteo_soil_moisture_0_to_1cm':0.3}
    live_weather._record_daily_samples([sample], 'Open-Meteo', datetime(2026, 9, 28, tzinfo=timezone.utc))
    sample['temperature_max_C'] = 31.0
    live_weather._record_daily_samples([sample], 'Open-Meteo', datetime(2026, 9, 29, tzinfo=timezone.utc))
    ranking = live_weather.historical_rank_layer('live_weather', 2)
    assert ranking['status'] == 'available'
    assert ranking['items'][0]['value'] == pytest.approx(30.0)
    assert ranking['items'][0]['sample_days'] == 2


def test_open_meteo_lightning_watch_uses_provider_convection_inputs():
    from backend.live_weather import _forecast_summary
    summary = _forecast_summary({
        'cape': [300.0, 1100.0],
        'precipitation_probability': [20.0, 70.0],
        'weather_code': [3, 95],
    })
    assert summary['open_meteo_lightning_watch_percent'] == 100.0


def test_live_timeline_question_uses_retained_samples(monkeypatch):
    from backend import live_weather, nlp
    monkeypatch.setattr(nlp, 'query_parser_configured', lambda: True)
    monkeypatch.setattr(nlp, 'parse_query', lambda question: {'parsed': {
        'data_domain':'live_layers', 'data_layer':'open_meteo_humidity', 'analysis_window_days':30,
        'query_scope':'spatial_ranking', 'use_case':'general', 'time_range':{'kind':'unspecified'},
    }})
    monkeypatch.setattr(live_weather, 'historical_rank_layer', lambda layer, days, state: {
        'status':'available', 'items':[{'city':'Kottayam','state':'Kerala','value':82.0,'sample_days':30}],
        'sample_count':1, 'label':'relative humidity', 'unit':'%', 'window_days':30,
        'window_start':'2026-09-01', 'window_end':'2026-09-30', 'source':'Retained daily samples',
    })
    data = client.post('/api/weathergpt', json={**selection(), 'question':'where was humidity highest over 30 days'}).json()
    assert data['engine'] == 'Retained live-layer timeline query'
    assert 'Kottayam, Kerala' in data['answer']


def test_observations_real_netcdf():
    data=client.get('/api/observations',params={'latitude':22.75,'longitude':75.75,'day':'2014-07-01'}).json()
    assert data['status']=='available'
    assert math.isfinite(data['value'])
    assert data['unit']=='mm'
    outside=client.get('/api/observations',params={'latitude':0,'longitude':0}).json()
    assert outside['status']=='integration_pending'


def test_verification_scope():
    data=client.get('/api/verification').json()
    assert data['rows']
    assert 'aggregate benchmark' in data['scope']
    assert any(r['model']=='gating_adaptive' for r in data['rows'])

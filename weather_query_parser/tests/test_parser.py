from datetime import date

from weather_query_parser.nlp.parser import parse_query
from weather_query_parser.nlp.time_parser import parse_time_range


def test_plain_forecast_keeps_extreme_event_null():
    parsed = parse_query("What's the weather in Chennai tomorrow?")
    assert parsed["location"] == "Chennai"
    assert parsed["coordinates"] == {"lat": 13.0827, "lon": 80.2707}
    assert parsed["weather_variable"] is None
    assert parsed["time_range"] == {"kind": "lead_hours", "lead_hours": 24, "start_date": None, "end_date": None, "text": "tomorrow"}
    assert parsed["extreme_event_type"] is None


def test_generic_current_weather_routes_to_live_summary_not_hydra_archive():
    parsed = parse_query("What is the weather today in Kerala?")
    assert parsed["location"] == "Kerala"
    assert parsed["weather_variable"] is None
    assert parsed["weather_summary_request"] is True
    assert parsed["data_domain"] == "live_layers"
    assert parsed["data_layer"] == "weather_summary"


def test_misspelled_location_is_suggested_without_correction():
    parsed = parse_query("Will it rain in Mumbay tomorrow?")
    assert parsed["location"] is None
    assert parsed["location_suggestion"] == "Mumbai"
    assert parsed["weather_variable"] == "rainfall"


def test_coordinates_variable_and_each_supported_extreme_event():
    parsed = parse_query("chance of heavy rainfall at 19.076, 72.8777 in 2 days")
    assert parsed["coordinates"] == {"lat": 19.076, "lon": 72.8777}
    assert parsed["time_range"]["lead_hours"] == 48
    assert parsed["extreme_event_type"] == "heavy_rain"
    assert parse_query("Heatwave temperature in Rajasthan today")["extreme_event_type"] == "heatwave"
    assert parse_query("High winds in Kerala tomorrow")["extreme_event_type"] == "high_wind"


def test_where_question_is_a_spatial_ranking_request():
    parsed = parse_query("Name places with high prone to heavy rains")
    assert parsed["query_scope"] == "spatial_ranking"
    assert parsed["extreme_event_type"] == "heavy_rain"


def test_live_and_imagery_layers_route_to_the_right_domain():
    hottest = parse_query("Where is the most temperature today?")
    assert hottest["query_scope"] == "spatial_ranking"
    assert hottest["data_domain"] == "live_layers"
    assert hottest["data_layer"] == "live_weather"
    assert parse_query("Where has the highest humidity now?")["data_layer"] == "open_meteo_humidity"
    assert parse_query("Where has the strongest lightning watch today?")["data_layer"] == "open_meteo_lightning"
    assert parse_query("Show radar over India")["data_domain"] == "imagery"
    assert parse_query("Show radar over India")["data_layer"] == "radar"


def test_unlisted_place_is_retained_for_india_only_live_resolution():
    parsed = parse_query("today's temperature in Kottayam")
    assert parsed["location"] is None
    assert parsed["location_candidate"] == "Kottayam"
    assert parsed["data_domain"] == "live_layers"


def test_agriculture_use_case_is_explicit():
    parsed = parse_query("For farmers, where is the best location to farm with perfect precipitation?")
    assert parsed["use_case"] == "agriculture"


def test_requested_historical_window_is_extracted():
    parsed = parse_query("Which place is highly prone to heavy rainfall based on 1 year of data?")
    assert parsed["analysis_window_days"] == 365
    assert parsed["analysis_window_text"] == "1 year of data"
    assert parse_query("very heavy rainfall over 1 year")["extreme_event_type"] == "heavy_rain"


def test_supplied_2014_rainfall_is_routed_to_historical_observations():
    parsed = parse_query("What was the highest rainfall in India in 2014?")
    assert parsed["data_domain"] == "historical_observations"
    assert parsed["data_layer"] == "rainfall_observations_2014"
    assert parsed["time_range"]["start_date"] == "2014-01-01"
    assert parsed["time_range"]["end_date"] == "2014-12-31"


def test_question_buckets_detect_time_rankings_and_capability_requests():
    monthly = parse_query("Which month had the most rainfall in India in 2014?")
    assert monthly["question_operation"] == "time_ranking"
    assert monthly["query_bucket"] == "historical_rainfall_2014"
    assert parse_query("What questions can HYDRA answer?")["question_operation"] == "capability"


def test_model_methodology_questions_have_reusable_explanation_topics():
    architecture = parse_query("How does HYDRA make a forecast?")
    assert architecture["query_bucket"] == "hydra_model_methodology"
    assert architecture["question_operation"] == "explanation"
    assert architecture["explanation_topic"] == "architecture"
    assert parse_query("What do expert weights mean?")["explanation_topic"] == "weights"
    assert parse_query("What data is missing for fresh inference?")["explanation_topic"] == "data_coverage"


def test_forecast_scoped_weight_questions_route_to_saved_hydra_evidence():
    generic = parse_query("What do HYDRA expert weights mean?")
    assert generic["query_bucket"] == "hydra_model_methodology"
    scoped = parse_query("Show HYDRA weight distribution for grid cell 12000 tomorrow")
    assert scoped["query_bucket"] == "hydra_short_range_forecast"
    assert scoped["model_evidence_request"] is True
    assert scoped["grid_cell"] == "12000"
    assert scoped["query_scope"] == "selected_location"
    state = parse_query("Which expert has the highest weight in Maharashtra at 48 hours?")
    assert state["model_evidence_request"] is True
    assert state["location"] == "Maharashtra"
    assert state["time_range"]["lead_hours"] == 48


def test_interval_and_ambiguous_times_are_preserved():
    upcoming = parse_time_range("rainfall next 3 days", date(2026, 9, 29))
    assert upcoming.kind == "date_range"
    assert upcoming.start_date == "2026-09-30"
    assert upcoming.end_date == "2026-10-02"
    assert parse_time_range("rain in Delhi today or tomorrow").kind == "ambiguous"
    yesterday = parse_time_range("temperature yesterday", date(2026, 9, 30))
    assert yesterday.kind == "date_range"
    assert yesterday.start_date == yesterday.end_date == "2026-09-29"
    window = parse_time_range("rainfall from 2026-09-10 to 2026-09-15", date(2026, 9, 30))
    assert window.start_date == "2026-09-10"
    assert window.end_date == "2026-09-15"
    assert parse_time_range("temperature last 7 days", date(2026, 9, 30)).start_date == "2026-09-23"


def test_lat_lon_labels_and_state_match():
    parsed = parse_query("wind at latitude 28.6139 longitude 77.2090 in Uttar Pradesh")
    assert parsed["location"] == "Uttar Pradesh"
    assert parsed["coordinates"] == {"lat": 28.6139, "lon": 77.209}
    assert parsed["weather_variable"] == "wind"


def test_state_ranking_is_not_a_selected_state_lookup():
    parsed = parse_query("Which state had the highest temperature today?")
    assert parsed["data_domain"] == "live_layers"
    assert parsed["query_scope"] == "spatial_ranking"
    assert parsed["ranking_scope"] == "states"
    assert parsed["data_layer"] == "live_weather"
    yesterday = parse_query("Which state had the highest temperature yesterday?")
    assert yesterday["data_domain"] == "live_layers"
    assert yesterday["ranking_scope"] == "states"
    assert yesterday["time_range"]["start_date"] == yesterday["time_range"]["end_date"]


def test_current_shared_variables_use_the_matching_live_layer():
    assert parse_query("Which state had the highest rainfall today?")["data_layer"] == "open_meteo_rainfall"
    assert parse_query("Which state had the strongest wind today?")["data_layer"] == "open_meteo_wind_speed"
    dated = parse_query("Which state had the highest rainfall from 2026-09-10 to 2026-09-15?")
    assert dated["data_domain"] == "live_layers"
    assert dated["data_layer"] == "open_meteo_rainfall"
    assert dated["time_range"]["start_date"] == "2026-09-10"

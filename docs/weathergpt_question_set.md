# WeatherGPT question set and answer contract

This is the reusable question set for the HYDRA prototype. It is an intent
catalog, not a set of one-off chatbot replies. The parser extracts the same
fields from wording variants and dispatches a data-backed answer plan.

## Shared intent fields

| Field | Values used by HYDRA |
| --- | --- |
| `query_bucket` | `historical_rainfall_2014`, `live_weather_layers`, `hydra_short_range_forecast`, `radar_satellite_imagery`, `hydra_model_methodology` |
| `weather_variable` | rainfall, temperature, wind, humidity, thunderstorm potential, wind gust, heat stress, soil moisture, radar, satellite |
| `question_operation` | point lookup, location ranking, time ranking, aggregate, comparison, explanation, capability |
| `ranking_direction` | highest/descending or lowest/ascending |
| `time_range` | forecast lead, date, date range, or retained-live timeline window |
| `location` | recognised state/UT, mapped city, India-only geocoded city, or coordinates |
| `use_case` | general, agriculture, aviation, marine, energy, urban, emergency |

## 1. HYDRA model methodology and logic

Expected fields: methodology topic, trained experts, saved weights or interval,
coverage, provenance, and a limitation.

- How does HYDRA make a rainfall forecast?
- What trained experts are used by HYDRA?
- What does the largest expert weight mean?
- Why does persistence have the highest weight here?
- Are expert weights confidence scores?
- How is uncertainty shown in HYDRA?
- What does forecast spread mean?
- What is the saved 80% interval?
- Which extreme-event classifiers are trained?
- What is the heavy-rain definition?
- Can HYDRA predict very heavy rainfall separately?
- What data does fresh inference need?
- Why is fresh inference unavailable?
- What is the difference between archived output and a fresh forecast?
- Which file produced this answer?

## 2. HYDRA short-range forecast questions

Expected fields: valid date, selected cell or state, value and unit, lead,
interval, expert weights, regime, event probability where available, and source
provenance.

- What is the rainfall forecast here tomorrow?
- What will temperature be in Kerala at 48 hours?
- What is the wind forecast at these coordinates?
- Where is temperature highest at 24 hours?
- Which supplied cells have the most rainfall tomorrow?
- Where is heavy-rain probability highest?
- Where is high-wind probability highest?
- What is the heatwave probability in this state?
- Explain this rainfall forecast.
- Why do the experts disagree on this forecast?
- Compare rainfall and wind at this selected grid cell.

### Adaptive-weight question set

These questions read the saved HYDRA blend output for the selected supplied grid
or a state only when a published full-grid HYDRA cycle exists.

- Show the HYDRA weight distribution for grid cell 12000 tomorrow.
- Which expert has the highest rainfall weight at 11.50, 84.50 tomorrow?
- Show temperature, rainfall, and wind expert weights for this grid at 48 hours.
- What expert input value did each rainfall expert contribute?
- Does the allocation sum to 100% for this grid?
- Which expert dominates the wind forecast at this location?
- Compare the dominant expert at 24 and 48 hours for this supplied grid.
- Explain the expert spread and 80% interval beside this allocation.
- Show HYDRA weight distribution for Maharashtra tomorrow.

For a state, WeatherGPT averages weights only across published HYDRA grid cells
inside that state. If no such fresh output exists, it reports that no saved
HYDRA weights are available; it does not label Open-Meteo or ECMWF values as
HYDRA weights.

## 3. Live weather and retained timeline questions

Expected fields: provider location, state, coordinates, value, unit, source,
timestamp or retained window, and the provider-location limitation.

- What is the temperature in Kottayam now?
- What is the current rainfall in Mumbai?
- What is humidity in Chennai?
- What is the next 24-hour rainfall outlook in Kerala?
- Where are wind gusts strongest today?
- Where is thunderstorm potential highest?
- Which location has the highest heat stress?
- Where is surface soil moisture highest?
- Where was humidity highest over 30 days?
- Which city had the most rainfall over the retained timeline?

## 4. Historical 2014 rainfall observation questions

Expected fields: calendar window, aggregation method, grid resolution, state or
coordinates, rainfall value, source, and an observation-data limitation.

- Which place had the most rainfall in India in 2014?
- Which place had the least rainfall in 2014?
- Which month was wettest in India in 2014?
- Which month was driest in 2014?
- What was the highest annual rainfall in Kerala in 2014?
- What was rainfall at 22.75, 75.75 on 2014-07-01?
- What was the annual rainfall total at this 2014 grid cell?
- Is Meghalaya wetter than Rajasthan in the supplied 2014 data?
- Which location is best for farming in 2014 based on rainfall?

The last question receives a rainfall-screening answer only. HYDRA does not
recommend a farm location without crop, soil, irrigation, terrain, yield,
market, and crop-calendar data.

## 5. Radar and satellite questions

Expected fields: layer, source, update time or frame count, bounds, and an
imagery limitation.

- Is radar available right now?
- How many radar frames are available?
- What does the satellite IR layer represent?
- Can satellite imagery tell rainfall in my city?
- Can radar determine lightning probability at this coordinate?

## 6. Capability and limitation questions

- What questions can HYDRA answer?
- Which weather parameters are available?
- What live layers are available?
- Which questions need the 2014 rainfall dataset?
- Which questions need the trained HYDRA model?
- What can HYDRA not answer from the supplied data?

## Implementation rule

The deterministic parser and optional Gemini wording normalizer are evaluated
against this set. A new phrasing should be handled by the shared intent fields;
new code is required only when it needs a genuinely new data source, derived
metric, or model output that HYDRA does not already have.

## Operational question bank

The following matrix is the reusable WeatherGPT training contract for this
prototype. “Training” here means deterministic vocabulary, intent rules, fuzzy
location matching, and regression tests. It is intentionally not an invented
language-model training run: the original service is a deterministic parser and
the forecast values remain grounded in HYDRA outputs or named providers.

| Question family | Example wording variants | Parsed route | Required answer evidence |
| --- | --- | --- | --- |
| Current point weather | temperature in Kottayam now; humidity in Chennai today; rain in Mumbai currently | Live provider point lookup | location, coordinate, value, unit, provider timestamp |
| Current India location ranking | where is hottest today; which city has most humidity now | Live map-location ranking | ranked provider locations, coverage count, provider limitation |
| Current state/UT comparison | which state had highest temperature today; which state has most rain now | Live state/UT representative ranking | ranked state/UT coordinates, value/unit, explicit representative-point limitation |
| Live layer intelligence | where are gusts strongest; lightning watch in Kerala; highest CAPE today; where is soil moisture highest | Matching Open-Meteo layer | layer definition, value/unit, provider timestamp, method limitation |
| Exact date or duration | rainfall in Kerala on 2026-09-20; which state had the highest rainfall from 2026-09-20 to 2026-09-25; highest temperature last 7 days | Open-Meteo historical date-window calculation | exact start/end dates, aggregation, source, representative-coordinate limitation |
| Live timeline | where was humidity highest in the last 30 days; highest rainfall over retained daily samples | Retained daily-layer history | retained window, recorded sample count, aggregation, no-estimation statement |
| HYDRA point forecast | rainfall forecast in Kerala tomorrow; temperature at 19.07, 72.88 in 48 hours | HYDRA forecast output | valid date, forecast/units, model weights, spread, interval, provenance |
| HYDRA ranked forecast | where is temperature highest tomorrow; which supplied cell has the most rainfall at 48 hours | HYDRA output ranking | only supplied cells, ranked forecasts, model evidence, coverage limit |
| Extreme events | heavy-rain chance in Chennai tomorrow; highest high-wind probability; heatwave probability in Rajasthan | HYDRA classifier output | classifier probability, lead/date, model coverage and limitation |
| HYDRA model explanation | why did persistence dominate; explain uncertainty; what do expert weights mean | Methodology/evidence explanation | saved weights/inputs, spread/interval meaning, no causal or calibrated-confidence claim |
| HYDRA adaptive weights | show HYDRA weights for grid cell 12000 tomorrow; which expert dominates at 48 hours; weight distribution for Maharashtra | Saved HYDRA grid/state output | every expert allocation, expert input, total allocation, dominant expert, spread, interval, source |
| Historical rainfall | wettest month in India in 2014; highest annual rain in Kerala in 2014 | 2014 rainfall observations | date window, aggregation, grid resolution, observation source |
| Use-case screening | farming weather screen; aviation wind outlook; emergency heavy-rain screen | Available weather source plus use-case guardrail | weather evidence and required missing non-weather inputs |
| Imagery metadata | is radar available; what does satellite show | Radar/satellite metadata | update/frame metadata and pixel-inference limitation |
| Capability | what can HYDRA answer; which layers are live | Capability catalog | supported parameter, source, coverage, restriction |

### Routing rules

1. Explicit HYDRA/model/forecast wording selects saved HYDRA forecast evidence
   when that lead and location are available.
2. Today, now, current, or live selects the corresponding live provider layer
   unless the question explicitly names a HYDRA forecast.
3. Which state searches state/UT representative coordinates across India. It
   never inherits the state currently selected in the dashboard.
4. Where/which city/location searches only live map locations or supplied HYDRA
   cells. It does not make an all-India claim from a partial layer.
5. A year or timeline is answered only from the matching retained or supplied
   historical dataset. Missing days are reported rather than filled.
6. Heavy rain, heatwave, and high wind are the only classifier event names
   routed to HYDRA. Very heavy rain is described using the available heavy-rain
   classifier, not fabricated as another trained classifier.
7. A completed date or date range for temperature/rainfall uses the requested
   dates exactly. Temperature rankings use the maximum daily temperature in the
   window; rainfall rankings sum daily precipitation in the window.

Every route has parser regression cases and API-level answer tests. Add a new
question phrasing to the matching family rather than creating a one-off reply.

### Regression coverage

The parser question bank exercises representative wording for rain aliases,
temperature, humidity, wind and gusts, current and past dates, date ranges,
state/UT and location rankings, supported live layers, grid-cell references,
HYDRA adaptive-weight requests, and uncertainty explanations. This keeps new
phrasing inside shared intent buckets while the answer service selects only the
data source that can support the requested claim.

# WeatherGPT v2

WeatherGPT v2 answers questions about any Indian state, UT or city, for live conditions, forecasts and past dates,
and about the HYDRA model itself. Every answer gives the numbers, what they mean, the source, and the coverage limits.

## How a question is answered

1. **Understand.** A trained intent classifier (24 intents) plus rule-based overrides decides what is being asked.
   A slot extractor finds places (36 states/UTs, 200+ cities, abbreviations like MP/TN/J&K, regions, typos via
   fuzzy matching), variables, dates and periods, statistics, thresholds, top-k, sources and advice topics.
   Common Hinglish works ("kal Mumbai mein barish hogi kya", "pichle hafte").
2. **Remember.** Each chat tab has a session. Short follow-ups ("and Goa?", "which day was wettest there?",
   "did HYDRA predict it?") reuse the previous place, period, variable, question type or highlighted date.
3. **Resolve the source.** For each place, variable and period it picks the best source that actually covers it
   (table below), and says which one it used.
4. **Reason.** Numbers are interpreted: IMD rainfall classes and departure categories, heat and cold thresholds,
   humidity comfort, Beaufort wind, CAPE instability, trends, comparisons, model error vs interval, and practical
   advice (umbrella, travel, farming, outdoor events, flood screening, marine, health).
5. **Answer.** Lead sentence, ranked lines, labelled sections ("What this means", "Versus normal", "Source",
   "Coverage") and suggested follow-up questions.

## Data sources and priority

| Question | First choice | Then |
| --- | --- | --- |
| Past rainfall, state | IMD state-wise daily (with normals, departures, categories, weekly/monthly/season totals) | ERA5 state means from the HYDRA replay, then Open-Meteo archive |
| Past rainfall, city | the city's state from IMD/ERA5 (stated as such) | Open-Meteo at the city point |
| Past temperature, humidity, wind, etc. | ECMWF analysis on its date (state means over grid cells) | Open-Meteo archive at the place |
| Current conditions | IndianAPI (IMD) then Open-Meteo, via the prototype's live adapter | newest offline data, clearly labelled |
| Forecast | HYDRA published state cycle when the date matches | Open-Meteo forecast (16 days) |
| Rankings across states | IMD / ERA5 / ECMWF / HYDRA cycle when they cover the period | Open-Meteo, one central point per state |
| Departure from normal | IMD normals | Open-Meteo: same calendar window averaged over the previous 10 years |
| HYDRA accuracy, weights, misses, intervals | the HYDRA rolling replay (v3 metrics, experts, weights, local peaks) | validation summary and cycle backtests when present |

## Intents

greeting, thanks_bye, capabilities, data_coverage, current_weather, forecast, observed, rank_places, rank_days,
compare, threshold_days, trend, anomaly, model_how, model_weights, model_accuracy, model_vs_actual, uncertainty,
extremes, advice, why, imagery, provenance, out_of_scope.

## Example questions

- Live: "weather in Pune now", "is it raining in Kochi", "aaj Jaipur mein kitni garmi hai"
- Forecast: "will it rain in Hyderabad tomorrow", "temperature forecast for Lucknow next 3 days", "HYDRA forecast for Kerala"
- Past: "rainfall in Kerala on 20 September 2026", "total rain in Odisha in August 2025", "rainfall in Kerala in March 2019"
- Rankings: "top 5 wettest states last week", "which state had the highest CAPE on 24 September 2026", "states ranked by season deficit"
- Days: "wettest day in Assam in August 2025", "how many days above 20 mm in Kerala in August 2025", "list dry days in Rajasthan last month"
- Compare: "Kerala vs Karnataka rainfall in September 2026", "Gujarat July 2025 vs August 2025"
- Normals and trends: "is Bihar in deficit", "season rainfall for Punjab", "is rainfall increasing in Assam"
- Why: "why was Kerala so wet in August 2025", "why did HYDRA under-predict the peak in Assam"
- HYDRA: "how accurate is HYDRA in Maharashtra", "rank states by HYDRA error", "which expert dominates in Odisha",
  "did HYDRA predict 18 August 2025 in Maharashtra", "what are HYDRA's biggest misses in Gujarat", "what does the 80% interval mean"
- Advice: "should I carry an umbrella in Bengaluru tomorrow", "is it safe to drive to Manali this weekend", "can farmers in Punjab spray tomorrow"

## Training and evaluation

- Corpus: 124,000 generated sentences across 24 intents from hand-written templates and compositional frames,
  filled with every state, city, variable and date pattern, then perturbed with fillers, casing, punctuation loss,
  typos, dropped words and Hinglish.
- Model: delexicalised text (places, dates, numbers replaced by tokens) into word 1-3-gram and character 2-5-gram
  TF-IDF, multinomial logistic regression.
- Unseen-template test (whole templates held out, strict): 76.2% classifier-only.
- Acceptance set (140 hand-written questions): 92.9% classifier-only before it was used for development,
  98.6% for the full system after tuning against it.
- Blind set (86 questions written after tuning): 95.3% for the full system on first run; 98.8% after two general
  fixes (an off-topic relevance gate and non-Indian place rejection).
- Every handler was run on all questions offline and with mocked live data: zero failures.

## Limits

- State values from IMD, ERA5 and ECMWF are area averages; live values are at one point (city or state centre).
- Offline coverage is what the repo ships: IMD 2026-08-19 to 2026-09-24, ERA5 replay 2025-07-01 to 2025-12-31,
  HYDRA cycle 2026-01-01/02, ECMWF 2026-09-24. Everything else needs the live connection.
- "Why" answers combine the observed evidence with the usual seasonal drivers; they are explanations, not a
  diagnosis, since no synoptic charts are in the prototype.
- Advice is general weather guidance, not an official warning.
- Grid-cell archive questions, the 2014 rainfall grid and "one year of data" model rankings stay with the original
  WeatherGPT pipeline.

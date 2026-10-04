"""Synthetic-but-diverse training corpus for the WeatherGPT intent model.

Each intent has many hand-written templates. Placeholders are filled from the full state/city/variable
vocabulary, then perturbed (casing, punctuation, fillers, typos, Hinglish, word drops) so the model learns
the intent rather than the wording. Template ids are kept so evaluation can hold out unseen templates.
"""
from __future__ import annotations

import random
import re

from . import lexicon as L

INTENTS = ("greeting", "thanks_bye", "capabilities", "data_coverage", "current_weather", "forecast", "observed",
           "rank_places", "rank_days", "compare", "threshold_days", "trend", "anomaly", "model_how", "model_weights",
           "model_accuracy", "model_vs_actual", "uncertainty", "extremes", "advice", "why", "imagery", "provenance",
           "out_of_scope")

T = {}
T["greeting"] = ["hi", "hello", "hey", "hey there", "good morning", "good evening", "hello weathergpt", "hi there, who are you",
                 "namaste", "yo", "hey bot", "hii", "helo", "are you there", "who are you", "what are you", "what is weathergpt",
                 "introduce yourself", "hello, can you help me", "hey weather gpt", "sup", "good afternoon", "kaise ho"]
T["thanks_bye"] = ["thanks", "thank you", "thanks a lot", "great, thanks", "ok thanks", "bye", "goodbye", "see you", "that's all",
                   "cool thanks", "perfect", "awesome thank you", "shukriya", "dhanyavad", "nice", "ok got it", "cheers",
                   "thanks for the help", "that helps", "understood, thanks"]
T["capabilities"] = ["what can you do", "what questions can i ask", "help", "what can i ask you", "show me example questions",
                     "how do i use you", "what kind of questions do you answer", "list your features", "what are your capabilities",
                     "can you answer forecast questions", "what all can weathergpt answer", "give me some sample queries",
                     "what are you able to tell me about weather", "how can you help me", "what do you know", "kya kya pooch sakte hai",
                     "menu", "options", "what can hydra answer", "what topics do you cover"]
T["data_coverage"] = ["what data do you have", "which datasets are available", "what is the date range of your data",
                      "how far back does your {V} data go", "do you have data for {S}", "do you have {V} data for {Y}",
                      "which sources do you use", "what period does the imd data cover", "is there data for {R}",
                      "until which date do you have observations", "what is the latest date in your data",
                      "does the prototype have live data", "what years are covered", "do you have historical {V} for {S}",
                      "which variables are available", "is ecmwf data available", "what does the replay cover",
                      "coverage of the dataset", "what dates can i ask about", "data kab tak ka hai", "do you have data before {Y}",
                      "is temperature data available for {S}", "can you access live weather", "what is the source of your rainfall data"]
T["current_weather"] = ["what is the weather in {P} now", "weather in {P} today", "current weather {P}", "how is the weather in {P} right now",
                        "is it raining in {P} now", "what's the temperature in {P} right now", "{P} weather now", "live weather {P}",
                        "how hot is it in {P} at the moment", "current conditions in {P}", "is it humid in {P} today",
                        "{P} ka mausam kaisa hai", "aaj {P} mein mausam kaisa hai", "abhi {P} mein barish ho rahi hai kya",
                        "tell me today's weather for {P}", "what's it like outside in {P}", "is it windy in {P} now",
                        "current temperature and humidity in {P}", "{P} right now", "how's {P} weather today",
                        "is it sunny in {P} today", "weather update {P}", "give me the live report for {P}",
                        "what is the humidity in {P} currently", "is it cloudy in {P} now", "how cold is {P} today",
                        "what is the weather like here", "weather today", "current weather"]
T["forecast"] = ["will it rain in {P} {TF}", "{V} forecast for {P} {TF}", "what will the {V} be in {P} {TF}",
                 "rain forecast {P} {TF}", "will {P} get rain {TF}", "how much rain is expected in {P} {TF}",
                 "what is the temperature forecast for {P} {TF}", "forecast for {P} {TF}", "{P} weather {TF}",
                 "is rain expected in {P} {TF}", "will it be hot in {P} {TF}", "what is the hydra forecast for {S}",
                 "hydra +24h forecast for {S}", "hydra 48 hour {V} forecast {S}", "{TF} {P} mein barish hogi kya",
                 "kal {P} mein barish hogi", "{P} mein kal ka mausam", "will there be thunderstorms in {P} {TF}",
                 "predict the {V} in {P} {TF}", "what's the outlook for {P} {TF}", "chance of rain in {P} {TF}",
                 "rain probability {P} {TF}", "how windy will {P} be {TF}", "expected {V} in {P} over the {TF}",
                 "next 5 days forecast for {P}", "weekend weather {P}", "what does hydra predict for {S} tomorrow",
                 "will {P} be cooler {TF}", "is a heatwave expected in {P} {TF}", "parso {P} ka mausam"]
T["observed"] = ["how much rain did {P} get {T}", "what was the {V} in {P} {T}", "{V} in {P} {T}", "{P} {V} {T}",
                 "total rainfall in {P} {R}", "average {V} in {P} {R}", "what was the rainfall in {S} on {D}",
                 "how much did it rain in {P} {R}", "{V} recorded in {S} {T}", "show me {S} {V} for {R}",
                 "give me daily rainfall for {S} {R}", "what was the maximum temperature in {P} {T}",
                 "rainfall amount {S} {D}", "how wet was {S} {R}", "how hot was {P} {T}", "humidity in {P} on {D}",
                 "{S} mein {T} kitni barish hui", "{T} {P} mein kitni barish hui", "kitni barish hui {S} mein {R}",
                 "{S} rainfall {R} summary", "summarise {V} in {S} {R}", "what did {S} receive in rain {R}",
                 "era5 rainfall for {S} on {D}", "imd rainfall for {S} {T}", "observed rainfall {S} {D}",
                 "cumulative rainfall in {S} {R}", "how much rain fell across india {T}", "all india rainfall {T}",
                 "wind speed in {P} {T}", "what was the weather in {P} {T}", "weather in {P} on {D}", "{S} weekly rainfall",
                 "monthly rainfall for {S}", "season rainfall so far in {S}", "daily rainfall table for {S} {R}"]
T["rank_places"] = ["which state had the highest rainfall {T}", "which state got the most rain {R}", "wettest state {T}",
                    "driest states {R}", "top {K} wettest states {R}", "rank states by rainfall {R}", "which state was hottest {T}",
                    "which states received the least rain {R}", "list states with most rainfall {T}",
                    "where did it rain the most {T}", "which city was the hottest {T}", "top {K} states by {V} {R}",
                    "which state is the most humid today", "which state will get the most rain {TF}",
                    "which region of india was wettest {R}", "state wise rainfall ranking {R}", "sabse zyada barish kis state mein hui {T}",
                    "kis rajya mein sabse kam barish hui {R}", "which states had large excess rainfall {T}",
                    "which states are deficient {R}", "rank all states by departure from normal {R}",
                    "which state had the highest {V} {T}", "lowest {V} among states {T}", "which state was coldest {T}",
                    "which states are windiest right now", "where is it raining heavily today in india",
                    "which state had the biggest rainfall deficit {R}", "top {K} rainiest states in india {R}",
                    "bottom {K} states for rainfall {R}", "where in india was the most rain {T}"]
T["rank_days"] = ["which day had the most rain in {S} {R}", "wettest day in {S} {R}", "driest day in {P} {R}",
                  "hottest day in {P} {R}", "on which date did {S} get the heaviest rain", "peak rainfall day {S} {R}",
                  "when did {S} receive maximum rainfall {R}", "top {K} wettest days in {S} {R}",
                  "which date had the highest {V} in {S}", "what was the rainiest day for {S} in {Y}",
                  "{S} mein sabse zyada barish kab hui", "when was the peak of the monsoon in {S} {R}",
                  "list the {K} heaviest rain days in {S}", "biggest single-day rainfall in {S} {R}",
                  "which day was coolest in {P} {R}", "when was the maximum {V} in {S} {R}"]
T["compare"] = ["compare rainfall in {S} and {S2} {R}", "{S} vs {S2} rainfall {T}", "was {S} wetter than {S2} {R}",
                "difference in {V} between {P} and {P2} {T}", "compare {V} of {P} and {P2}", "{S} or {S2}, which got more rain {R}",
                "is {P} hotter than {P2} today", "compare {S} rainfall {R} with {R2}", "how does {S} compare to {S2} {R}",
                "{S} aur {S2} mein kahan zyada barish hui {T}", "compare {S} {V} {Y} vs {Y2}",
                "which was wetter {R} or {R2} in {S}", "{P} vs {P2} weather tomorrow", "compare departures of {S} and {S2}",
                "was this september wetter than last september in {S}", "compare hydra accuracy in {S} and {S2}",
                "contrast {S} and {S2} {V}", "{S} versus {S2} temperature {T}", "rain in {S}, {S2} and {S3} {R}",
                "difference between {S} and {S2} rainfall {T}"]
T["threshold_days"] = ["how many days had more than {N} mm rain in {S} {R}", "days with rainfall above {N} mm in {S} {R}",
                       "list days when {S} got over {N} mm", "count heavy rain days in {S} {R}", "how many dry days in {S} {R}",
                       "number of days above {N} degrees in {P} {R}", "on how many days was rainfall below normal in {S} {R}",
                       "how often did {S} exceed {N} mm {R}", "days with large excess rain in {S} {R}",
                       "how many days was {S} in deficit {R}", "{N} mm se zyada barish kitne din hui {S} mein",
                       "show the days {S} crossed {N} mm", "which days were wetter than {N} mm in {S}",
                       "how many rainy days in {S} {R}", "number of days with no rain in {S} {R}",
                       "days where hydra predicted more than {N} mm in {S}", "how many days did it rain in {P} {R}"]
T["trend"] = ["rainfall trend in {S} {R}", "is rainfall increasing in {S}", "how has rainfall changed in {S} {R}",
              "is {S} getting wetter or drier {R}", "trend of {V} in {P} {R}", "show the rainfall pattern in {S} over time",
              "has the monsoon weakened in {S} {R}", "how did rain evolve over {R} in {S}", "{S} rainfall over time",
              "is temperature rising in {P}", "week by week rainfall in {S} {R}", "monthly rainfall trend {S} {Y}",
              "{S} mein barish badh rahi hai ya kam ho rahi hai", "is the rain picking up in {S}",
              "how did the season progress in {S}", "trajectory of rainfall across {R} in {S}", "is it drying out in {S}",
              "daily rainfall series {S} {R} trend", "rainfall progression for india {R}"]
T["anomaly"] = ["was {S} rainfall above normal {T}", "rainfall departure for {S} {R}", "how much rain deficit does {S} have",
                "is {S} in excess or deficit this season", "how does {S} rainfall compare to normal {R}",
                "was it a normal monsoon in {S}", "rainfall anomaly in {S} {T}", "{S} rainfall percent of normal {R}",
                "is {S} getting less rain than usual", "how far below normal is {S}", "imd category for {S} {T}",
                "was {T} unusually wet in {S}", "{S} mein normal se kitni kam barish hui", "departure from normal all india {R}",
                "cumulative departure for {S}", "is the season deficient in {S}", "how abnormal was rainfall in {S} {R}",
                "did {S} get more rain than normal", "weekly departure {S}", "monthly departure for {S}"]
T["model_how"] = ["how does hydra work", "explain the hydra model", "what is hydra", "how are hydra forecasts made",
                  "what experts does hydra use", "what is the neural gate", "how does the gating network blend experts",
                  "what is the hurdle expert", "what does the spatial neighbourhood expert do", "what is the monsoon specialist",
                  "what is the upper quantile expert", "what inputs does hydra use", "which era5 variables feed the model",
                  "is hydra trained at grid level", "how is the state value computed from the grid", "what is persistence",
                  "what is climatology expert", "what is anomaly persistence", "what training data was used",
                  "when was the model trained", "what is the training cutoff", "explain the architecture",
                  "hydra kaise kaam karta hai", "what changed in hydra v3", "what is the difference between v2 and v3",
                  "how does hydra predict heavy rain", "what loss function is used", "how does the heavy rain head work",
                  "does hydra use cape and humidity", "how is the replay built", "what does lead time mean",
                  "what is the 35 day history", "how many experts are there"]
T["model_weights"] = ["show hydra weights for {S}", "which expert dominates in {S}", "expert weights for {S} {T}",
                      "what is the weight of persistence in {S}", "weight distribution for {S}", "which expert had the most weight {T} in {S}",
                      "how did the gate allocate weights in {S} {R}", "show expert allocation for {S}",
                      "does the gate change weights over time in {S}", "is the gate static", "how dynamic are the expert weights",
                      "show hydra weights for grid cell {CELL}", "which expert dominates the rainfall forecast in {S} tomorrow",
                      "expert contributions for {S} on {D}", "why does monsoon expert have high weight in {S}",
                      "average expert weights in {S} {R}", "gate weights {S}", "which expert is trusted most in {S}",
                      "show weights at 24 hours for {S}", "{S} ke liye expert weights dikhao"]
T["model_accuracy"] = ["how accurate is hydra in {S}", "hydra error in {S}", "what is the mae of hydra for {S}", "rmse for {S}",
                       "how good is the model", "how reliable is hydra", "hydra performance {R}", "is hydra better than persistence in {S}",
                       "does hydra beat climatology", "heavy rain recall for {S}", "how well does hydra catch heavy rain",
                       "what is the bias of hydra in {S}", "does hydra under predict peaks", "interval coverage for {S}",
                       "hydra accuracy in monsoon", "which state does hydra predict best", "where is hydra worst",
                       "rank states by hydra error", "validation results", "backtest metrics {S}", "skill score in {S}",
                       "how accurate was hydra {R}", "false alarm rate for heavy rain", "precision of heavy rain forecasts in {S}",
                       "hydra kitna sahi hai {S} mein", "can i trust hydra for heavy rainfall", "how accurate is the temperature forecast",
                       "error statistics by season", "compare hydra with persistence and climatology", "model accuracy by state"]
T["model_vs_actual"] = ["did hydra predict the rain in {S} on {D}", "what did hydra forecast vs actual in {S} on {D}",
                        "hydra prediction vs observed {S} {R}", "how close was hydra in {S} {T}",
                        "was hydra right about {S} on {D}", "predicted vs actual rainfall {S} {R}", "show hydra vs era5 for {S} {R}",
                        "did the model miss the heavy rain in {S} {R}", "how wrong was hydra on the wettest day in {S}",
                        "hydra forecast for {S} on {D} and what actually happened", "did hydra catch the peak in {S}",
                        "compare forecast and observation {S} {D}", "what was the forecast error on {D} in {S}",
                        "{S} mein {D} ko hydra ne kya predict kiya tha", "biggest misses of hydra in {S}",
                        "worst forecast day for {S}", "show the days hydra under predicted in {S}"]
T["uncertainty"] = ["what is the 80% interval", "explain forecast uncertainty", "how confident is the forecast for {S}",
                    "what does the interval mean", "what is expert spread", "why is the interval so wide in {S}",
                    "how is the interval calibrated", "what does conformal mean", "is the 80% interval really 80%",
                    "range of possible rainfall in {S} {TF}", "uncertainty band for {S}", "how sure is hydra",
                    "what is the confidence level", "why do experts disagree in {S}", "what is the spread on {D} in {S}",
                    "interval for {S} forecast", "what does disagreement mean", "kitna confident hai model"]
T["extremes"] = ["was there heavy rain in {S} {R}", "any extreme rainfall events in {S} {R}", "heavy rain days in {S}",
                 "probability of heavy rain in {S} {TF}", "chance of very heavy rainfall in {P} {TF}",
                 "where did the heaviest rain fall in {S}", "highest single-cell rainfall in {S} {R}",
                 "local extremes in {S} {T}", "did any district in {S} cross 64.5 mm", "heatwave in {P} {TF}",
                 "was there a heatwave in {P} {T}", "cyclone risk {S}", "extreme weather in {S} {R}",
                 "biggest downpour in {S} {R}", "is {P} likely to see a cloudburst", "heavy rain alert {S}",
                 "wettest grid cell in {S} {T}", "{S} mein bhari barish ki sambhavna", "high wind risk in {S} {TF}",
                 "extreme heat {P} {T}", "very heavy rain events {R} in india"]
T["advice"] = ["should i carry an umbrella in {P} {TF}", "is it a good day for a picnic in {P} {TF}", "can i travel to {P} {TF}",
               "is it safe to drive to {P} {TF}", "good time to sow paddy in {S}", "should farmers in {S} irrigate {TF}",
               "can i spray pesticide in {S} {TF}", "is there flood risk in {P} {TF}", "will there be waterlogging in {P} {TF}",
               "is it safe to trek in {P} {TF}", "should i plan an outdoor wedding in {P} {TF}", "can we play cricket in {P} {TF}",
               "is it safe for fishermen in {S} {TF}", "should i go to {P} for vacation {R}", "best time to visit {P}",
               "{TF} {P} mein chhata le jaun kya", "kya {TF} {P} jaana theek rahega", "will my flight from {P} be affected {TF}",
               "is it good weather for construction work in {P} {TF}", "is it too hot to go out in {P} {TF}",
               "should i postpone harvest in {S}", "landslide risk in {S} {TF}", "will solar panels produce well in {P} {TF}",
               "is it safe for kids to play outside in {P} today", "do i need a raincoat in {P} {TF}",
               "can i hold an event outdoors in {P} {TF}", "what should farmers in {S} do this week"]
T["why"] = ["why was {S} so wet {T}", "why did it rain so much in {S} {R}", "why is {P} so hot {T}", "what caused the heavy rain in {S}",
            "why was rainfall deficient in {S} {R}", "why is it humid in {P}", "reason for heavy rain in {S} {T}",
            "why did hydra predict high rain in {S}", "why did the model under predict in {S}", "explain why {S} got excess rain",
            "what drives rainfall in {S}", "why does {S} get so much rain", "why was {S} drier than {S2}",
            "{S} mein itni barish kyun hui", "why are thunderstorms likely in {P}", "what made {T} so stormy in {S}",
            "why is the cape high in {S}", "why was the interval wide on {D}"]
T["imagery"] = ["show radar", "is radar available", "show satellite image", "latest satellite picture over india",
                "what does the radar show", "radar over {S}", "satellite view of the clouds", "imd satellite ir image",
                "cloud imagery now", "doppler radar {P}", "show me the radar loop", "how many radar frames are available",
                "satellite image of the cyclone", "can you read rainfall from radar"]
T["provenance"] = ["what is the source of this answer", "which file did this come from", "where does this data come from",
                   "cite your source", "how was this computed", "is this from imd or era5", "what dataset is used for {S} rainfall",
                   "show provenance", "which model produced this", "is this live or archived", "how do you calculate state averages",
                   "what is the sha of the forecast file", "is this observed or modelled"]
T["out_of_scope"] = ["what is the capital of france", "tell me a joke", "who won the cricket match", "write a poem",
                     "what is the stock price of tcs", "how do i cook biryani", "translate hello to hindi", "what is 2+2",
                     "book a cab", "who is the prime minister", "recommend a movie", "what is bitcoin", "how to lose weight",
                     "play music", "what time is it in london", "solve this equation", "write python code for sorting",
                     "who are you dating", "news headlines", "train timings to {C}", "hotel booking in {C}",
                     "population of {S}", "gdp of {S}", "best restaurants in {C}", "what language is spoken in {S}",
                     "election results {S}", "how far is {C} from {C2}"]

FILLERS_PRE = ["", "", "", "", "hey ", "please ", "can you tell me ", "i want to know ", "quick question: ", "pls ", "bro ",
               "could you check ", "tell me ", "i need ", "weathergpt, ", "ok ", "so ", "hmm ", "yaar ", "bhai ", "dear bot, "]
FILLERS_POST = ["", "", "", "", "?", "?", " please", " pls", " thanks", " asap", " in detail", " briefly", " with numbers",
                " and explain", "??", " bata do", " quickly", " for my report", " and why"]
PAST_T = ["yesterday", "last week", "last month", "on 20 september 2026", "on 15 sep 2026", "on 2026-09-10", "last 7 days",
          "in august 2025", "on 18 august 2025", "in september 2026", "during monsoon 2025", "in july 2025", "two days ago",
          "past fortnight", "in the last 10 days", "on 1st september", "day before yesterday", "this week", "in october 2025",
          "kal", "pichle hafte", "in 2025", "earlier this month"]
FUTURE_T = ["tomorrow", "day after tomorrow", "next 3 days", "this weekend", "at 24 hours", "at 48h", "next week",
            "on monday", "tonight", "kal", "parso", "next 5 days", "over the next two days", "in the coming days"]
RANGES = ["in september 2026", "last week", "last month", "from 1 to 15 september 2026", "between 20 aug and 10 sep 2026",
          "in august 2025", "during monsoon 2025", "from july to september 2025", "in the last 30 days", "post monsoon 2025",
          "in october 2025", "this season", "since 1 august 2026", "first week of september", "in 2025", "over the past 2 weeks",
          "mid august 2025", "november 2025", "december 2025", "this month"]
DATES = ["20 september 2026", "2026-09-15", "1 sep 2026", "18 august 2025", "15/09/2026", "sept 5", "24 sep 2026",
         "july 10 2025", "3rd october 2025", "28 august 2026", "12 nov 2025"]
YEARS = ["2025", "2026", "2024", "2023"]
VARS = ["rainfall", "rain", "temperature", "humidity", "wind speed", "wind", "precipitation", "max temperature",
        "gusts", "cloud cover", "pressure", "cape", "temp", "barish", "showers"]
TYPO_RATE = 0.08


def _place(rng, kind="any"):
    if kind == "state" or (kind == "any" and rng.random() < 0.6):
        if rng.random() < 0.15:
            return rng.choice([a for a in L.STATE_ALIASES if a not in L.AMBIGUOUS_ALIASES] + ["UP", "MP", "TN", "AP", "WB", "J&K"])
        return rng.choice(L.STATES)
    return rng.choice(list(L.CITIES))


def _typo(word, rng):
    if len(word) < 5 or rng.random() > TYPO_RATE:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.33:
        return word[:i] + word[i + 1:]
    if op < 0.66:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def fill(template: str, rng: random.Random) -> str:
    s1 = _place(rng, "state")
    s2 = rng.choice([x for x in L.STATES if x != s1])
    s3 = rng.choice(L.STATES)
    reps = {
        "{S}": s1, "{S2}": s2, "{S3}": s3, "{P}": _place(rng), "{P2}": _place(rng), "{C}": rng.choice(list(L.CITIES)),
        "{C2}": rng.choice(list(L.CITIES)), "{V}": rng.choice(VARS), "{T}": rng.choice(PAST_T), "{TF}": rng.choice(FUTURE_T),
        "{R}": rng.choice(RANGES), "{R2}": rng.choice(RANGES), "{D}": rng.choice(DATES), "{Y}": rng.choice(YEARS),
        "{Y2}": rng.choice(YEARS), "{K}": rng.choice(["3", "5", "10", "three", "five"]), "{N}": rng.choice(["10", "20", "50", "64.5", "100", "35", "40", "5"]),
        "{CELL}": str(rng.choice([7000, 7001, 12000, 9500])),
    }
    out = template
    for key, value in reps.items():
        out = out.replace(key, value)
    return out


def perturb(text: str, rng: random.Random) -> str:
    text = rng.choice(FILLERS_PRE) + text + rng.choice(FILLERS_POST)
    words = text.split()
    words = [_typo(w, rng) for w in words]
    if len(words) > 6 and rng.random() < 0.08:
        del words[rng.randrange(len(words))]
    text = " ".join(words)
    r = rng.random()
    if r < 0.25:
        text = text.lower()
    elif r < 0.35:
        text = text.capitalize()
    elif r < 0.40:
        text = text.upper()
    if rng.random() < 0.3:
        text = re.sub(r"[?.!,]", "", text)
    return text.strip()


def generate(per_intent: int = 3000, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for intent in INTENTS:
        templates = T[intent]
        for k in range(per_intent):
            tid = k % len(templates)
            text = perturb(fill(templates[tid], rng), rng)
            rows.append({"text": text, "intent": intent, "template": f"{intent}:{tid}"})
    rows += generate_compositional(per_intent // 2, seed + 1)
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------- compositional frames (wider phrasing coverage)
OPEN_Q = ["", "can you tell me", "i'd like to know", "do you know", "please find", "check", "look up", "find out",
          "could you say", "any idea", "let me know", "i wonder", "quickly tell", "mujhe batao", "zara batao", "show me"]
FRAMES = {
    "current_weather": (["how is the weather", "what are conditions like", "is it raining", "how hot is it", "what's the temperature",
                         "how humid is it", "is it windy", "what's happening weather-wise", "mausam kaisa hai", "is it sunny",
                         "what's the sky like", "is it pouring", "feel like outside"],
                        ["in {P}", "at {P}", "over {P}", "for {P}"], ["right now", "now", "currently", "at the moment", "today", "this moment", "abhi", "aaj", "as we speak"]),
    "forecast": (["will it rain", "what will the weather be", "how much rain will fall", "will it be hot", "what's expected",
                  "what's the outlook", "will there be storms", "what is predicted", "how warm will it get", "barish hogi kya",
                  "is rain likely", "what temperature is forecast", "will showers come", "what are the odds of rain"],
                 ["in {P}", "for {P}", "over {P}", "at {P}"], ["tomorrow", "day after tomorrow", "next 3 days", "this weekend", "next week",
                                                              "kal", "parso", "tonight", "on monday", "in the coming days", "+48h", "at 24 hours"]),
    "observed": (["how much rain fell", "what was the rainfall", "how much did it rain", "what was the temperature",
                  "what rainfall was recorded", "how wet was it", "how much precipitation was observed", "what were the conditions",
                  "kitni barish hui", "give the rainfall numbers", "what was the total rain", "what was the daily rain"],
                 ["in {S}", "in {P}", "for {S}", "over {S}", "across {S}"], ["yesterday", "last week", "on {D}", "{R}", "in {R}", "during {R}", "kal", "pichle hafte"]),
    "rank_places": (["which state had the most rain", "which states were wettest", "where did it rain the most",
                     "which state was the driest", "rank the states by rainfall", "top states for rain", "which region got the most rain",
                     "which state recorded the highest rainfall", "list the wettest states", "which state topped rainfall", "kis state mein sabse zyada barish",
                     "which state had the lowest rain", "which states saw the biggest deficit", "where was rain heaviest across india"],
                    [""], ["yesterday", "last week", "{R}", "on {D}", "this month", "this season", "today", "pichle hafte"]),
    "rank_days": (["which day had the most rain", "what was the wettest day", "on which date was rain heaviest", "when did it rain most",
                   "which date was the driest", "what was the peak rain day", "which day topped rainfall", "sabse zyada barish kis din"],
                  ["in {S}", "for {S}", "over {S}"], ["{R}", "during {R}", "in {Y}", "this season", ""]),
    "compare": (["compare rainfall", "which got more rain", "how do the rainfall totals differ", "was it wetter", "contrast the weather",
                 "rainfall comparison", "who received more rain", "difference in rain", "kahan zyada barish hui", "temperature comparison"],
                ["between {S} and {S2}", "in {S} vs {S2}", "for {S} and {S2}", "of {P} and {P2}", "{S} versus {S2}"], ["{R}", "yesterday", "last week", "", "on {D}", "today"]),
    "threshold_days": (["how many days had rain above {N} mm", "count days with more than {N} mm", "on how many days did rain exceed {N} mm",
                        "how many dry days were there", "number of days over {N} mm", "list days when rain crossed {N} mm",
                        "how many rainy days", "how many days were in deficit", "kitne din {N} mm se zyada barish hui"],
                       ["in {S}", "for {S}", "across {S}"], ["{R}", "during {R}", "this season", "last month"]),
    "trend": (["is rainfall increasing", "what is the trend in rainfall", "is it getting wetter", "is it drying out", "how has rain evolved",
               "is the rain picking up or slowing", "show the rainfall trajectory", "how did rainfall change over time", "barish badh rahi hai kya",
               "is the monsoon strengthening", "is temperature trending up"],
              ["in {S}", "for {S}", "over {S}"], ["{R}", "over {R}", "this season", "recently", "lately", ""]),
    "anomaly": (["was rainfall above normal", "what is the departure from normal", "is there a rain deficit", "was it wetter than usual",
                 "how does rain compare with normal", "is the season deficient", "what percent of normal did it get", "normal se kitni kam barish",
                 "is rainfall in excess", "how abnormal was rainfall", "what is the imd category"],
                ["in {S}", "for {S}", "over {S}"], ["{R}", "yesterday", "this season", "last week", "so far", ""]),
    "model_accuracy": (["how accurate is hydra", "what is the model error", "how reliable is the forecast model", "does hydra beat persistence",
                        "how good is hydra at heavy rain", "what's hydra's mae", "how well does the model perform", "what is the rmse",
                        "is hydra biased", "how often does hydra catch heavy rain", "hydra kitna sahi hai", "what is the skill score"],
                       ["in {S}", "for {S}", "across india", "overall", ""], ["", "{R}", "during monsoon", "in the replay"]),
    "model_vs_actual": (["did hydra get it right", "how close was the forecast", "what did hydra predict versus what fell",
                         "was the model forecast correct", "show forecast vs observed", "did the model catch the rain", "how far off was hydra",
                         "hydra ne kya predict kiya tha"],
                        ["in {S}", "for {S}"], ["on {D}", "{R}", "on the wettest day", "yesterday", ""]),
    "extremes": (["were there heavy rain events", "was there extreme rain", "what was the heaviest local rainfall", "any very heavy rain",
                  "chance of heavy rain", "did any area cross 64.5 mm", "was there a heatwave", "any cloudburst", "heavy rain risk",
                  "bhari barish ki sambhavna"], ["in {S}", "in {P}", "over {S}"], ["{R}", "tomorrow", "this week", "{T}", ""]),
    "advice": (["should i carry an umbrella", "is it safe to travel", "can i plan a picnic", "should farmers irrigate", "is it ok to spray crops",
                "is there a flood risk", "can we play outside", "should i postpone my trip", "is it safe for fishermen", "chhata le jaun kya",
                "can i do construction work", "is it good for sowing", "should i go trekking", "will my event be ruined by rain"],
               ["in {P}", "to {P}", "around {P}", "in {S}"], ["tomorrow", "today", "this weekend", "next week", "kal", ""]),
    "why": (["why was it so wet", "why did it rain so much", "what caused the heavy rain", "why is it so hot", "why the deficit",
             "what explains the rainfall", "why so humid", "itni barish kyun hui", "what is behind the dry spell", "why did hydra miss"],
            ["in {S}", "in {P}", "over {S}"], ["{T}", "{R}", "", "this season"]),
}


def generate_compositional(per_intent: int = 1500, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for intent, (cores, places, times) in FRAMES.items():
        for k in range(per_intent):
            core, place, when = rng.choice(cores), rng.choice(places), rng.choice(times)
            parts = [rng.choice(OPEN_Q), core, place, when]
            if rng.random() < 0.3:  # move time before place
                parts = [parts[0], parts[1], parts[3], parts[2]]
            if rng.random() < 0.15:  # time first
                parts = [parts[3], parts[0], parts[1], parts[2]]
            text = " ".join(p for p in parts if p)
            rows.append({"text": perturb(fill(text, rng), rng), "intent": intent, "template": f"{intent}:frame:{cores.index(core)}"})
    return rows

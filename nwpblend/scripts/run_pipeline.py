"""End-to-end CLI.

    python scripts/run_pipeline.py all            # everything below, in order
    python scripts/run_pipeline.py ingest         # raw ERA5 NetCDF -> daily truth parquet
    python scripts/run_pipeline.py features       # feature tables (blocked + time_forward splits)
    python scripts/run_pipeline.py train          # models + test predictions (blocked, full; time_forward, point models)
    python scripts/run_pipeline.py evaluate       # benchmark CSVs + external validation
    python scripts/run_pipeline.py report         # reports/BENCHMARK_REPORT.md + figures
    python scripts/run_pipeline.py demo           # inference demo from saved artifacts
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from nwpblend import config as C  # noqa: E402


def ingest():
    from nwpblend.ingest import build_era5_daily
    build_era5_daily()


def features():
    from nwpblend.features import build_all
    build_all("blocked")
    build_all("time_forward", cellsets=("subgrid",))


def train():
    from nwpblend import pipeline
    pipeline.run("blocked", full=True)
    pipeline.run("time_forward", full=False)


def inventory() -> pd.DataFrame:
    from nwpblend import ingest as I
    rows = []
    d = I.load_era5_daily(columns=["date", "latitude", "longitude"])
    rows.append(("data_stream-oper_stepType-{accum,instant}.nc", "ERA5 6-hourly 2025, 121x117 grid (7-37N, 68-97E)",
                 f"{d.date.min().date()}..{d.date.max().date()}", f"{len(d):,} cell-days", "USED: truth archive (rebuilt; the brief's processed csv.gz is absent)"))
    f = I.load_ecmwf_snapshot()
    rows.append(("forecast_india_2026-09-24_0h_clean.csv (+ .grib2)", "ECMWF oper analysis step 0 h", "2026-09-24", f"{len(f):,} points",
                 "NOT SCORABLE: tp=0 at step 0, no truth at valid time; optional inference input"))
    imd = I.load_imd_statewise()
    rows.append(("rainfall_statewise_daily_imd_clean.csv", "IMD statewise daily rain: actual/normal/departure",
                 f"{imd.date.min().date()}..{imd.date.max().date()}", f"{imd.state.nunique()} states/regions", "USED: external validation (unseen year)"))
    m = I.load_meteostat()
    rows.append(("export_meteostat_2025_clean.csv", "single station daily 2025 (wpgt column holds shifted pressure values - dropped)",
                 f"{m.date.min().date()}..{m.date.max().date()}", f"{len(m)} days", "USED: ERA5-vs-gauge check"))
    inv = I.grib_inventory(C.ERA5_FEB_GRIB)
    rows.append(("72d1.../data.grib (6.9 GB)", f"ERA5 global GRIB1, {inv['messages']} msgs: soil temp/moisture, runoff, skt, ssrd, tp",
                 f"{inv['first_time'][:3]}..{inv['last_time'][:3]}", "Feb 2025 only", "NOT USED: one month, no new target info"))
    rows.append(("model_skill_summary.csv / blended_predictions_2025_Q4_test.csv", "earlier prototype, Q4 time split", "2025-10..12",
                 "", "USED: comparison in s.7"))
    rows.append(("blend_framework.py, *_by_season.csv (named in brief)", "-", "-", "-", "ABSENT from workspace"))
    return pd.DataFrame(rows, columns=["file", "content", "dates", "size", "role"])


def evaluate():
    from nwpblend.evaluation import evaluate as E, external as X
    from nwpblend.splits import SPLITTERS, summarize
    out = C.RESULTS_DIR / "blocked"
    inventory().to_csv(out / "data_inventory.csv", index=False)
    s = summarize(SPLITTERS["blocked"](pd.date_range("2025-01-01", "2025-12-31"))).reset_index(names="season")
    s.to_csv(out / "split_summary.csv", index=False)
    E.run_all("blocked", full=True)
    E.run_all("time_forward", full=False)
    st = X.meteostat_check("blocked")
    st["agreement"].to_csv(out / "ext_station_agreement.csv", index=False)
    st["skill"].to_csv(out / "ext_station_skill.csv", index=False)
    imd = X.imd_transfer("blocked")
    imd["skill"].to_csv(out / "ext_imd_skill.csv", index=False)
    imd["weights"].to_csv(out / "ext_imd_weights.csv", index=False)
    X.ecmwf_note().to_csv(out / "ext_ecmwf.csv", index=False)


def report():
    from nwpblend.evaluation.report import write_report
    write_report("blocked")


def demo():
    from nwpblend.ingest import load_era5_daily
    from nwpblend.inference import BlendingForecaster
    fc = BlendingForecaster(C.ARTIFACTS_DIR / "blocked")
    hist = load_era5_daily()
    hist = hist[hist.date >= "2025-11-15"]
    out = fc.forecast(hist, "2025-12-28", leads=(1, 2, 3), cells=[7000, 7001, 12000])
    pd.set_option("display.width", 200)
    print(out[["cell", "latitude", "longitude", "target_date", "lead_days", "regime", "pred_t2m_C_mean", "lo80_t2m_C_mean",
               "hi80_t2m_C_mean", "w_persistence_t2m_C_mean", "pred_tp_mm", "prob_heavy_rain_day_flag", "spread_wind_speed_mean"]])
    out.to_csv(C.REPORTS_DIR / "inference_demo_2025-12-28.csv", index=False)


STEPS = {"ingest": ingest, "features": features, "train": train, "evaluate": evaluate, "report": report, "demo": demo}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STEPS if step == "all" else [step]):
        print(f"=== {name} ===", flush=True)
        STEPS[name]()

"""Load the HYDRA rolling replay into a tidy frame and add issue-time-only features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "runtime" / "hydra_rolling_rainfall_replay.json"
SEASON = {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 0}  # monsoon, post-monsoon, winter, pre-monsoon


IMD_CSV = ROOT / "runtime" / "imd_state_daily.csv"


def load(path: Path = REPLAY, lead: int = 1, truth: str = "era5", imd_csv: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """lead=2 needs a replay built with patches/replay_lead2_rows.patch (key 'states_lead2').

    truth="imd" replaces the ERA5 state-mean target (and local maximum) with IMD gridded rainfall from
    runtime/imd_state_daily.csv (python -m hydra_imd.build). ERA5 values are kept as y_era5 / y_local_max_era5.
    Rows without IMD data (island UTs, missing days) are dropped.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key = "states" if lead == 1 else f"states_lead{lead}"
    if key not in payload:
        raise KeyError(f"This replay has no '{key}' rows. Apply patches/replay_lead2_rows.patch and rebuild the replay.")
    rows = []
    for state, records in payload[key].items():
        for r in records:
            d = {"state": state, "date": pd.Timestamp(r["valid_date"]), "issue_date": pd.Timestamp(r["issue_date"]),
                 "y": r["actual_mm"], "hydra": r["hydra_mm"], "lo80": r["interval80"][0], "hi80": r["interval80"][1],
                 "spread": r.get("spread_mm"), "p_rain": r.get("p_rain"), "local_peak": r.get("local_peak_mm"),
                 "y_local_max": r.get("actual_local_max_mm")}
            for e in r["experts"]:
                d[f"e_{e['name']}"] = e["value"]
                d[f"w_{e['name']}"] = e["weight"]
            for k, v in (r.get("p_heavy_area") or {}).items():
                d[f"pa_{k}"] = v
            for k, v in (r.get("p_heavy_max_cell") or {}).items():
                d[f"pm_{k}"] = v
            rows.append(d)
    df = pd.DataFrame(rows).sort_values(["state", "date"]).reset_index(drop=True)
    if truth == "imd":
        df = attach_imd(df, imd_csv or IMD_CSV)
    elif truth != "era5":
        raise ValueError("truth must be 'era5' or 'imd'")
    return add_features(df), payload


def attach_imd(df: pd.DataFrame, csv: Path) -> pd.DataFrame:
    if not Path(csv).exists():
        raise FileNotFoundError(f"{csv} not found. Run: python -m hydra_imd.build --dir <imd folder>")
    imd = pd.read_csv(csv, parse_dates=["date"])
    merged = df.merge(imd[["state", "date", "imd_mm", "imd_local_max_mm"]], on=["state", "date"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between the replay dates/states and the IMD file")
    merged = merged.rename(columns={"y": "y_era5", "y_local_max": "y_local_max_era5"})
    merged["y"], merged["y_local_max"] = merged.pop("imd_mm"), merged.pop("imd_local_max_mm")
    return merged.sort_values(["state", "date"]).reset_index(drop=True)


def experts(df: pd.DataFrame) -> list[str]:
    return [c[2:] for c in df.columns if c.startswith("e_")]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged observations known on the issue day (valid day t-1 and earlier), never the target day."""
    df = df.copy()
    g = df.groupby("state", sort=False)
    df["obs_lag1"] = g["y"].shift(1)
    df["obs_mean3"] = g["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["obs_mean7"] = g["y"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["obs_max7"] = g["y"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
    df["obs_wetdays7"] = g["y"].transform(lambda s: (s.shift(1) >= 2.5).rolling(7, min_periods=1).sum())
    df["localmax_lag1"] = g["y_local_max"].shift(1)
    df["localmax_max7"] = g["y_local_max"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
    df["hydra_lag1_err"] = g["hydra"].shift(1) - df["obs_lag1"]
    ex = [c for c in df.columns if c.startswith("e_")]
    df["expert_max"] = df[ex].max(1)
    df["expert_min"] = df[ex].min(1)
    df["upper_minus_hydra"] = df["hi80"] - df["hydra"]
    df["season"] = df["date"].dt.month.map(SEASON).astype(int)
    df["state_id"] = df["state"].astype("category").cat.codes
    return df


FEATURES_BASE = ["hydra", "lo80", "hi80", "spread", "p_rain", "local_peak", "obs_lag1", "obs_mean3", "obs_mean7", "obs_max7",
                 "obs_wetdays7", "localmax_lag1", "localmax_max7", "hydra_lag1_err", "expert_max", "expert_min",
                 "upper_minus_hydra", "season", "state_id"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(FEATURES_BASE)
    cols += [c for c in df.columns if c.startswith(("e_", "w_", "pa_", "pm_"))]
    cols += ["state_mean_train", "state_p90_train", "state_wetfrac_train"]
    return cols


def add_state_climate(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """State climatology from the training window only (no peeking at the test month)."""
    stats = train.groupby("state")["y"].agg(state_mean_train="mean", state_p90_train=lambda s: s.quantile(0.9),
                                            state_wetfrac_train=lambda s: (s >= 1).mean())
    return frame.drop(columns=[c for c in stats.columns if c in frame], errors="ignore").merge(stats, left_on="state", right_index=True, how="left")

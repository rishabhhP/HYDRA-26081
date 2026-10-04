"""State- and cell-level evaluation tables shared by the replay builder and rolling validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .experts import EXPERTS

STATE_THRESHOLDS = (10.0, 20.0)   # on state-mean rainfall


def _refs(frame: pd.DataFrame) -> dict:
    return {name: frame[f"exp_{name}"].to_numpy() for name in ("persistence", "climatology")}


def state_metrics(frame: pd.DataFrame) -> dict:
    """Area-mean verification plus local-extreme detection inside the state."""
    if frame.empty:
        return {}
    out = M.full_report(frame["actual_mm"], frame["hydra_mm"], lower=frame["lower80"], upper=frame["upper80"],
                        nominal=1 - C.ALPHA, heavy_thresholds=STATE_THRESHOLDS, references=_refs(frame),
                        dates=frame["valid_date"].to_numpy())
    local = {}
    for t in C.HEAVY_THRESHOLDS_MM:
        key = f"{t:g}"
        obs = frame["actual_local_max_mm"].to_numpy() >= t
        prob = frame[f"max_p_heavy_{key}"].to_numpy()
        scores = M.contingency(obs, prob >= 0.5)
        scores["brier"] = M.brier(obs, prob)
        scores["definition"] = f"any cell in the state >= {key} mm/day; forecast = max cell P >= 0.5"
        local[key] = scores
    out["local_heavy"] = local
    out["local_peak"] = M.peaks(frame["actual_local_max_mm"], frame["local_peak_mm"], frame["valid_date"].to_numpy())
    return out


def cell_metrics(frame: pd.DataFrame) -> dict:
    """Grid-cell verification: the scale where heavy rain actually happens."""
    if frame.empty:
        return {}
    probs = {f"{t:g}": frame[f"p_heavy_{t:g}"].to_numpy() for t in C.HEAVY_THRESHOLDS_MM}
    out = M.full_report(frame["actual_mm"], frame["hydra_mm"], lower=frame["lower80"], upper=frame["upper80"],
                        nominal=1 - C.ALPHA, p_rain=frame["p_rain"].to_numpy(), heavy_probs=probs,
                        heavy_thresholds=C.HEAVY_THRESHOLDS_MM, references=_refs(frame))
    obs = frame["actual_mm"].to_numpy() >= frame["state_p95_mm"].to_numpy()
    prob = frame["p_heavy_state_p95"].to_numpy()
    rel = M.contingency(obs, prob >= 0.5)
    rel["brier"] = M.brier(obs, prob)
    rel["definition"] = "cell rain >= its state's training-period wet-day p95"
    out["heavy"]["state_p95"] = rel
    # point-forecast heavy detection for comparison with the probability heads
    out["heavy_from_point_forecast"] = M.heavy(frame["actual_mm"], frame["hydra_mm"], C.HEAVY_THRESHOLDS_MM)
    return out


def breakdown(frame: pd.DataFrame, fn, by: str, labels=None) -> dict:
    out = {}
    for key, group in frame.groupby(by):
        name = labels[int(key)] if labels is not None else str(key)
        out[name] = fn(group)
    return out


def gate_dynamics(states: pd.DataFrame) -> dict:
    ordered = states.sort_values(["state", "lead_days", "issue_idx"])
    pooled = M.weight_dynamics(ordered[[f"w_{e}" for e in EXPERTS]].to_numpy(), list(EXPERTS))
    per_state = {state: M.weight_dynamics(g.sort_values("issue_idx")[[f"w_{e}" for e in EXPERTS]].to_numpy(), list(EXPERTS))
                 for state, g in ordered[ordered["lead_days"] == C.LEADS[0]].groupby("state")}
    return {"pooled": pooled, "states": per_state}


def compact_cells(cells: pd.DataFrame) -> pd.DataFrame:
    keep = ["issue_idx", "lead_days", "state_id", "season_id", "regime_id", "cell", "actual_mm", "hydra_mm", "p_rain",
            "lower80", "upper80", "state_p95_mm", "exp_persistence", "exp_climatology"]
    keep += [c for c in cells.columns if c.startswith("p_heavy_")]
    out = cells[keep].copy()
    for col in out.columns:
        if out[col].dtype == np.float64:
            out[col] = out[col].astype(np.float32)
    return out

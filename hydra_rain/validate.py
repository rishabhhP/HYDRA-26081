"""Rolling-origin validation for HYDRA rainfall v3.

For each origin o (every ROLLING_STEP_DAYS once ROLLING_MIN_TRAIN_DAYS of history exist):
    fit the full model on truth dated <= o - 1 day, then forecast every issue day whose
    valid date falls in [o, o + horizon). Issue-day features may use the newest observed
    days (as an operational forecast would); model parameters stay frozen at the cutoff.

Results are pooled and broken down by fold, year, season, state, lead, regime and rainfall
intensity, at both state-mean and grid-cell scale, with skill against persistence and
climatology, heavy-rain precision/recall, peak error and interval coverage/width.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import report as R
from .grid import GridCube
from .pipeline import HydraRainModel, WARMUP_DAYS


def origins(cube: GridCube, step: int = C.ROLLING_STEP_DAYS, horizon: int = C.ROLLING_HORIZON_DAYS,
            min_train: int = C.ROLLING_MIN_TRAIN_DAYS, first=None, last=None) -> list[np.datetime64]:
    start = cube.dates[0] + WARMUP_DAYS + min_train
    if first is not None:
        start = max(start, np.datetime64(first, "D"))
    stop = cube.dates[-1] - max(C.LEADS) + 1
    if last is not None:
        stop = min(stop, np.datetime64(last, "D"))
    out, o = [], start
    while o <= stop:
        out.append(o)
        o = o + step
    return out


def run(cube: GridCube, step=C.ROLLING_STEP_DAYS, horizon=C.ROLLING_HORIZON_DAYS, min_train=C.ROLLING_MIN_TRAIN_DAYS,
        first=None, last=None, max_folds=None, train_gate=True) -> dict:
    folds = origins(cube, step, horizon, min_train, first, last)
    if max_folds:
        folds = folds[-max_folds:]
    if not folds:
        raise ValueError("No rolling origin has enough history; supply more years or lower min_train")
    state_frames, cell_frames, fold_info = [], [], []
    for k, origin in enumerate(folds):
        print(f"[fold {k + 1}/{len(folds)}] origin {origin}", flush=True)
        model = HydraRainModel().fit(cube, origin - 1, train_gate=train_gate)
        o = cube.day_index(origin)
        end = min(o + horizon, cube.n_days)
        issue = np.arange(max(o - max(C.LEADS), 0), end - 1)
        cells, states = model.predict(cube, issue)
        valid = (cells["issue_idx"] + cells["lead_days"]).between(o, end - 1)
        cells = cells[valid & cells["actual_mm"].notna()]
        sv = (states["issue_idx"] + states["lead_days"]).between(o, end - 1)
        states = states[sv & states["actual_mm"].notna()]
        states = states.assign(fold=k, origin=str(origin))
        cells = R.compact_cells(cells).assign(fold=np.int16(k))
        state_frames.append(states)
        cell_frames.append(cells)
        fold_info.append({"fold": k, "origin": str(origin), "cutoff": str(origin - 1),
                          "test_end": str(cube.dates[end - 1]), "state_rows": int(len(states)),
                          "cell_rows": int(len(cells)), "splits": model.meta["splits"],
                          "gate_dynamics": model.meta["gate_dynamics_calibration"]})
    states = pd.concat(state_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True)
    states["year"] = states["valid_date"].str[:4]
    cells["year"] = (cube.dates[0] + (cells["issue_idx"] + cells["lead_days"]).to_numpy()).astype(str).astype("U4")
    state_names = cube.state_names
    return {
        "model_version": C.MODEL_VERSION,
        "design": {"step_days": step, "horizon_days": horizon, "min_train_days": min_train,
                   "leads": list(C.LEADS), "folds": len(folds), "alpha": C.ALPHA,
                   "heavy_thresholds_mm": list(C.HEAVY_THRESHOLDS_MM)},
        "folds": fold_info,
        "state_scale": {
            "pooled": R.state_metrics(states),
            "by_fold": R.breakdown(states, R.state_metrics, "fold"),
            "by_year": R.breakdown(states, R.state_metrics, "year"),
            "by_season": R.breakdown(states, R.state_metrics, "season_id", C.SEASONS),
            "by_lead": R.breakdown(states, R.state_metrics, "lead_days"),
            "by_regime": R.breakdown(states, R.state_metrics, "regime_id", C.REGIMES),
            "by_state": R.breakdown(states, R.state_metrics, "state"),
        },
        "cell_scale": {
            "pooled": R.cell_metrics(cells),
            "by_year": R.breakdown(cells, R.cell_metrics, "year"),
            "by_season": R.breakdown(cells, R.cell_metrics, "season_id", C.SEASONS),
            "by_lead": R.breakdown(cells, R.cell_metrics, "lead_days"),
            "by_regime": R.breakdown(cells, R.cell_metrics, "regime_id", C.REGIMES),
            "by_state": {state_names[int(s)]: R.cell_metrics(g) for s, g in cells.groupby("state_id") if s >= 0},
        },
        "gate_dynamics": R.gate_dynamics(states),
    }


def markdown(result: dict) -> str:
    """Short human-readable summary of the validation JSON."""
    def row(name, rep):
        c, i = rep["continuous"], rep.get("interval", {})
        h = rep["heavy"].get("20") or next(iter(rep["heavy"].values()))
        sk = rep.get("skill", {})
        fmt = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        return (f"| {name} | {fmt(c['mae'])} | {fmt(c['rmse'])} | {fmt(c['bias'])} | {fmt(h['recall'])} | "
                f"{fmt(h['precision'])} | {fmt(rep['peaks'].get('ratio_at_max'))} | {fmt(i.get('coverage'), 3)} | "
                f"{fmt(i.get('mean_width'))} | {fmt(sk.get('persistence', {}).get('mae_skill'))} | "
                f"{fmt(sk.get('climatology', {}).get('mae_skill'))} |")

    head = ("| Slice | MAE | RMSE | Bias | Heavy recall | Heavy precision | Peak ratio | 80% coverage | Width | "
            "MAE skill vs persistence | MAE skill vs climatology |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [f"# HYDRA rainfall v3 rolling-origin validation", "",
             f"{result['design']['folds']} origins, step {result['design']['step_days']} days, horizon "
             f"{result['design']['horizon_days']} days, leads {result['design']['leads']}.", "",
             "## Grid-cell scale (heavy = 20 mm/day head)", "", head,
             row("Pooled", result["cell_scale"]["pooled"])]
    lines += [row(k, v) for k, v in result["cell_scale"]["by_season"].items()]
    lines += [row(f"Lead {k}", v) for k, v in result["cell_scale"]["by_lead"].items()]
    lines += ["", "## State-mean scale (heavy = state mean >= 20 mm/day)", "", head,
              row("Pooled", result["state_scale"]["pooled"])]
    lines += [row(k, v) for k, v in result["state_scale"]["by_season"].items()]
    lines += [row(k, v) for k, v in result["state_scale"]["by_year"].items()]
    dyn = result["gate_dynamics"]["pooled"]
    lines += ["", "## Gate dynamics", "",
              f"Mean per-expert weight std {dyn['mean_std']}, mean day-to-day total variation "
              f"{dyn['mean_total_variation_step']}, normalised entropy {dyn['mean_normalised_entropy']}"
              + (" (WARNING: weights are nearly static)" if dyn["static_warning"] else "") + "."]
    return "\n".join(lines) + "\n"

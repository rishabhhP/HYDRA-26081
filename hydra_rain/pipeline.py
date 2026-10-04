"""HYDRA rainfall v3: grid-cell training, multi-head neural gate, stratified conformal, state aggregation.

    fit(cube, cutoff)    uses only truth dated <= cutoff
        issue days split chronologically: gate-train | gate-val | calibration (with embargo gaps)
        experts: out-of-fold predictions for the gate, then refit on gate-train + gate-val
        gate:    trained on out-of-fold experts, early-stopped on gate-val
        conformal: fitted on the untouched calibration slice, at cell level and at state level
    predict(cube, issue indices) -> cell rows and state aggregates
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .calibration import StratifiedConformal
from .experts import AUX, EXPERTS, ExpertBank
from .features import FeatureBuilder, climatology, heavy_labels, regime_from, season_id, state_p95
from .gate import RainGate
from .grid import GridCube

WARMUP_DAYS = 30


def split_days(days: np.ndarray) -> dict[str, np.ndarray]:
    """Chronological gate-train / gate-val / calibration with embargo gaps between them."""
    days = np.sort(days)
    n = len(days)
    n_cal = max(1, int(round(n * C.CALIBRATION_FRAC)))
    n_val = max(1, int(round(n * C.GATE_VAL_FRAC)))
    gap = C.EMBARGO_DAYS + max(C.LEADS)
    cal = days[n - n_cal:]
    val = days[n - n_cal - n_val:n - n_cal]
    val = val[val <= cal.min() - gap]
    train = days[:n - n_cal - n_val]
    train = train[train <= (val.min() if len(val) else cal.min()) - gap]
    if len(train) < 20 or len(val) < 3:
        raise ValueError(f"Too little training history for HYDRA v3 ({n} issue days before cutoff)")
    return {"gate_train": train, "gate_val": val, "calibration": cal}


class HydraRainModel:
    def __init__(self):
        self.bank: ExpertBank | None = None
        self.gate: RainGate | None = None
        self.cell_conformal: StratifiedConformal | None = None
        self.state_conformal: StratifiedConformal | None = None
        self.meta: dict = {}
        self._builder: FeatureBuilder | None = None
        self._cube_id = None

    # -------------------------------------------------- shared setup
    def _prepare(self, cube: GridCube) -> FeatureBuilder:
        if self._builder is not None and self._cube_id == id(cube):
            return self._builder
        cutoff = np.datetime64(self.meta["cutoff"], "D")
        train_mask = cube.dates <= cutoff
        self._builder = FeatureBuilder(cube, climatology(cube, train_mask))
        self._cube_id = id(cube)
        return self._builder

    def _heavy(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        return heavy_labels(rows["target"].to_numpy(), rows["state_id"].to_numpy(), np.asarray(self.meta["state_p95"]))

    # -------------------------------------------------- fitting
    def fit(self, cube: GridCube, cutoff, train_gate=True) -> "HydraRainModel":
        cutoff = np.datetime64(cutoff, "D")
        self.cell_conformal = self.state_conformal = None
        self._builder = None
        cut_idx = int(np.searchsorted(cube.dates, cutoff, side="right")) - 1
        train_mask = cube.dates <= cutoff
        self.meta = {"model_version": C.MODEL_VERSION, "cutoff": str(cutoff), "first_date": str(cube.dates[0]),
                     "state_names": cube.state_names, "missing_optional_fields": cube.missing_optional,
                     "state_p95": state_p95(cube, train_mask).tolist(), "leads": list(C.LEADS)}
        builder = self._prepare(cube)
        issue = np.arange(WARMUP_DAYS, cut_idx - max(C.LEADS) + 1)
        splits = split_days(issue)
        self.meta["splits"] = {k: [str(cube.dates[v.min()]), str(cube.dates[v.max()]), int(len(v))] for k, v in splits.items()}
        fit_cells = cube.model_cells(C.TRAIN_CELL_STRIDE)
        print(f"  HYDRA v3 fit: cutoff {cutoff}, {len(issue)} issue days, {len(fit_cells)} training cells", flush=True)

        fit_days = np.concatenate([splits["gate_train"], splits["gate_val"]])
        rows = builder.rows(fit_days, fit_cells)
        print(f"  building out-of-fold expert predictions on {len(rows):,} rows", flush=True)
        self.bank = ExpertBank(builder.columns)
        oof = self.bank.oof_predict(rows)
        self.bank.fit(rows)

        is_train = rows["issue_idx"].isin(splits["gate_train"]).to_numpy()
        tr_rows, va_rows = rows[is_train], rows[~is_train]
        tr_ex, va_ex = oof[is_train], oof[~is_train]
        if len(tr_rows) > C.MAX_GATE_ROWS:
            keep = tr_rows.sample(n=C.MAX_GATE_ROWS, random_state=C.SEED).index
            tr_rows, tr_ex = tr_rows.loc[keep], tr_ex.loc[keep]
        self.gate = RainGate(builder.columns, EXPERTS, AUX, n_states=len(cube.state_names))
        if train_gate:
            print(f"  training neural gate on {len(tr_rows):,} rows", flush=True)
            self.gate.fit(tr_rows, tr_ex, va_rows, va_ex, self._heavy(tr_rows), self._heavy(va_rows))
        else:  # dry run without torch: shape-correct untrained gate
            self.gate.random_init()

        print("  calibrating conformal 80% intervals", flush=True)
        # Calibration needs the same representative grid cells used to fit the
        # experts.  Predicting every cell here only to discard most of them below
        # creates an unnecessarily large in-memory frame; full-grid state outputs
        # are still generated by the replay runner after this fit completes.
        cells, states = self.predict(cube, splits["calibration"], cells=fit_cells)
        sub = cells
        self.cell_conformal = StratifiedConformal(C.CELL_STRATA, C.MIN_STRATUM).fit(
            sub["raw_lower"], sub["raw_upper"], sub["actual_mm"], _strata(sub))
        self.state_conformal = StratifiedConformal(C.STATE_STRATA, C.MIN_STATE_STRATUM).fit(
            states["raw_lower"], states["raw_upper"], states["actual_mm"], _strata(states))
        cells, states = self._apply_conformal(cells, states)
        self.meta["calibration_check"] = {
            "cell": M.interval(sub["actual_mm"], *self._cell_interval(sub), 1 - C.ALPHA),
            "state": M.interval(states["actual_mm"], states["lower80"], states["upper80"], 1 - C.ALPHA),
            "note": "In-sample for the conformal step by construction; see rolling validation for out-of-sample coverage.",
        }
        ordered = states.sort_values(["state", "lead_days", "issue_idx"])
        self.meta["gate_dynamics_calibration"] = M.weight_dynamics(ordered[[f"w_{e}" for e in EXPERTS]].to_numpy(), list(EXPERTS))
        return self

    def _cell_interval(self, cells: pd.DataFrame):
        lo, hi, _ = self.cell_conformal.apply(cells["raw_lower"], cells["raw_upper"], _strata(cells))
        return lo, hi

    def _apply_conformal(self, cells: pd.DataFrame, states: pd.DataFrame):
        if self.cell_conformal is not None:
            lo, hi, used = self.cell_conformal.apply(cells["raw_lower"], cells["raw_upper"], _strata(cells))
            cells = cells.assign(lower80=lo, upper80=hi, interval_stratum=used)
        if self.state_conformal is not None:
            lo, hi, used = self.state_conformal.apply(states["raw_lower"], states["raw_upper"], _strata(states))
            states = states.assign(lower80=lo, upper80=hi, interval_stratum=used)
        return cells, states

    # -------------------------------------------------- prediction
    def predict(self, cube: GridCube, issue_indices, cells: np.ndarray | None = None):
        builder = self._prepare(cube)
        cells = cube.model_cells(1) if cells is None else cells
        frames = []
        for chunk in np.array_split(np.asarray(issue_indices), max(1, len(issue_indices) // 20)):
            if not len(chunk):
                continue
            rows = builder.rows(chunk, cells, with_target=False)
            ex = self.bank.predict(rows)
            out = self.gate.forward(rows, ex)
            frame = pd.DataFrame({
                "issue_idx": rows["issue_idx"].to_numpy(), "cell": rows["cell"].to_numpy(),
                "lead_days": rows["lead_days"].to_numpy(), "state_id": rows["state_id"].to_numpy(),
                "season_id": rows["season_id"].to_numpy(), "regime_id": rows["regime_id"].to_numpy(),
                "actual_mm": rows["target"].to_numpy(), "hydra_mm": out["blend"], "p_rain": out["p_rain"],
                "wet_amount_mm": out["wet_amount"], "raw_lower": out["raw_lower"], "raw_upper": out["raw_upper"],
            })
            for key, prob in out["heavy"].items():
                frame[f"p_heavy_{key}"] = prob
            for k, name in enumerate(EXPERTS):
                frame[f"exp_{name}"] = ex[name].to_numpy()
                frame[f"w_{name}"] = out["weights"][:, k]
            frames.append(frame)
        cells_df = pd.concat(frames, ignore_index=True)
        cells_df["valid_date"] = (cube.dates[0] + (cells_df["issue_idx"] + cells_df["lead_days"]).to_numpy()).astype(str)
        cells_df["issue_date"] = cube.dates[cells_df["issue_idx"]].astype(str)
        p95 = np.asarray(self.meta["state_p95"])
        cells_df["state_p95_mm"] = p95[np.clip(cells_df["state_id"], 0, len(p95) - 1)]
        cells_df["state"] = [cube.state_names[s] if s >= 0 else "" for s in cells_df["state_id"]]
        states_df = self.aggregate(cube, cells_df)
        return self._apply_conformal(cells_df, states_df)

    def aggregate(self, cube: GridCube, cells: pd.DataFrame) -> pd.DataFrame:
        """Area means per state plus the local extremes that the mean would hide."""
        tp = cube.flat("tp_mm")
        frames = []
        heavy_cols = [c for c in cells.columns if c.startswith("p_heavy_")]
        mean_cols = (["hydra_mm", "p_rain", "raw_lower", "raw_upper", "wet_amount_mm"] + heavy_cols
                     + [c for c in cells.columns if c.startswith(("exp_", "w_"))])
        for sid, name in enumerate(cube.state_names):
            members = cells[cells["cell"].isin(cube.state_cells[name])]
            if members.empty:
                continue
            g = members.groupby(["issue_idx", "lead_days"], sort=True)
            agg = g[mean_cols].mean()
            agg["local_peak_mm"] = g["hydra_mm"].max()
            agg["local_peak_upper_mm"] = g["raw_upper"].max()
            for col in heavy_cols:
                agg[f"max_{col}"] = g[col].max()
            agg = agg.reset_index()
            agg["state"], agg["state_id"] = name, sid
            vi = (agg["issue_idx"] + agg["lead_days"]).to_numpy()
            ok = vi < cube.n_days
            state_tp = tp[:, cube.state_cells[name]]
            agg["actual_mm"] = np.where(ok, state_tp[np.minimum(vi, cube.n_days - 1)].mean(1), np.nan)
            agg["actual_local_max_mm"] = np.where(ok, state_tp[np.minimum(vi, cube.n_days - 1)].max(1), np.nan)
            for t in C.HEAVY_THRESHOLDS_MM:
                agg[f"actual_heavy_frac_{t:g}"] = np.where(ok, (state_tp[np.minimum(vi, cube.n_days - 1)] >= t).mean(1), np.nan)
            hist = state_tp.mean(1)
            agg["season_id"] = [season_id(cube.dates[i]) for i in agg["issue_idx"]]
            agg["regime_id"] = [int(regime_from(hist[max(0, i - 2):i + 1].mean(), hist[max(0, i - 6):i + 1].mean(),
                                                season_id(cube.dates[i]))) for i in agg["issue_idx"]]
            frames.append(agg)
        out = pd.concat(frames, ignore_index=True)
        out["issue_date"] = cube.dates[out["issue_idx"]].astype(str)
        vi = out["issue_idx"] + out["lead_days"]
        out["valid_date"] = (cube.dates[0] + vi.to_numpy()).astype(str)
        out["spread_mm"] = out[[f"exp_{e}" for e in EXPERTS]].std(axis=1, ddof=0)
        return out

    # -------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.bank.save(directory)
        self.gate.save(directory)
        self.cell_conformal.save(directory / "conformal_cell.json")
        self.state_conformal.save(directory / "conformal_state.json")
        (directory / "model_meta.json").write_text(json.dumps(self.meta, indent=1, default=str))

    @classmethod
    def load(cls, directory: Path) -> "HydraRainModel":
        model = cls()
        model.meta = json.loads((directory / "model_meta.json").read_text())
        model.bank = ExpertBank.load(directory)
        model.gate = RainGate.load(directory)
        model.cell_conformal = StratifiedConformal.load(directory / "conformal_cell.json")
        model.state_conformal = StratifiedConformal.load(directory / "conformal_state.json")
        return model


def _strata(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"state": frame["state_id"].to_numpy(), "season": frame["season_id"].to_numpy(),
                         "lead": frame["lead_days"].to_numpy(), "regime": frame["regime_id"].to_numpy()})

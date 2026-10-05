"""Serve the HYDRA v3.1 post-processed rainfall (calibrated heavy-rain probabilities, alert tiers, corrected interval,
reliability cards) together with the release-gate verdict.

Replay rows come from the out-of-sample predictions file when it exists (each month predicted only from earlier
months), so the prototype never shows in-sample fits as if they were forecasts.

    GET /api/hydra-postprocessed?state=Kerala
    GET /api/hydra-postprocessed/alerts?date=2025-09-28&event=state_20
    GET /api/hydra-release-gate
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
POST = RUNTIME / "hydra_v3_1_postprocessed.json"
TIERS = (("warning", 0.6), ("alert", 0.4), ("watch", 0.2))
EVENTS = {"state_10": "state-average rain ≥ 10 mm", "state_20": "state-average rain ≥ 20 mm",
          "state_64.5": "state-average rain ≥ 64.5 mm", "local_64.5": "≥ 64.5 mm somewhere in the state"}
OBS_COL = {"state_10": ("y", 10.0), "state_20": ("y", 20.0), "state_64.5": ("y", 64.5), "local_64.5": ("y_local_max", 64.5)}


def tier(p) -> str | None:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    return next((name for name, cut in TIERS if p >= cut), None)


def _num(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, d)


def _oos_file() -> Path | None:
    for name in ("hydra_v3_1_oos_predictions_imd.csv", "hydra_v3_1_oos_predictions.csv"):
        if (RUNTIME / name).exists():
            return RUNTIME / name
    return None


def _mtime(p: Path | None) -> float:
    return p.stat().st_mtime if p and p.exists() else 0.0


@lru_cache(maxsize=4)
def _post(mtime: float) -> dict:
    return json.loads(POST.read_text(encoding="utf-8")) if POST.exists() else {}


@lru_cache(maxsize=4)
def _oos(path: str, mtime: float):
    import pandas as pd
    return pd.read_csv(path, parse_dates=["date"])


def post() -> dict:
    return _post(_mtime(POST))


def oos():
    p = _oos_file()
    return (_oos(str(p), _mtime(p)), p.name) if p else (None, None)


def release_gate() -> dict:
    try:
        from hydra_post.release import current
        return current()
    except Exception as exc:  # never break the page because of the gate
        return {"status": "not_evaluated", "failed": [f"gate unavailable ({type(exc).__name__})"], "checks": []}


def _rows_from_oos(df, state: str) -> list[dict]:
    g = df[df["state"] == state].sort_values("date")
    rows = []
    for r in g.to_dict("records"):
        probs = {k: _num(r.get(f"final_p_{k}"), 4) for k in EVENTS}
        rows.append({"valid_date": str(r["date"].date()), "observed_mm": _num(r["y"]), "observed_local_max_mm": _num(r.get("y_local_max")),
                     "hydra_v3_mm": _num(r["hydra"]), "amount_mm": _num(r["final_amount"]),
                     "interval80": [_num(r.get("lo_final", r["lo_asym"])), _num(r.get("hi_final", r["hi_asym"]))],
                     "v3_interval80": [_num(r["lo80"]), _num(r["hi80"])],
                     "probabilities": probs, "alert": {"state_20": tier(probs["state_20"]), "local_64.5": tier(probs["local_64.5"])},
                     "observed_events": {k: (None if r.get(c) is None else bool(r.get(c) >= t)) for k, (c, t) in OBS_COL.items()}})
    return rows


def _rows_from_post(p: dict, state: str) -> list[dict]:
    out = []
    for r in p.get("states", {}).get(state, []):
        r = dict(r)
        r["observed_events"] = {"state_10": None if r["observed_mm"] is None else r["observed_mm"] >= 10,
                                "state_20": None if r["observed_mm"] is None else r["observed_mm"] >= 20,
                                "state_64.5": None if r["observed_mm"] is None else r["observed_mm"] >= 64.5, "local_64.5": None}
        out.append(r)
    return out


def tier_summary(rows: list[dict]) -> dict:
    """Issued / hit-rate / events-captured for each tier, on the rows shown (only meaningful for out-of-sample rows)."""
    out = {}
    for event in ("state_20", "local_64.5"):
        obs = [(r["probabilities"].get(event), r["observed_events"].get(event)) for r in rows]
        obs = [(p, o) for p, o in obs if p is not None and o is not None]
        n_events = sum(o for _, o in obs)
        out[event] = []
        for name, cut in TIERS[::-1]:
            issued = [o for p, o in obs if p >= cut]
            out[event].append({"tier": name, "min_prob": cut, "issued": len(issued),
                               "hit_rate": _num(sum(issued) / len(issued)) if issued else None,
                               "events_captured": _num(sum(issued) / n_events) if n_events else None, "events": n_events})
    return out


def state_payload(state: str) -> dict:
    p = post()
    df, oos_name = oos()
    if not p and df is None:
        return {"status": "integration_pending",
                "message": "No post-processed HYDRA yet. Run: python -m hydra_post.evaluate, then python -m hydra_post.apply."}
    if df is not None and state in set(df["state"]):
        rows, provenance = _rows_from_oos(df, state), {"kind": "out_of_sample", "file": oos_name,
                                                       "note": "Each month was predicted by models trained only on earlier months."}
    else:
        rows, provenance = _rows_from_post(p, state), {"kind": "fitted", "file": POST.name,
                                                       "note": "In-sample fit on verified days; see the evaluation for honest scores."}
    if not rows:
        return {"status": "integration_pending", "state": state, "message": f"No post-processed rows for {state}."}
    alerts = [r for r in rows if r["alert"]["state_20"] in ("alert", "warning") or r["alert"]["local_64.5"] in ("alert", "warning")]
    return {"status": "available", "state": state, "model_version": p.get("model_version", "hydra-rain-v3.4-post"),
            "base_model": p.get("base_model"), "amount_method": p.get("amount_method"), "events": EVENTS,
            "truth": p.get("truth", "imd"),
            "tiers": dict(TIERS), "interval": p.get("interval"), "notes": p.get("notes", []), "provenance": provenance,
            "reliability": (p.get("reliability") or {}).get(state), "release_gate": release_gate(),
            "tier_summary": tier_summary(rows) if provenance["kind"] == "out_of_sample" else None,
            "recent_alerts": alerts[-12:], "rows": rows}


def alerts_on(day: str, event: str = "state_20") -> dict:
    if event not in EVENTS:
        return {"status": "error", "message": f"event must be one of {sorted(EVENTS)}"}
    df, oos_name = oos()
    items = []
    if df is not None:
        g = df[df["date"].dt.strftime("%Y-%m-%d") == day]
        for r in g.to_dict("records"):
            prob = _num(r.get(f"final_p_{event}"), 4)
            col, thr = OBS_COL[event]
            items.append({"state": r["state"], "probability": prob, "tier": tier(prob), "observed": bool(r[col] >= thr) if r.get(col) is not None else None,
                          "observed_mm": _num(r[col])})
        source = oos_name
    else:
        for state, rows in (post().get("states") or {}).items():
            r = next((x for x in rows if x["valid_date"] == day), None)
            if r:
                prob = r["probabilities"].get(event)
                items.append({"state": state, "probability": prob, "tier": tier(prob), "observed": None, "observed_mm": r.get("observed_mm")})
        source = POST.name
    items.sort(key=lambda x: -(x["probability"] or 0))
    return {"status": "available" if items else "no_data", "date": day, "event": event, "event_label": EVENTS[event],
            "source": source, "states": items, "issued": [x for x in items if x["tier"]], "release_gate": release_gate()}


def build_router():
    from fastapi import APIRouter, Query

    router = APIRouter()

    @router.get("/api/hydra-postprocessed")
    def hydra_postprocessed(state: str = Query(..., min_length=2, max_length=80)):
        """Calibrated heavy-rain risk, alert tiers, corrected interval and reliability card for one state."""
        return state_payload(state)

    @router.get("/api/hydra-postprocessed/alerts")
    def hydra_postprocessed_alerts(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"), event: str = "state_20"):
        """Every state's calibrated probability and tier for one date."""
        return alerts_on(date, event)

    @router.get("/api/hydra-release-gate")
    def hydra_release_gate():
        """Whether the published post-processed HYDRA passed its release checks."""
        return release_gate()

    return router

"""HYDRA v3.4 calibrated heavy-rain risk for WeatherGPT (same data and gate as /api/hydra-postprocessed)."""
from __future__ import annotations

from datetime import date


def _pp():
    from backend import postprocessed as P
    return P


def available() -> bool:
    try:
        P = _pp()
        return bool(P.post()) or P.oos()[0] is not None
    except Exception:
        return False


def coverage() -> tuple[date, date] | None:
    try:
        df, _ = _pp().oos()
        if df is not None and len(df):
            return df["date"].min().date(), df["date"].max().date()
        rows = next(iter((_pp().post().get("states") or {}).values()), [])
        return (date.fromisoformat(rows[0]["valid_date"]), date.fromisoformat(rows[-1]["valid_date"])) if rows else None
    except Exception:
        return None


def state_day(state: str, day: date) -> dict | None:
    payload = _pp().state_payload(state)
    if payload.get("status") != "available":
        return None
    row = next((r for r in payload["rows"] if r["valid_date"] == day.isoformat()), None)
    if row:
        row = {**row, "provenance": payload["provenance"]["kind"], "gate": payload["release_gate"]["status"],
               "reliability": payload.get("reliability")}
    return row


def state_summary(state: str) -> dict | None:
    payload = _pp().state_payload(state)
    return payload if payload.get("status") == "available" else None


def alerts(day: date, event: str = "state_20") -> dict:
    return _pp().alerts_on(day.isoformat(), event)


def gate() -> dict:
    return _pp().release_gate()


def pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def risk_sentence(row: dict) -> str:
    p = row["probabilities"]
    a20, a64 = row["alert"]["state_20"], row["alert"]["local_64.5"]
    tier = lambda t: t.upper() if t else "no tier"  # noqa: E731
    return (f"P(state average ≥ 20 mm) {pct(p['state_20'])} ({tier(a20)}); P(≥ 64.5 mm somewhere in the state) "
            f"{pct(p['local_64.5'])} ({tier(a64)}); v3.4 calibrated amount {row['amount_mm']:.1f} mm with an adaptive 80% range of "
            f"{row['interval80'][0]:.1f} to {row['interval80'][1]:.1f} mm.")


def outcome_sentence(row: dict) -> str:
    ev = row.get("observed_events") or {}
    bits = []
    if ev.get("state_20") is not None:
        bits.append("the state average did reach 20 mm" if ev["state_20"] else "the state average stayed below 20 mm")
    if ev.get("local_64.5") is not None:
        bits.append(f"a grid cell did reach 64.5 mm (wettest {row.get('observed_local_max_mm') or 0:.0f} mm)" if ev["local_64.5"]
                    else "no grid cell reached 64.5 mm")
    return ("What happened: " + "; ".join(bits) + ".") if bits else ""


def label(row_or_payload: dict) -> str:
    g = row_or_payload.get("gate") or row_or_payload.get("release_gate", {}).get("status")
    kind = row_or_payload.get("provenance")
    kind = kind if isinstance(kind, str) else (kind or {}).get("kind")
    return ("validated release" if g == "validated" else "provisional release (not all checks pass)") + \
           (", out-of-sample replay" if kind == "out_of_sample" else ", in-sample fit")

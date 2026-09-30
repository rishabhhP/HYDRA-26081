"""Run the forecast-app acceptance suite through the live WeatherGPT API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "training" / "acceptance" / "weathergpt_forecast_app_50.jsonl"
RESULTS = ROOT / "training" / "acceptance" / "weathergpt_forecast_app_50_results.json"
API_URL = "http://127.0.0.1:8000/api/weathergpt"
DEFAULT_CONTEXT = {"latitude": 11.5, "longitude": 84.5, "lead": 24, "source": "archive", "state": None}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def field(parsed: dict, name: str):
    return {
        "domain": parsed.get("data_domain"), "layer": parsed.get("data_layer"),
        "scope": parsed.get("ranking_scope"), "operation": parsed.get("question_operation"),
        "bucket": parsed.get("query_bucket"), "lead": (parsed.get("time_range") or {}).get("lead_hours"),
        "evidence": parsed.get("model_evidence_request"),
    }.get(name, parsed.get(name))


def run_case(row: dict) -> dict:
    payload = json.dumps({**DEFAULT_CONTEXT, "question": row["text"]}).encode("utf-8")
    request = Request(API_URL, data=payload, method="POST", headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        return {"text": row["text"], "passed": False, "reasons": [f"request failed: {exc}"], "response": None}
    parsed = body.get("parsed_query") or {}
    reasons = []
    if not isinstance(body.get("answer"), str) or not body["answer"].strip():
        reasons.append("empty WeatherGPT answer")
    for key, expected in row["expected"].items():
        actual = field(parsed, key)
        if actual != expected:
            reasons.append(f"{key}: expected {expected!r}, got {actual!r}")
    learned = parsed.get("trained_intent") or {}
    if learned.get("status") != "available":
        reasons.append("trained intent evidence unavailable")
    elif learned.get("intent") != row["intent"]:
        reasons.append(f"learned intent: expected {row['intent']!r}, got {learned.get('intent')!r}")
    return {"text": row["text"], "passed": not reasons, "reasons": reasons, "response": {"engine": body.get("engine"), "parsed": parsed}}


def main() -> None:
    rows = read_rows(SUITE)
    if len(rows) != 50:
        raise ValueError(f"Expected 50 suite rows, found {len(rows)}")
    results = [run_case(row) for row in rows]
    failures = [result for result in results if not result["passed"]]
    output = {"total": len(results), "passed": len(results) - len(failures), "failed": len(failures), "failures": failures, "results": results}
    RESULTS.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("total", "passed", "failed")}, indent=2))
    for failure in failures:
        print(f"FAIL: {failure['text']} :: {' | '.join(failure['reasons'])}")


if __name__ == "__main__":
    main()

"""Promote failed WeatherGPT acceptance cases into remediation training data."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "training" / "acceptance" / "weathergpt_forecast_app_50.jsonl"
RESULTS = ROOT / "training" / "acceptance" / "weathergpt_forecast_app_50_results.json"
OUTPUT = ROOT / "training" / "remediation" / "weathergpt_acceptance_failures.jsonl"


def main() -> None:
    suite = {row["text"]: row for row in (json.loads(line) for line in SUITE.read_text(encoding="utf-8").splitlines() if line.strip())}
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    failures = [suite[result["text"]] for result in results["failures"]]
    if len({row["text"] for row in failures}) != len(failures):
        raise ValueError("Remediation examples must be unique")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            record = {key: row[key] for key in ("text", "intent", "secondary_intent", "slots")}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"remediation_examples": len(failures), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()

"""Create ten distinct 200-query WeatherGPT training iterations."""

from __future__ import annotations

import json
from pathlib import Path

from generate_weathergpt_training_data import SEED, generate, scaled_counts


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training" / "weathergpt_intent_slots_1000.jsonl"
ITERATIONS = ROOT / "training" / "iterations"
MANIFEST = ITERATIONS / "manifest.json"


def read_texts(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["text"] for line in handle if line.strip()}


def main() -> None:
    ITERATIONS.mkdir(parents=True, exist_ok=True)
    for path in ITERATIONS.glob("*.jsonl"):
        path.unlink()
    seen = read_texts(BASE)
    batches = []
    counts = scaled_counts(200)
    for number in range(1, 11):
        seed = SEED + number
        records = generate(counts=counts, seed=seed, excluded_texts=seen)
        path = ITERATIONS / f"weathergpt_intent_slots_iteration_{number:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        seen.update(record["text"] for record in records)
        batches.append({"iteration": number, "seed": seed, "file": path.name, "records": len(records), "noisy_queries": 30, "intent_distribution": counts})
    MANIFEST.write_text(json.dumps({"base_corpus": BASE.name, "iterations": batches, "combined_records": len(seen)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"iterations": len(batches), "new_records": 2000, "combined_records": len(seen)}, indent=2))


if __name__ == "__main__":
    main()

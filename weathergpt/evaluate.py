"""System-level evaluation: trained classifier + routing rules, on the acceptance and blind sets.

    python -m weathergpt.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

from . import engine, intent
from .slots import extract

HERE = Path(__file__).resolve().parent


def load(name: str) -> list[tuple[str, str]]:
    rows = []
    for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            text, label = line.rsplit("\t", 1)
            rows.append((text, label))
    return rows


def run(name: str) -> dict:
    rows = load(name)
    model_hits = system_hits = 0
    misses = []
    for text, gold in rows:
        m = intent.predict(text)["intent"]
        sysint, _, _ = engine.resolve_intent(text, extract(text))
        model_hits += m == gold
        system_hits += sysint == gold
        if sysint != gold:
            misses.append({"text": text, "gold": gold, "system": sysint, "model": m})
    n = len(rows)
    return {"set": name, "size": n, "model_only_accuracy": round(model_hits / n, 4), "system_accuracy": round(system_hits / n, 4), "misses": misses}


def main() -> None:
    out = {name: run(name) for name in ("acceptance.tsv", "blind.tsv")}
    for name, r in out.items():
        print(f"{name}: model-only {r['model_only_accuracy']:.1%}, full system {r['system_accuracy']:.1%} (n={r['size']})")
        for m in r["misses"]:
            print("   MISS", m)
    (HERE / "model" / "system_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

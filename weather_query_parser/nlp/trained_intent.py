"""Optional trained intent and slot classifier for WeatherGPT.

The deterministic parser remains authoritative for routing, time parsing and
location matching.  This module provides trained intent/slot evidence from the
versioned JSONL corpus without silently changing any deterministic result.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


MODEL_PATH = Path(__file__).resolve().parents[1] / "training" / "models" / "weathergpt_intent_slot_model.joblib"
NULL_LABEL = "__null__"


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib

        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict) or "intent_model" not in loaded:
            return None
        return loaded
    except Exception:
        # A missing optional dependency or corrupt artifact must never stop the
        # deterministic parser from serving a query.
        return None


def status() -> dict[str, Any]:
    bundle = _bundle()
    if not bundle:
        return {"status": "unavailable", "model_path": str(MODEL_PATH)}
    return {
        "status": "available",
        "model_path": str(MODEL_PATH),
        "training": bundle.get("training", {}),
    }


def _prediction(model: Any, text: str) -> tuple[str, float]:
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])


def predict(text: str) -> dict[str, Any]:
    """Return learned labels as evidence only; never raise into parsing."""

    bundle = _bundle()
    if not bundle:
        return {"status": "unavailable"}
    try:
        intent, confidence = _prediction(bundle["intent_model"], text)
        slots: dict[str, dict[str, Any]] = {}
        for name, model in bundle.get("slot_models", {}).items():
            value, slot_confidence = _prediction(model, text)
            slots[name] = {
                "value": None if value == NULL_LABEL else value,
                "confidence": round(slot_confidence, 4),
            }
        return {
            "status": "available",
            "intent": intent,
            "confidence": round(confidence, 4),
            "slots": slots,
            "model_version": bundle.get("training", {}).get("model_version"),
        }
    except Exception:
        return {"status": "unavailable"}

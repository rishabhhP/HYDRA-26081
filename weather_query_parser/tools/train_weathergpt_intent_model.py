"""Train WeatherGPT's optional intent and categorical-slot classifier.

This trains from the generated JSONL corpus. The resulting artifact is an
assistive classifier: deterministic extraction remains authoritative in the
runtime parser.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "training" / "weathergpt_intent_slots_1000.jsonl"
ITERATIONS = ROOT / "training" / "iterations"
REMEDIATION = ROOT / "training" / "remediation"
MODEL_PATH = ROOT / "training" / "models" / "weathergpt_intent_slot_model.joblib"
METRICS_PATH = ROOT / "training" / "models" / "weathergpt_intent_slot_metrics.json"
NULL_LABEL = "__null__"
SLOT_FIELDS = ("variable", "aggregation", "time_type", "inference_type", "model_explain_type", "output_format")
REMEDIATION_WEIGHT = 20


def make_model() -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=12000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260930)),
    ])


def read_rows() -> list[dict]:
    paths = [DATASET, *sorted(ITERATIONS.glob("*.jsonl")), *sorted(REMEDIATION.glob("*.jsonl"))]
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_remediation"] = path.parent == REMEDIATION
                rows.append(row)
    if len(rows) < 100:
        raise ValueError("Training corpus must contain at least 100 JSONL records")
    texts = [row["text"] for row in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("Training corpora must not contain duplicate query text")
    return rows


def metric(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def fit(texts, labels, weights) -> Pipeline:
    return make_model().fit(texts, labels, classifier__sample_weight=weights)


def main() -> None:
    rows = read_rows()
    texts = [row["text"] for row in rows]
    intents = [row["intent"] for row in rows]
    train_indexes, test_indexes = train_test_split(
        range(len(rows)), test_size=0.2, random_state=20260930, stratify=intents,
    )
    train_indexes, test_indexes = list(train_indexes), list(test_indexes)
    train_texts = [texts[index] for index in train_indexes]
    test_texts = [texts[index] for index in test_indexes]
    weights = [REMEDIATION_WEIGHT if row["_remediation"] else 1 for row in rows]
    train_weights = [weights[index] for index in train_indexes]

    intent_evaluator = fit(train_texts, [intents[index] for index in train_indexes], train_weights)
    intent_metrics = metric([intents[index] for index in test_indexes], intent_evaluator.predict(test_texts))

    slot_metrics: dict[str, dict[str, float]] = {}
    for field in SLOT_FIELDS:
        labels = [row["slots"].get(field) or NULL_LABEL for row in rows]
        evaluator = fit(train_texts, [labels[index] for index in train_indexes], train_weights)
        slot_metrics[field] = metric([labels[index] for index in test_indexes], evaluator.predict(test_texts))

    # Fit deployable classifiers on the full, fixed corpus after measuring the
    # held-out split. This is a text classifier only; it cannot replace source
    # selection, geographic matching, date normalisation or HYDRA inference.
    intent_model = fit(texts, intents, weights)
    slot_models = {
        field: fit(texts, [row["slots"].get(field) or NULL_LABEL for row in rows], weights)
        for field in SLOT_FIELDS
    }
    training = {
        "model_version": "intent-slots-2026-09-30-iter10-acceptance50",
        "datasets": [str(path.relative_to(ROOT)) for path in [DATASET, *sorted(ITERATIONS.glob("*.jsonl")), *sorted(REMEDIATION.glob("*.jsonl"))]],
        "records": len(rows),
        "remediation_weight": REMEDIATION_WEIGHT,
        "intent_distribution": dict(Counter(intents)),
        "evaluation": {"test_fraction": 0.2, "intent": intent_metrics, "slots": slot_metrics},
        "trained_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_contract": "assistive evidence only; deterministic parser remains authoritative",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"intent_model": intent_model, "slot_models": slot_models, "training": training}, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(training, indent=2))


if __name__ == "__main__":
    main()

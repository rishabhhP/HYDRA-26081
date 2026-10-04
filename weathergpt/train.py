"""Train and evaluate the WeatherGPT intent model.

    python -m weathergpt.train                 # 3000 sentences per intent
    python -m weathergpt.train --per-intent 6000

Evaluation: (1) held-out templates (phrasings never seen in training) and (2) an independent hand-written
acceptance set (weathergpt/data/acceptance.tsv). Metrics are saved next to the model.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from . import corpus
from .intent import MODEL_PATH, delex

HERE = Path(__file__).resolve().parent


def build() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=120_000)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=8.0, max_iter=400))])


def acceptance() -> list[tuple[str, str]]:
    rows = []
    for line in (HERE / "data" / "acceptance.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            text, label = line.rsplit("\t", 1)
            rows.append((text, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-intent", type=int, default=3000)
    parser.add_argument("--holdout", type=float, default=0.2)
    args = parser.parse_args()
    t0 = time.time()
    rows = corpus.generate(args.per_intent)
    rng = random.Random(1)
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"], set()).add(r["template"])
    held = set()
    for intent, templates in by_intent.items():
        templates = sorted(templates)
        rng.shuffle(templates)
        held |= set(templates[:max(1, int(len(templates) * args.holdout))])
    train = [r for r in rows if r["template"] not in held]
    test = [r for r in rows if r["template"] in held]
    print(f"corpus {len(rows):,} sentences, {len(by_intent)} intents; training on {len(train):,}, unseen-template test {len(test):,}")

    model = build().fit([delex(r["text"]) for r in train], [r["intent"] for r in train])
    pred = model.predict([delex(r["text"]) for r in test])
    gold = [r["intent"] for r in test]
    unseen = {"accuracy": round(accuracy_score(gold, pred), 4), "macro_f1": round(f1_score(gold, pred, average="macro"), 4)}
    acc_rows = acceptance()
    acc_pred = model.predict([delex(t) for t, _ in acc_rows])
    acc_gold = [g for _, g in acc_rows]
    accept = {"accuracy": round(accuracy_score(acc_gold, acc_pred), 4), "macro_f1": round(f1_score(acc_gold, acc_pred, average="macro"), 4),
              "size": len(acc_rows), "errors": [(t, g, p) for (t, g), p in zip(acc_rows, acc_pred) if g != p]}
    print("unseen-template test:", unseen)
    print("hand-written acceptance:", {k: v for k, v in accept.items() if k != "errors"})
    for e in accept["errors"]:
        print("  MISS", e)

    # final model: every template plus the acceptance sentences' phrasing is NOT added (kept independent)
    final = build().fit([delex(r["text"]) for r in rows], [r["intent"] for r in rows])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    metrics = {"corpus_size": len(rows), "per_intent": args.per_intent, "intents": sorted(by_intent),
               "templates": sum(len(v) for v in by_intent.values()), "unseen_template_test": unseen,
               "acceptance": {k: v for k, v in accept.items()}, "per_intent_f1_unseen": {k: round(v["f1-score"], 4)
               for k, v in report.items() if k in by_intent}, "train_seconds": round(time.time() - t0, 1)}
    (MODEL_PATH.parent / "intent_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

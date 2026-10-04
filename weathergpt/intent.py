"""Delexicalised intent classifier (TF-IDF words + characters, multinomial logistic regression)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import lexicon as L
from .timeparse import MONTH_RE

MODEL_PATH = Path(__file__).resolve().parent / "model" / "intent_model.joblib"

_place_words = sorted({*(s.lower() for s in L.STATES), *(c.lower() for c in L.CITIES),
                       *(a for a in L.STATE_ALIASES if a not in L.AMBIGUOUS_ALIASES)}, key=len, reverse=True)
_PLACE_RE = re.compile(r"(?<![\w])(" + "|".join(re.escape(w) for w in _place_words) + r")(?![\w])")
_SHORT_RE = re.compile(r"(?<![\w])(UP|MP|TN|AP|WB|J&K|HP|MH|KA|JK)(?![\w])")


def delex(text: str) -> str:
    t = _SHORT_RE.sub(" placex ", text)
    t = t.lower().replace("&", " and ")
    t = _PLACE_RE.sub(" placex ", t)
    t = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b", " datex ", t)
    t = re.sub(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTH_RE})\b|\b(?:{MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", " datex ", t)
    t = re.sub(r"\b(?:19|20)\d\d\b", " yearx ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", " numx ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def model():
    """Load the trained model; if it is missing or was saved by an incompatible scikit-learn, return None
    (routing then relies on the rule layer) and log how to retrain."""
    import logging
    try:
        import joblib
        return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    except Exception as exc:
        logging.getLogger("weathergpt").warning("Intent model could not be loaded (%s); run: python -m weathergpt.train", exc)
        return None


def predict(text: str, top: int = 3) -> dict:
    m = model()
    if m is None:
        return {"intent": None, "confidence": 0.0, "ranked": []}
    probs = m.predict_proba([delex(text)])[0]
    order = probs.argsort()[::-1][:top]
    ranked = [(str(m.classes_[i]), float(probs[i])) for i in order]
    return {"intent": ranked[0][0], "confidence": ranked[0][1], "ranked": ranked}

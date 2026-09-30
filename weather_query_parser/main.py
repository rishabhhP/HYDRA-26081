"""FastAPI entry point for the HYDRA weather-query parser."""

from fastapi import FastAPI

from .ai.normalizer import normalize_query
from .nlp.models import ParseRequest
from .nlp.parser import parse_query
from .nlp.trained_intent import status as trained_intent_status


app = FastAPI(
    title="HYDRA Weather Query Parser",
    description="Converts weather questions into validated structured intent.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "weather query parser running", "trained_intent": trained_intent_status()}


@app.post("/parse")
def parse(request: ParseRequest) -> dict:
    """Normalize opportunistically, then always use the deterministic parser."""

    original_query = request.query
    try:
        normalized_query = normalize_query(original_query)
        ai_normalization_used = True
    except Exception:
        # Gemini is deliberately non-authoritative. A missing key, timeout, or
        # malformed response must never prevent deterministic parsing.
        normalized_query = original_query
        ai_normalization_used = False

    return {
        "original_query": original_query,
        "normalized_query": normalized_query,
        "ai_normalization_used": ai_normalization_used,
        "parsed": parse_query(normalized_query),
    }

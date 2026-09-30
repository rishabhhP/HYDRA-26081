"""Optional Gemini normalization before deterministic weather parsing."""

from __future__ import annotations

import os


SYSTEM_INSTRUCTION = """You normalize informal weather questions for HYDRA.

Return one plain English sentence only. Preserve every place name, coordinate,
date, number, weather variable, and extreme-event phrase supplied by the user.
Do not answer the question, infer missing facts, or add a location, date,
weather variable, or event. Use these terms when the user clearly means them:
rain/rainfall/precipitation -> rainfall; temperature/heat -> temperature;
wind/breeze/gust -> wind; heavy rain, heatwave, high wind. Preserve uncertain
spellings so the deterministic location matcher can offer a suggestion.
"""


class NormalizationError(RuntimeError):
    """Raised when Gemini normalization cannot produce usable text."""


def normalize_query(text: str) -> str:
    """Return an optional Gemini cleanup, never a model interpretation."""

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        raise NormalizationError("GEMINI_API_KEY and GEMINI_MODEL must be configured")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise NormalizationError("google-genai is not installed") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    normalized = " ".join((response.text or "").split())
    if not normalized:
        raise NormalizationError("Gemini returned empty normalization text")
    return normalized

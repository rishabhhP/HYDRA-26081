"""Optional answer and weather-query parser adapters for WeatherGPT."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError


def configured():
    return bool(os.getenv('HYDRA_NLP_API_URL'))


def query_parser_configured():
    """Use the co-located parser unless a deployment overrides its URL.

    The caller already handles a failed request as a safe fallback to the map
    selection, so a locally unavailable parser never makes WeatherGPT fail.
    """
    return True


def _query_parser_url() -> str:
    return os.getenv('HYDRA_QUERY_PARSER_URL') or 'http://127.0.0.1:8010/parse'


def _headers(parser: bool = False) -> dict[str, str]:
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    token = os.getenv('HYDRA_QUERY_PARSER_API_KEY' if parser else 'HYDRA_NLP_API_KEY')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def parse_query(question: str) -> dict:
    """Use the parser service, with an in-process deterministic fallback.

    The standalone service remains the deployment boundary. The fallback keeps
    a stopped local parser port from discarding a WeatherGPT query intent.
    """
    url = _query_parser_url()
    try:
        request = Request(url, data=json.dumps({'query': question}).encode('utf-8'), headers=_headers(parser=True), method='POST')
        timeout = float(os.getenv('HYDRA_QUERY_PARSER_TIMEOUT_SECONDS', '4'))
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except (OSError, URLError, TimeoutError):
        from weather_query_parser.nlp.parser import parse_query as deterministic_parse
        payload = {
            'original_query': question,
            'normalized_query': question,
            'ai_normalization_used': False,
            'parsed': deterministic_parse(question),
        }
    # The optional service may normalize informal wording, but routing must use
    # the parser revision shipped with this HYDRA backend. This also prevents a
    # stale long-running parser process from omitting newly added intent fields.
    from weather_query_parser.nlp.parser import parse_query as deterministic_parse
    normalized_query = payload.get('normalized_query')
    if not isinstance(normalized_query, str) or not normalized_query.strip():
        normalized_query = question
    payload['parsed'] = deterministic_parse(normalized_query)
    parsed = payload.get('parsed')
    if not isinstance(parsed, dict) or not isinstance(parsed.get('time_range'), dict):
        raise ValueError('Weather-query parser returned an invalid response')
    return payload


def answer(question: str, context: dict) -> dict:
    url = os.environ['HYDRA_NLP_API_URL']
    body = json.dumps({'question': question, 'context': context}).encode('utf-8')
    request = Request(url, data=body, headers=_headers(), method='POST')
    timeout = float(os.getenv('HYDRA_NLP_TIMEOUT_SECONDS', '20'))
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode('utf-8'))
    text = payload.get('answer') or payload.get('text') or payload.get('response')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('NLP service must return a non-empty answer, text, or response field')
    return {'answer': text.strip(), 'engine': payload.get('model', 'Configured NLP service')}

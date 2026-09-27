"""Ask a local Ollama model to read the same article as Naive Bayes.

The model is configurable with LLM_MODEL (default llama3.2). If Ollama is
not running, this returns a clear error and the rest of the app continues.
The model may only use the five Naive Bayes categories.
"""

from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv

from src.constants import CATEGORIES

SENTIMENTS = ("Positive", "Negative", "Neutral")

load_dotenv()

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _canon(value, allowed: tuple[str, ...] | list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    lookup = {item.lower(): item for item in allowed}
    return lookup.get(value.strip().lower())


def _parse_json(content: str) -> dict:
    content = (content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(content)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")
    return data


def _prompt(article: str) -> str:
    categories = ", ".join(CATEGORIES)
    return f"""You analyse one news article for a college NLP demo.
Return ONLY a JSON object with these keys:
- category: exactly one of {categories}
- sentiment: exactly one of Positive, Negative, Neutral
- summary: 2 or 3 sentences
- explanation: 2 to 4 sentences on why that category fits the article

Do not invent a new category. Do not add markdown.

Article:
\"\"\"
{article}
\"\"\""""


def analyze_with_llm(article: str) -> dict:
    model = os.getenv("LLM_MODEL", "llama3.2").strip() or "llama3.2"
    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    timeout = float(os.getenv("LLM_TIMEOUT", "90"))
    excerpt = (article or "").strip()
    if len(excerpt) > 6000:
        excerpt = excerpt[:6000]

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": _prompt(excerpt)}],
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(f"{base}/api/chat", json=payload, timeout=(5, timeout))
        response.raise_for_status()
        body = response.json()
        content = body.get("message", {}).get("content", "")
        data = _parse_json(content)
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "model": model,
            "error": (
                f"Ollama is not running at {base}. Classical NLP results are still shown. "
                f"Start Ollama, then run: ollama pull {model}"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "model": model,
            "error": f"The LLM ({model}) did not answer within {int(timeout)} seconds. Try again, or set a larger LLM_TIMEOUT.",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "ok": False,
            "model": model,
            "error": f"The LLM request failed ({exc}). Check that '{model}' is pulled in Ollama.",
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "ok": False,
            "model": model,
            "error": "The LLM replied, but the response was not the JSON object the app asked for.",
        }

    category = _canon(data.get("category"), CATEGORIES)
    sentiment = _canon(data.get("sentiment"), SENTIMENTS)
    summary = str(data.get("summary") or "").strip()
    explanation = str(data.get("explanation") or "").strip()

    if category is None or sentiment is None or not summary:
        returned = data.get("category")
        return {
            "ok": False,
            "model": model,
            "error": (
                "The LLM returned a category or sentiment outside the allowed set "
                f"({CATEGORIES}; {', '.join(SENTIMENTS)}). Raw category: {returned!r}."
            ),
        }

    return {
        "ok": True,
        "model": model,
        "category": category,
        "sentiment": sentiment,
        "summary": summary,
        "explanation": explanation or "The model did not add an explanation.",
    }

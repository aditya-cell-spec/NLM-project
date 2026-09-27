"""Classical sentiment with VADER (Valence Aware Dictionary and sEntiment Reasoner).

VADER is a lexicon and rule-based model, not the category Naive Bayes.
Each known word has a valence score. Negation and intensifiers adjust it.
The compound score is a normalized sum from -1 (negative) to +1 (positive).

Thresholds are the ones published with VADER:
compound >= 0.05 Positive, <= -0.05 Negative, otherwise Neutral.
"""

from __future__ import annotations

from functools import lru_cache

from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from src.bootstrap import ensure_nltk
from src.preprocess import clean_text

POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    ensure_nltk()
    return SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= POSITIVE_CUTOFF:
        return "Positive"
    if compound <= NEGATIVE_CUTOFF:
        return "Negative"
    return "Neutral"


def _contributors(text: str, limit: int = 4) -> tuple[list[dict], list[dict]]:
    """Show lexicon hits so the sentiment score can be explained in a viva."""
    analyzer = _analyzer()
    lexicon = analyzer.lexicon
    seen: set[str] = set()
    hits = []
    tokens = word_tokenize(clean_text(text))
    for token in tokens:
        if token in seen or token not in lexicon:
            continue
        seen.add(token)
        hits.append({"word": token, "valence": round(float(lexicon[token]), 3)})

    positive = sorted((h for h in hits if h["valence"] > 0), key=lambda h: -h["valence"])[:limit]
    negative = sorted((h for h in hits if h["valence"] < 0), key=lambda h: h["valence"])[:limit]
    return positive, negative


def analyze_sentiment(text: str) -> dict:
    scores = _analyzer().polarity_scores(text or "")
    compound = float(scores["compound"])
    positive_words, negative_words = _contributors(text)
    return {
        "label": _label(compound),
        "compound": round(compound, 4),
        "positive": round(float(scores["pos"]), 4),
        "neutral": round(float(scores["neu"]), 4),
        "negative": round(float(scores["neg"]), 4),
        "positive_words": positive_words,
        "negative_words": negative_words,
        "method": "VADER lexicon and rules",
    }

"""Run both pipelines on one article and package the result for the page."""

from __future__ import annotations

from src.compare import build_comparison
from src.llm_client import analyze_with_llm
from src.morphology import chunk_sentence, morphology_rows, parse_cfg
from src.preprocess import linguistic_analysis
from src.sentiment import analyze_sentiment
from src.wordnet_insights import wordnet_insights
from src import model as news_model

MIN_CHARS = 40
MAX_CHARS = 12000


def analyze_article(text: str, bundle: dict | None = None) -> dict:
    original = " ".join((text or "").split())
    if len(original) < MIN_CHARS:
        raise ValueError("Paste a longer news article — at least a few sentences.")

    truncated = False
    if len(original) > MAX_CHARS:
        original = original[:MAX_CHARS]
        truncated = True

    if bundle is None:
        bundle = news_model.get_bundle()

    linguistics = linguistic_analysis(original)
    prediction = news_model.predict_category(bundle, linguistics["processed"])
    sentiment = analyze_sentiment(original)
    similar = news_model.similar_articles(bundle, linguistics["processed"])
    morph_rows = morphology_rows(original)
    penn_by_lemma = {}
    for row in morph_rows:
        penn_by_lemma.setdefault(row["lemma"], row["penn"])
    # Prefer informative TF-IDF unigrams, then other lemmas, for WordNet.
    lookup_words = []
    for item in prediction["top_tokens"]:
        token = item["token"]
        if " " in token:
            continue
        lookup_words.append(token)
    lookup_words.extend(linguistics["content_lemmas"])
    insights = wordnet_insights(
        lookup_words,
        penn_by_lemma,
        category=prediction["category"],
        context=linguistics["content_lemmas"],
    )
    chunks = chunk_sentence(original)
    cfg = parse_cfg(original)
    llm = analyze_with_llm(original)
    comparison = build_comparison(prediction["category"], sentiment["label"], llm)

    return {
        "original": original,
        "truncated": truncated,
        "linguistics": linguistics,
        "prediction": prediction,
        "sentiment": sentiment,
        "similar": similar,
        "morphology": morph_rows,
        "wordnet": insights,
        "chunks": chunks,
        "cfg": cfg,
        "llm": llm,
        "comparison": comparison,
    }

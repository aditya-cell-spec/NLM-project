"""Check that metrics, classification, parsing, WordNet, and retrieval are real."""

from __future__ import annotations

import numpy as np

from src.compare import build_comparison
from src.model import get_bundle, predict_category, similar_articles
from src.morphology import morphology_rows, parse_cfg
from src.preprocess import linguistic_analysis, preprocess_document
from src.samples import SAMPLES
from src.sentiment import analyze_sentiment
from src.wordnet_insights import wordnet_insights


def main() -> None:
    bundle = get_bundle()
    metrics = bundle["metrics"]
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    accuracy_from_matrix = float(np.trace(matrix) / matrix.sum())
    if abs(accuracy_from_matrix - metrics["accuracy"]) > 1e-9:
        raise SystemExit("Accuracy does not match the confusion matrix.")
    if metrics["n_train"] + metrics["n_test"] != metrics["n_total"]:
        raise SystemExit("Train and test counts do not add up to the dataset.")
    if int(matrix.sum()) != metrics["n_test"]:
        raise SystemExit("Confusion matrix does not cover the test set.")
    if not (0.0 < metrics["accuracy"] < 1.0):
        raise SystemExit(f"Unexpected accuracy: {metrics['accuracy']}")
    if metrics["n_categories"] != 5:
        raise SystemExit("Expected five categories.")

    expectations = {
        "technology": ("Technology", "Positive"),
        "sports": ("Sports", "Positive"),
        "politics": ("Politics", "Negative"),
    }
    for sample in SAMPLES:
        processed = preprocess_document(sample["text"])
        linguistics = linguistic_analysis(sample["text"])
        if processed != linguistics["processed"]:
            raise SystemExit("Displayed preprocessing does not match the classifier input.")
        prediction = predict_category(bundle, processed)
        sentiment = analyze_sentiment(sample["text"])
        expected_category, expected_sentiment = expectations[sample["id"]]
        print(
            f"{sample['id']}: NB={prediction['category']} "
            f"({prediction['probabilities'][0]['probability']:.2f}) "
            f"sentiment={sentiment['label']} ({sentiment['compound']:+.3f})"
        )
        if prediction["category"] != expected_category:
            raise SystemExit(f"{sample['id']} was classified as {prediction['category']}")
        if sentiment["label"] != expected_sentiment:
            raise SystemExit(f"{sample['id']} sentiment was {sentiment['label']}")

        rows = morphology_rows(sample["text"])
        if not rows:
            raise SystemExit("Morphology produced no rows.")
        cfg = parse_cfg(sample["text"])
        if not cfg["ok"] or cfg["simplified"]:
            raise SystemExit(f"CFG did not fully parse the {sample['id']} sample: {cfg}")
        similar = similar_articles(bundle, processed)
        if len(similar) != 3:
            raise SystemExit("Retrieval did not return 3 articles.")
        if not all(0 <= item["score"] <= 1 for item in similar):
            raise SystemExit("Similarity scores are out of range.")

    tech_lemmas = {row["lemma"] for row in morphology_rows(SAMPLES[0]["text"], limit=40)}
    if "company" not in tech_lemmas or "technology" not in tech_lemmas or "run" not in tech_lemmas:
        raise SystemExit(f"Expected lemmas were missing: {sorted(tech_lemmas)}")

    insights = wordnet_insights(
        ["chip", "technology", "minister"],
        {"chip": "NN", "technology": "NN", "minister": "NN"},
        category="Technology",
        context=["computer", "software", "processor", "digital"],
    )
    if len(insights) < 2 or not insights[0]["definition"]:
        raise SystemExit("WordNet lookup failed.")
    if not any(word in insights[0]["definition"] for word in ("electronic", "semiconductor", "circuit")):
        raise SystemExit(f"Technology sense of chip was not selected: {insights[0]}")

    agreed = build_comparison("Sports", "Positive", {"ok": True, "category": "Sports", "sentiment": "Positive", "explanation": "Because sport."})
    if agreed["category_agreement"] is not True or "same category" not in agreed["category_message"]:
        raise SystemExit("Agreement detection failed.")
    split = build_comparison("Sports", "Positive", {"ok": True, "category": "Business", "sentiment": "Negative", "explanation": "Markets."})
    if split["category_agreement"] is not False or "disagreed" not in split["category_message"].lower():
        raise SystemExit("Disagreement detection failed.")
    missing = build_comparison("Sports", "Positive", {"ok": False, "error": "down"})
    if missing["category_agreement"] is not None:
        raise SystemExit("Missing LLM should not be scored as agreement or disagreement.")

    print(
        f"Test accuracy {metrics['accuracy']:.3f} · macro F1 {metrics['f1_macro']:.3f} · "
        f"errors {metrics['n_errors']}/{metrics['n_test']}"
    )
    print("Self-check passed.")


if __name__ == "__main__":
    main()

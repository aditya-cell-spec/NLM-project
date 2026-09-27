"""Train, save, and apply the news category classifier.

Pipeline, fit only on the training split:
    raw article → preprocess_document → TF-IDF → Multinomial Naive Bayes

The held-out test split is used once, here, to compute accuracy, precision,
recall, F1, and the confusion matrix. User articles are never added to
those scores.

A second TF-IDF matrix is fit on the full local collection for retrieval
only. It is not the matrix inside the classifier, so test articles in the
search index do not leak into the evaluation.
"""

from __future__ import annotations

from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.bootstrap import ensure_nltk
from src.constants import (
    CATEGORIES,
    DATA_PATH,
    LABEL_MAP,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.preprocess import preprocess_many

# Re-exported for callers that already import categories from this module.
__all__ = ["CATEGORIES", "get_bundle", "predict_category", "similar_articles", "train_and_save"]


def _snippet(text: str, limit: int = 180) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _title(text: str) -> str:
    for line in (text or "").splitlines():
        title = " ".join(line.split())
        if title:
            return title[:110]
    return "Untitled article"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "The BBC news CSV should live at data/bbc_news.csv."
        )
    frame = pd.read_csv(DATA_PATH, encoding="latin-1")
    if not {"news", "type"}.issubset(frame.columns):
        raise ValueError("Expected columns 'news' and 'type' in the BBC dataset.")

    frame = frame.dropna(subset=["news", "type"]).copy()
    frame["news"] = frame["news"].astype(str)
    frame["category"] = frame["type"].str.strip().str.lower().map(LABEL_MAP)
    frame = frame.dropna(subset=["category"])
    frame = frame[frame["news"].str.strip().str.len() > 0]
    return frame.reset_index(drop=True)


def _diversified_errors(rows: list[dict], limit: int = 6) -> list[dict]:
    """Keep a mix of confusion pairs instead of six copies of the same mistake."""
    picked: list[dict] = []
    used: Counter = Counter()
    for row in rows:
        key = (row["actual"], row["predicted"])
        if used[key] >= 2:
            continue
        picked.append(row)
        used[key] += 1
        if len(picked) >= limit:
            return picked
    if len(picked) < limit:
        seen = {id(row) for row in picked}
        for row in rows:
            if id(row) in seen:
                continue
            picked.append(row)
            if len(picked) >= limit:
                break
    return picked


def train_and_save() -> dict:
    """Fit the classifier, score the test set, build the search index, and save."""
    ensure_nltk()
    print("Loading the BBC news dataset…")
    frame = load_dataset()
    texts = frame["news"].tolist()
    labels = frame["category"].tolist()

    train_text, test_text, train_y, test_y = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
        shuffle=True,
    )

    print(f"Preprocessing {len(train_text)} training and {len(test_text)} test articles…")
    train_processed = preprocess_many(train_text)
    test_processed = preprocess_many(test_text)

    # TF-IDF: term frequency, down-weighted by how common the term is in the corpus.
    # sublinear_tf uses log(1 + tf) so one repeated word does not dominate.
    # ngram_range keeps single words and two-word phrases ("prime minister").
    # The vectorizer is fit on the training articles only.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=6000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            # alpha=1 is Laplace smoothing: unseen words do not force a probability of zero.
            ("nb", MultinomialNB(alpha=1.0)),
        ]
    )
    pipeline.fit(train_processed, train_y)

    predicted = pipeline.predict(test_processed)
    accuracy = float(accuracy_score(test_y, predicted))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_y, predicted, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        test_y, predicted, average="weighted", zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        test_y, predicted, labels=CATEGORIES, average=None, zero_division=0
    )
    matrix = confusion_matrix(test_y, predicted, labels=CATEGORIES)

    majority_label, _majority_count = Counter(train_y).most_common(1)[0]
    baseline = sum(label == majority_label for label in test_y) / len(test_y)

    per_class = []
    for index, category in enumerate(CATEGORIES):
        per_class.append(
            {
                "category": category,
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
        )

    error_rows = []
    for actual, guess, raw in zip(test_y, predicted, test_text):
        if actual == guess:
            continue
        error_rows.append(
            {
                "actual": actual,
                "predicted": guess,
                "title": _title(raw),
                "snippet": _snippet(raw),
            }
        )
    confused = []
    for row_index, actual in enumerate(CATEGORIES):
        for col_index, guess in enumerate(CATEGORIES):
            count = int(matrix[row_index, col_index])
            if row_index != col_index and count > 0:
                confused.append({"actual": actual, "predicted": guess, "count": count})
    confused.sort(key=lambda item: -item["count"])

    train_counts = Counter(train_y)
    test_counts = Counter(test_y)

    # Separate retrieval index over the whole local collection.
    print(f"Building the TF-IDF search index for {len(texts)} articles…")
    collection_processed = preprocess_many(texts)
    retrieval_vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    retrieval_matrix = retrieval_vectorizer.fit_transform(collection_processed)
    records = [
        {
            "title": _title(raw),
            "category": category,
            "snippet": _snippet(raw, 220),
        }
        for raw, category in zip(texts, labels)
    ]

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "labels": list(CATEGORIES),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "n_total": len(labels),
        "n_categories": len(CATEGORIES),
        "train_counts": {category: int(train_counts[category]) for category in CATEGORIES},
        "test_counts": {category: int(test_counts[category]) for category in CATEGORIES},
        "majority_label": majority_label,
        "majority_baseline": float(baseline),
        "errors": _diversified_errors(error_rows, limit=6),
        "n_errors": len(error_rows),
        "confused_pairs": confused[:5],
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
    }

    bundle = {
        "version": MODEL_VERSION,
        "pipeline": pipeline,
        "metrics": metrics,
        "retrieval_vectorizer": retrieval_vectorizer,
        "retrieval_matrix": retrieval_matrix,
        "records": records,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(
        f"Saved model to {MODEL_PATH} · accuracy {accuracy:.3f} · "
        f"macro F1 {float(f1_macro):.3f}"
    )
    return bundle


def get_bundle() -> dict:
    """Load the saved model, or train it once if the file is missing or stale."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("version") == MODEL_VERSION and "metrics" in bundle:
                return bundle
        except Exception:
            # A broken or incompatible file should not stop the demo.
            pass
    return train_and_save()


def indicative_terms(pipeline: Pipeline, category: str, k: int = 6) -> list[str]:
    """Words that most separate this category from the others in the trained model."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["nb"]
    names = vectorizer.get_feature_names_out()
    classes = list(classifier.classes_)
    index = classes.index(category)
    log_prob = classifier.feature_log_prob_
    others = np.delete(log_prob, index, axis=0).mean(axis=0)
    salience = log_prob[index] - others
    top = np.argsort(salience)[-k:][::-1]
    return [str(names[i]) for i in top]


def predict_category(bundle: dict, processed_text: str) -> dict:
    """Classify one already-preprocessed article and return real probabilities."""
    if not processed_text.strip():
        raise ValueError(
            "After preprocessing, no content words were left. Paste a longer news article."
        )

    pipeline: Pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba([processed_text])[0]
    classes = list(pipeline.classes_)
    order = np.argsort(probabilities)[::-1]
    ranked = [
        {"category": classes[index], "probability": float(probabilities[index])}
        for index in order
    ]
    predicted = ranked[0]["category"]

    vectorizer = pipeline.named_steps["tfidf"]
    weights = vectorizer.transform([processed_text])
    feature_names = vectorizer.get_feature_names_out()
    columns = weights.nonzero()[1]
    scored = sorted(
        ((str(feature_names[col]), float(weights[0, col])) for col in columns),
        key=lambda item: -item[1],
    )

    return {
        "category": predicted,
        "probabilities": ranked,
        "top_tokens": [{"token": token, "weight": round(weight, 4)} for token, weight in scored[:8]],
        "indicative_terms": indicative_terms(pipeline, predicted),
    }


def similar_articles(bundle: dict, processed_text: str, k: int = 3) -> list[dict]:
    """Top-k collection articles by cosine similarity of TF-IDF vectors."""
    if not processed_text.strip():
        return []
    vector = bundle["retrieval_vectorizer"].transform([processed_text])
    scores = cosine_similarity(vector, bundle["retrieval_matrix"]).ravel()
    order = np.argsort(scores)[::-1]
    found = []
    for index in order:
        score = float(scores[index])
        # Skip a near-copy if the user pasted an article from the collection.
        if score > 0.999:
            continue
        record = bundle["records"][int(index)]
        found.append(
            {
                "rank": len(found) + 1,
                "title": record["title"],
                "category": record["category"],
                "score": round(score, 3),
                "snippet": record["snippet"],
            }
        )
        if len(found) >= k:
            break
    return found


if __name__ == "__main__":
    train_and_save()

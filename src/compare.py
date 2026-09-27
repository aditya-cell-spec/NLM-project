"""Turn the two pipeline outputs into an agreement decision.

Nothing here is typed in by hand. Category and sentiment strings are
compared after both models have returned.
"""

from __future__ import annotations

CLASSICAL_EXPLANATION = (
    "Statistical classification. Multinomial Naive Bayes picks the category "
    "with the highest posterior probability given TF-IDF word and bigram weights. "
    "It does not write a prose reason. The probability bars are the result."
)


def build_comparison(nb_category: str, nb_sentiment: str, llm: dict) -> dict:
    base = {
        "nb_category": nb_category,
        "nb_sentiment": nb_sentiment,
        "nb_explanation": CLASSICAL_EXPLANATION,
        "llm_available": bool(llm.get("ok")),
    }
    if not llm.get("ok"):
        base.update(
            {
                "llm_category": "—",
                "llm_sentiment": "—",
                "llm_explanation": llm.get("error") or "The LLM did not return a result.",
                "category_agreement": None,
                "sentiment_agreement": None,
                "category_message": "Category agreement was not scored because the LLM result is missing.",
                "sentiment_message": "Sentiment agreement was not scored because the LLM result is missing.",
                "approach_note": (
                    "Classical NLP still ran. Start Ollama if you want the side-by-side comparison."
                ),
            }
        )
        return base

    llm_category = llm["category"]
    llm_sentiment = llm["sentiment"]
    cat_agree = nb_category == llm_category
    sent_agree = nb_sentiment == llm_sentiment

    if cat_agree:
        category_message = "Both systems predicted the same category."
    else:
        category_message = (
            f"The models disagreed. The classical model predicted {nb_category} "
            f"while the LLM predicted {llm_category}."
        )

    if sent_agree:
        sentiment_message = "Both systems predicted the same sentiment."
    else:
        sentiment_message = (
            f"The models disagreed. The classical model predicted {nb_sentiment} "
            f"while the LLM predicted {llm_sentiment}."
        )

    if cat_agree and sent_agree:
        approach_note = (
            "The two pipelines agree on this article. They can still be wrong together: "
            "agreement is not the same thing as proof."
        )
    else:
        approach_note = (
            "This is a disagreement between two approaches, not a verdict that one of them "
            "is automatically correct. Naive Bayes is a statistical classifier trained on "
            "labelled news. The LLM is a prompted language model and can fail in a different way."
        )

    base.update(
        {
            "llm_category": llm_category,
            "llm_sentiment": llm_sentiment,
            "llm_explanation": llm.get("explanation") or "",
            "category_agreement": cat_agree,
            "sentiment_agreement": sent_agree,
            "category_message": category_message,
            "sentiment_message": sentiment_message,
            "approach_note": approach_note,
        }
    )
    return base

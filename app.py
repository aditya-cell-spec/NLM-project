"""NewsLens — one page that compares classical NLP with a local LLM.

Run from the project root:

    streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st

from src.analyze import analyze_article
from src.constants import CATEGORIES
from src.model import get_bundle
from src.morphology import GRAMMAR_NOTES, GRAMMAR_TEMPLATE
from src.plots import confusion_figure, tree_figure
from src.samples import SAMPLES

st.set_page_config(
    page_title="NewsLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f0e6; }
      .block-container { padding-top: 1.4rem; max-width: 1180px; }
      h1, h2, h3 { font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; }
      .masthead {
        border-top: 4px solid #a24b2d;
        border-bottom: 1px solid #1c1915;
        padding: 0.75rem 0 0.9rem;
        margin-bottom: 0.4rem;
      }
      .mast-kicker {
        font-family: system-ui, sans-serif;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.74rem;
        color: #a24b2d;
        margin: 0;
      }
      .masthead h1 {
        font-size: 3.1rem;
        line-height: 0.95;
        margin: 0.2rem 0 0.25rem;
        letter-spacing: -0.03em;
      }
      .mast-sub { margin: 0; font-size: 1.12rem; color: #3f3832; }
      .section-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        border-bottom: 1px solid #e4dccf;
        margin: 1.35rem 0 0.7rem;
      }
      .section-head h2 { margin: 0; font-size: 1.55rem; }
      .section-head span {
        color: #6b645b;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: system-ui, sans-serif;
        text-align: right;
      }
      .prob-row {
        display: grid;
        grid-template-columns: 124px 1fr 52px;
        gap: 8px;
        align-items: center;
        margin: 0.28rem 0;
        font-family: system-ui, sans-serif;
        font-size: 0.92rem;
      }
      .prob-track { background: #efe8dc; border-radius: 999px; height: 10px; }
      .prob-fill { background: #1c6b58; height: 10px; border-radius: 999px; }
      .prob-fill.indigo { background: #3c4a86; }
      .pill {
        display: inline-block;
        padding: 0.12rem 0.55rem;
        border-radius: 999px;
        font-family: system-ui, sans-serif;
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 0.35rem;
      }
      .pill-teal { background: #e5f3ef; color: #145444; }
      .pill-indigo { background: #eceef8; color: #2c3870; }
      .pill-good { background: #e5f5ec; color: #14613a; }
      .pill-warn { background: #fbf1d8; color: #7a5200; }
      .pill-ink { background: #efe8dc; color: #1c1915; }
      .prose { font-size: 0.98rem; line-height: 1.5; margin: 0.15rem 0 0.45rem; }
      .chip {
        display: inline-block;
        background: #fffdf8;
        border: 1px solid #e4dccf;
        border-radius: 999px;
        padding: 0.12rem 0.55rem;
        margin: 0.12rem 0.2rem 0.12rem 0;
        font-family: system-ui, sans-serif;
        font-size: 0.84rem;
      }
      .tiny {
        color: #6b645b;
        font-family: system-ui, sans-serif;
        font-size: 0.84rem;
        margin-top: 0.15rem;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str, note: str) -> None:
    st.markdown(
        f'<div class="section-head"><h2>{html.escape(title)}</h2>'
        f"<span>{html.escape(note)}</span></div>",
        unsafe_allow_html=True,
    )


def prose(text: str) -> None:
    st.markdown(f"<div class='prose'>{html.escape(text or '')}</div>", unsafe_allow_html=True)


def chips(items: list[str]) -> None:
    if not items:
        st.caption("None found.")
        return
    html_chips = "".join(f"<span class='chip'>{html.escape(item)}</span>" for item in items)
    st.markdown(html_chips, unsafe_allow_html=True)


def probability_bars(ranked: list[dict]) -> None:
    rows = []
    for item in ranked:
        width = max(0.0, min(100.0, item["probability"] * 100))
        rows.append(
            "<div class='prob-row'>"
            f"<div>{html.escape(item['category'])}</div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{width:.1f}%'></div></div>"
            f"<div>{width:.1f}%</div>"
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def agreement_pill(flag: bool | None, agree_text: str = "Agreement", disagree_text: str = "Disagreement") -> str:
    if flag is True:
        return f"<span class='pill pill-good'>{agree_text}</span>"
    if flag is False:
        return f"<span class='pill pill-warn'>{disagree_text}</span>"
    return "<span class='pill pill-ink'>Not scored</span>"


def use_sample(text: str) -> None:
    st.session_state.article = text
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_error", None)


@st.cache_resource(show_spinner=False)
def load_model():
    # Tokenizer, tagger, and WordNet are needed as soon as an article is analysed.
    from src.bootstrap import ensure_nltk

    ensure_nltk()
    return get_bundle()


st.markdown(
    """
    <div class="masthead">
      <p class="mast-kicker">College NLP demonstration</p>
      <h1>NewsLens</h1>
      <p class="mast-sub">Classical NLP vs LLM for Intelligent News Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "One article goes through two separate pipelines: TF-IDF + Multinomial Naive Bayes, "
    "and a local language model. Scores in Model performance come from the held-out test set, not from this box."
)

with st.spinner("Loading the trained Naive Bayes model…"):
    bundle = load_model()
metrics = bundle["metrics"]

if "article" not in st.session_state:
    st.session_state.article = ""

st.text_area(
    "News article",
    key="article",
    height=220,
    placeholder="Paste a news article here…",
    label_visibility="collapsed",
)

sample_cols = st.columns(len(SAMPLES))
for column, sample in zip(sample_cols, SAMPLES):
    column.button(
        sample["label"],
        on_click=use_sample,
        args=(sample["text"],),
        width="stretch",
    )
analyze_clicked = st.button("Analyze Article", type="primary", width="stretch")

if analyze_clicked:
    try:
        with st.spinner("Preprocessing, classifying, retrieving similar articles, and asking the LLM…"):
            st.session_state.analysis = analyze_article(st.session_state.article, bundle)
            st.session_state.analysis_error = None
    except ValueError as exc:
        st.session_state.analysis = None
        st.session_state.analysis_error = str(exc)
    except Exception as exc:
        st.session_state.analysis = None
        st.session_state.analysis_error = f"Analysis failed: {exc}"

if st.session_state.get("analysis_error"):
    st.error(st.session_state.analysis_error)

analysis = st.session_state.get("analysis")
if analysis:
    submitted = " ".join(st.session_state.article.split())[:12000]
    if submitted != analysis["original"]:
        st.info("The text box has changed. Click Analyze Article to refresh the results.")

if analysis is None:
    st.info(
        "Paste an article, or load a sample, then click Analyze Article. "
        "You can already scroll to Model performance — those figures are from the test set."
    )
else:
    linguistics = analysis["linguistics"]
    prediction = analysis["prediction"]
    sentiment = analysis["sentiment"]
    llm = analysis["llm"]
    comparison = analysis["comparison"]

    section("NLP preprocessing", "CO1 · language basics    CO2 · tokens and tags")
    if analysis["truncated"]:
        st.caption("Only the first 12,000 characters were analysed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cleaned tokens", linguistics["token_count"])
    c2.metric("Content lemmas", linguistics["lemma_count"])
    c3.metric("Processing result", "Ready for TF-IDF" if linguistics["lemma_count"] else "Too short")

    with st.expander("NLP analysis — original text, processed text, tokens, POS tags", expanded=True):
        st.markdown("**Original text**")
        preview = analysis["original"]
        if len(preview) > 1400:
            preview = preview[:1400].rstrip() + "…"
        st.text(preview)

        st.markdown("**Processed text**")
        st.caption("Lowercased, punctuation removed, stop words removed, then lemmatized with the POS tag.")
        processed_preview = linguistics["processed"]
        if len(processed_preview) > 1400:
            processed_preview = processed_preview[:1400].rstrip() + "…"
        st.text(processed_preview or "(no content words)")

        st.markdown("**Important tokens**")
        st.caption("Highest TF-IDF weights in this article, using the vectorizer that was fit on the training set.")
        chips(
            [f"{item['token']}  {item['weight']:.3f}" for item in prediction["top_tokens"]]
            or ["No features matched the vocabulary."]
        )

        st.markdown("**POS tags**")
        st.caption("First 30 cleaned tokens. Stop words are tagged, then removed before classification.")
        st.dataframe(
            [
                {
                    "Token": row["token"],
                    "Penn tag": row["pos"],
                    "Part of speech": row["readable_pos"],
                    "Decision": row["decision"],
                    "Lemma": row["lemma"],
                }
                for row in linguistics["token_rows"]
            ],
            width="stretch",
            hide_index=True,
        )
        if linguistics["token_rows_truncated"]:
            st.caption("The table shows the opening of the article so the page stays readable.")

    section("Classical NLP and LLM", "Same article · two independent pipelines")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(
                "<span class='pill pill-teal'>Classical NLP</span> Multinomial Naive Bayes",
                unsafe_allow_html=True,
            )
            st.markdown("<p class='tiny'>PREDICTED CATEGORY</p>", unsafe_allow_html=True)
            st.subheader(prediction["category"])
            st.markdown("**Category probabilities**")
            st.caption("From `predict_proba` on this article. These are not accuracy.")
            probability_bars(prediction["probabilities"])
            st.markdown("**Sentiment**")
            st.subheader(sentiment["label"])
            st.caption(
                f"VADER compound {sentiment['compound']:+.3f}. "
                "≥ 0.05 Positive, ≤ −0.05 Negative, otherwise Neutral."
            )
            st.caption(
                f"Lexicon proportions — positive {sentiment['positive']:.2f}, "
                f"neutral {sentiment['neutral']:.2f}, negative {sentiment['negative']:.2f}."
            )
            if sentiment["positive_words"] or sentiment["negative_words"]:
                pos_bits = ", ".join(
                    f"{item['word']} ({item['valence']:+.1f})" for item in sentiment["positive_words"]
                )
                neg_bits = ", ".join(
                    f"{item['word']} ({item['valence']:+.1f})" for item in sentiment["negative_words"]
                )
                if pos_bits:
                    st.caption(f"Positive lexicon hits: {pos_bits}")
                if neg_bits:
                    st.caption(f"Negative lexicon hits: {neg_bits}")
            st.markdown("**Processing result**")
            prose(
                f"{linguistics['lemma_count']} content lemmas were passed to TF-IDF and "
                f"Multinomial Naive Bayes. The model chose {prediction['category']}."
            )
            st.markdown("**Terms the classifier associates with this category**")
            chips(prediction["indicative_terms"])

    with right:
        with st.container(border=True):
            model_name = llm.get("model") or "llama3.2"
            st.markdown(
                f"<span class='pill pill-indigo'>LLM</span> {html.escape(model_name)}",
                unsafe_allow_html=True,
            )
            if not llm.get("ok"):
                st.subheader("Unavailable")
                st.error(llm.get("error") or "The language model did not return a result.")
                st.caption("Category, sentiment, summary, and explanation appear here when Ollama answers.")
            else:
                st.markdown("<p class='tiny'>PREDICTED CATEGORY</p>", unsafe_allow_html=True)
                st.subheader(llm["category"])
                st.markdown("**Sentiment**")
                st.subheader(llm["sentiment"])
                st.markdown("**Summary**")
                prose(llm["summary"])
                st.markdown("**Explanation**")
                prose(llm["explanation"])
                st.caption(
                    "The model was instructed to return JSON and to use only "
                    + ", ".join(CATEGORIES)
                    + ". Summary is the sequence-to-sequence reading of the article."
                )

    section("Classical NLP vs LLM", "Agreement is computed from the two outputs")
    st.markdown(
        "<table style='width:100%; border-collapse:collapse; background:#fffdf8;'>"
        "<thead><tr>"
        "<th style='text-align:left; padding:8px; border-bottom:1px solid #e4dccf;'>Aspect</th>"
        "<th style='text-align:left; padding:8px; border-bottom:1px solid #e4dccf;'>Naive Bayes</th>"
        "<th style='text-align:left; padding:8px; border-bottom:1px solid #e4dccf;'>LLM</th>"
        "</tr></thead><tbody>"
        f"<tr><td style='padding:8px; border-bottom:1px solid #f0e9de;'>Category</td>"
        f"<td style='padding:8px; border-bottom:1px solid #f0e9de;'>{html.escape(comparison['nb_category'])}</td>"
        f"<td style='padding:8px; border-bottom:1px solid #f0e9de;'>{html.escape(comparison['llm_category'])}</td></tr>"
        f"<tr><td style='padding:8px; border-bottom:1px solid #f0e9de;'>Sentiment</td>"
        f"<td style='padding:8px; border-bottom:1px solid #f0e9de;'>{html.escape(comparison['nb_sentiment'])}</td>"
        f"<td style='padding:8px; border-bottom:1px solid #f0e9de;'>{html.escape(comparison['llm_sentiment'])}</td></tr>"
        f"<tr><td style='padding:8px; vertical-align:top;'>Explanation</td>"
        f"<td style='padding:8px; vertical-align:top;'>{html.escape(comparison['nb_explanation'])}</td>"
        f"<td style='padding:8px; vertical-align:top;'>{html.escape(comparison['llm_explanation'])}</td></tr>"
        "</tbody></table>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    a, b = st.columns(2)
    with a:
        st.markdown(
            agreement_pill(comparison["category_agreement"]) + " Category",
            unsafe_allow_html=True,
        )
        prose(comparison["category_message"])
    with b:
        st.markdown(
            agreement_pill(comparison["sentiment_agreement"]) + " Sentiment",
            unsafe_allow_html=True,
        )
        prose(comparison["sentiment_message"])
    st.info(comparison["approach_note"])

section("Model performance", "CO3 · scored on the held-out test set only")
st.caption(
    f"Stratified shuffle split, test size {metrics['test_size']:.0%}, random seed {metrics['seed']}. "
    "Precision, recall, and F1 on the cards are macro averages: each category counts equally. "
    "Nothing on this row is computed from the article box."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Precision", f"{metrics['precision_macro']:.1%}")
m3.metric("Recall", f"{metrics['recall_macro']:.1%}")
m4.metric("F1-score", f"{metrics['f1_macro']:.1%}")

n1, n2, n3 = st.columns(3)
n1.metric("Training samples", f"{metrics['n_train']:,}")
n2.metric("Test samples", f"{metrics['n_test']:,}")
n3.metric("Categories", metrics["n_categories"])
st.caption(
    f"Majority-class baseline (always predict {metrics['majority_label']}): "
    f"{metrics['majority_baseline']:.1%}. "
    f"Weighted F1, which gives larger categories more influence: {metrics['f1_weighted']:.1%}."
)

chart_col, table_col = st.columns([1.15, 0.85])
with chart_col:
    figure = confusion_figure(metrics["confusion_matrix"], metrics["labels"])
    st.pyplot(figure, clear_figure=True)
    st.caption("Rows are the true category. Columns are what Naive Bayes predicted. The diagonal is correct.")
with table_col:
    st.markdown("**Per-category scores on the test set**")
    st.dataframe(
        [
            {
                "Category": row["category"],
                "Precision": f"{row['precision']:.3f}",
                "Recall": f"{row['recall']:.3f}",
                "F1": f"{row['f1']:.3f}",
                "Test articles": row["support"],
            }
            for row in metrics["per_class"]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Train / test counts — "
        + ", ".join(
            f"{name} {metrics['train_counts'][name]}/{metrics['test_counts'][name]}"
            for name in metrics["labels"]
        )
    )

section("Error analysis", "CO3 · where the test-set classifier was wrong")
st.caption(
    f"Naive Bayes misclassified {metrics['n_errors']} of {metrics['n_test']} test articles. "
    "The rows below are real test articles, shortened."
)
if metrics["confused_pairs"]:
    pair_text = "; ".join(
        f"{pair['actual']} predicted as {pair['predicted']} ({pair['count']})"
        for pair in metrics["confused_pairs"]
    )
    prose(f"Most frequent confusions: {pair_text}.")
else:
    prose("The confusion matrix has no off-diagonal counts on this split.")

if metrics["errors"]:
    st.dataframe(
        [
            {
                "Actual category": str(row["actual"]),
                "Predicted category": str(row["predicted"]),
                "Text snippet": row["snippet"],
            }
            for row in metrics["errors"]
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.success("The classifier made no mistakes on this test split.")

if analysis and comparison["category_agreement"] is False:
    st.warning(comparison["category_message"] + " " + comparison["approach_note"])
elif analysis and comparison["sentiment_agreement"] is False:
    st.warning(comparison["sentiment_message"] + " " + comparison["approach_note"])

if analysis:
    section("Information retrieval", "CO4 · TF-IDF and cosine similarity")
    st.caption(
        "A separate TF-IDF index over the local BBC collection. "
        "This index is not the one inside Naive Bayes, and these scores are similarity, not probability."
    )
    if not analysis["similar"]:
        st.info("No similar articles were retrieved.")
    for item in analysis["similar"]:
        with st.container(border=True):
            st.markdown(
                f"**{item['rank']}. {html.escape(item['title'])}**",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='pill pill-ink'>{html.escape(item['category'])}</span>"
                f"<span class='pill pill-teal'>cosine {item['score']:.3f}</span>",
                unsafe_allow_html=True,
            )
            prose(item["snippet"])

    section("WordNet and NLP insights", "CO2 · morphology and parsing    CO4 · WordNet")
    st.markdown("**Morphology — word, lemma, part of speech**")
    st.caption("Lemmas that differ from the surface word are listed first.")
    if analysis["morphology"]:
        st.dataframe(
            [
                {
                    "Word": row["word"],
                    "Lemma": row["lemma"],
                    "Part of speech": row["pos"],
                }
                for row in analysis["morphology"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No content words were available for morphology.")

    st.markdown("**WordNet insights**")
    st.caption("A few nouns or verbs from this article. Synonyms share a synset. Hypernyms are broader concepts.")
    if not analysis["wordnet"]:
        st.info("None of the selected words were found in WordNet.")
    insight_cols = st.columns(max(1, len(analysis["wordnet"])))
    for column, entry in zip(insight_cols, analysis["wordnet"]):
        with column:
            with st.container(border=True):
                st.markdown(f"**{html.escape(entry['word'])}** · {html.escape(entry['pos'])}", unsafe_allow_html=True)
                prose(entry["definition"])
                st.caption("Synonyms")
                chips(entry["synonyms"] or ["—"])
                st.caption("Hypernyms")
                chips(entry["hypernyms"] or ["—"])

    st.markdown("**Shallow parse (chunking)**")
    chunks = analysis["chunks"]
    if not chunks.get("ok"):
        st.info("The article did not contain a sentence long enough to chunk.")
    else:
        st.caption(chunks["sentence"])
        chunk_fig = tree_figure(chunks.get("tree"), "Noun and verb chunks")
        if chunk_fig is not None:
            st.pyplot(chunk_fig, clear_figure=True)
        with st.expander("Chunk tree as text"):
            st.code(chunks["tree_text"] or "", language="text")

    st.markdown("**Context-free grammar**")
    st.code(GRAMMAR_TEMPLATE, language="text")
    st.caption(GRAMMAR_NOTES)
    cfg = analysis["cfg"]
    if cfg.get("ok"):
        kind = "Simplified sentence" if cfg["simplified"] else "Parsed sentence"
        st.markdown(f"**{kind}:** {html.escape(cfg['sentence'])}", unsafe_allow_html=True)
        prose(cfg["note"])
        cfg_fig = tree_figure(cfg.get("tree"), "CFG parse")
        if cfg_fig is not None:
            st.pyplot(cfg_fig, clear_figure=True)
        with st.expander("CFG tree as text"):
            st.code(cfg["tree_text"], language="text")
    else:
        st.info(cfg["note"])

st.caption(
    "Dataset: BBC news articles (business, entertainment, politics, sport, tech), "
    "as released for research by Greene and Cunningham, ICML 2006. "
    "Labels in the app are Business, Entertainment, Politics, Sports, and Technology. "
    "Sentiment uses NLTK VADER. WordNet lookups use NLTK WordNet."
)

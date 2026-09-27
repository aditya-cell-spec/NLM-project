# NewsLens: Classical NLP vs LLM for Intelligent News Analysis

A single-page college project. Paste one news article and the page runs **two separate pipelines** on it:

1. **Classical NLP** — cleaning, lemmatization, TF-IDF, Multinomial Naive Bayes, and VADER sentiment.
2. **Local LLM** — category, sentiment, a short summary, and a written explanation (Ollama).

The page then compares the two outputs. It also shows test-set metrics, a confusion matrix, similar articles, WordNet, and a small parser.

This is a classroom demo, not a production system. There is no login, database, or multi-page dashboard.

## What you can show in a viva (about 8 minutes)

1. Open the app. The **Model performance** cards are already filled from the held-out test set.
2. Point at accuracy, macro precision, recall, F1, the training/test counts, and the confusion matrix.
3. Click **Technology sample** (or paste your own article) and **Analyze Article**.
4. Open **NLP analysis**: original text, lemmas, TF-IDF tokens, POS tags.
5. Read the **Naive Bayes** card: category, `predict_proba` bars, VADER sentiment.
6. Read the **LLM** card. If Ollama is off, that card explains how to start it. The classical side still works.
7. Read **Classical NLP vs LLM**. Agreement is decided in code from the two outputs.
8. Show **Similar articles** (TF-IDF cosine similarity) and **WordNet** (synonyms and hypernyms).
9. Show the morphology table (`companies` → `company`) and the CFG tree for the short opening sentence.
10. Show **Error analysis**: real test articles the classifier got wrong, and which categories it mixes up.

## Syllabus map

| Outcome | Where it is in the app |
| --- | --- |
| CO1 Fundamental NLP, language models, an NLP application | Preprocessing, the news classifier, and the LLM summary |
| CO2 Morphological analysis and parsing | Lemma table, POS tags, chunk tree, classroom CFG |
| CO3 Text classification and sentiment with Naive Bayes | TF-IDF + Multinomial Naive Bayes, test metrics, confusion matrix, error table. Sentiment is VADER, a classical lexicon model, kept separate so you can explain it on its own |
| CO4 Information retrieval, WordNet, LLM / sequence-to-sequence | Similar articles, WordNet synsets, LLM summary (article → short text) |

## How to run

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads NLTK data (tokenizer, tagger, stop words, WordNet, VADER) into `nltk_data/`. A trained model is already saved in `models/newslens.joblib`, so the app does not retrain on every article. If that file is missing, the app trains once and saves it.

Optional checks:

```bash
python -m src.bootstrap   # NLTK data only
python -m src.model       # retrain and print the test scores
python -m src.selfcheck   # metrics, samples, parser, WordNet, retrieval
```

## Local LLM

Copy `.env.example` to `.env` if you want to change the defaults.

```bash
# .env
LLM_MODEL=llama3.2
OLLAMA_URL=http://127.0.0.1:11434
LLM_TIMEOUT=90
```

```bash
ollama pull llama3.2
ollama serve
```

The prompt asks for JSON with `category`, `sentiment`, `summary`, and `explanation`. The category must be one of Business, Sports, Technology, Politics, Entertainment. If Ollama is not running, the LLM card shows that message and the rest of the page still works. The app does not invent an LLM answer.

## Dataset and model

- **File:** `data/bbc_news.csv` (2,225 articles).
- **Source:** the BBC news dataset distributed for research by D. Greene and P. Cunningham, “Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering”, ICML 2006. Categories in the file are `business`, `entertainment`, `politics`, `sport`, and `tech`. The app shows them as Business, Entertainment, Politics, Sports, and Technology.
- **Split:** 80% train / 20% test, stratified, `random_state=42`.
- **Classifier:** `TfidfVectorizer` (unigrams and bigrams, `min_df=2`, `sublinear_tf`, max 6,000 features, **fit on the training set only**) and `MultinomialNB(alpha=1.0)` (Laplace smoothing).
- **Metrics:** scikit-learn `accuracy_score`, `precision_recall_fscore_support`, and `confusion_matrix` on the test predictions. Macro precision, recall, and F1 treat each category equally. A majority-class baseline is shown so the accuracy has something to beat. User text is never added to these scores.
- **Retrieval index:** a second TF-IDF model fit on the whole local collection. It is not the classifier’s vectorizer, so putting test articles in the search index does not leak into the accuracy.

## Preprocessing (same function for training and for the text box)

Lowercase → remove URLs, digits, and punctuation → tokenize → POS tag → drop stop words → lemmatize with the POS tag (`running` + verb → `run`).

## Parsing

The CFG is deliberately small:

```
S  -> NP VP
NP -> Det Nom | Nom
Nom -> Adj Nom | N
VP -> V NP | V
```

The lexicon is built from the sentence. Long news sentences will not match; the page still shows a chunk parse (noun phrases and verb phrases). Each sample article starts with a short sentence this grammar can parse, so the tree is easy to talk through.

## Project layout

```
app.py                  Streamlit page
data/bbc_news.csv       Labelled news
models/newslens.joblib  Saved classifier, metrics, retrieval index
src/preprocess.py       Cleaning, lemmas, POS table
src/model.py            Train / test split, Naive Bayes, cosine retrieval
src/sentiment.py        VADER
src/morphology.py       Lemmas and the CFG / chunker
src/wordnet_insights.py WordNet synonyms and hypernyms
src/llm_client.py       Ollama JSON call
src/compare.py          Agreement / disagreement text
src/analyze.py          One article through both pipelines
```

## Concepts in one line each

- **TF-IDF:** a word weighs more when it is frequent in this article and rare in the training set.
- **Naive Bayes:** picks the category with the highest posterior probability given those weights.
- **Train/test split:** the model is fit on one part of the data and scored on another, so the score is not just memory.
- **Accuracy:** correct test predictions ÷ all test articles.
- **Precision / recall / F1:** precision is “of the articles called Sports, how many were Sports”; recall is “of the real Sports articles, how many were found”; F1 combines them. Macro average means each category counts once.
- **Confusion matrix:** rows are the true category, columns are the prediction. Off-diagonal cells are mistakes.
- **POS tagging and lemmatization:** the tagger labels word class; the lemmatizer reduces an inflected word to its dictionary form.
- **CFG:** rewrite rules that build a tree for a short sentence.
- **WordNet:** a lexical database of meanings, synonyms, and more general concepts (hypernyms).
- **Cosine similarity:** how close two TF-IDF vectors point. It is a retrieval score, not a class probability.
- **LLM:** a local language model prompted to return a structured reading of the same article. It can agree with Naive Bayes and still be wrong, and it can disagree without either side being “the answer”.

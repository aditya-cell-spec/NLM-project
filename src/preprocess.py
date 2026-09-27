"""Shared text preprocessing for training and for user articles.

The classifier must see user text in the same form as the training text:
lowercase, punctuation removed, tokens, stop words removed, then lemmas.
POS tags are used so the lemmatizer knows the word class
("running" as a verb becomes "run", not the noun "running").
"""

from __future__ import annotations

import re
from functools import lru_cache

from nltk import pos_tag, pos_tag_sents, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from src.bootstrap import ensure_nltk

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")

# Short labels for the POS table in the interface.
READABLE_POS = {
    "NN": "Noun",
    "NNS": "Noun (plural)",
    "NNP": "Proper noun",
    "NNPS": "Proper noun (plural)",
    "VB": "Verb",
    "VBD": "Verb (past)",
    "VBG": "Verb (gerund)",
    "VBN": "Verb (past participle)",
    "VBP": "Verb",
    "VBZ": "Verb (3rd person)",
    "JJ": "Adjective",
    "JJR": "Adjective (comparative)",
    "JJS": "Adjective (superlative)",
    "RB": "Adverb",
    "RBR": "Adverb (comparative)",
    "RBS": "Adverb (superlative)",
    "DT": "Determiner",
    "PDT": "Predeterminer",
    "IN": "Preposition",
    "PRP": "Pronoun",
    "PRP$": "Possessive pronoun",
    "CC": "Conjunction",
    "CD": "Number",
    "MD": "Modal",
    "TO": "To",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    "WDT": "Wh-determiner",
    "RP": "Particle",
    "EX": "Existential there",
    "UH": "Interjection",
    "POS": "Possessive ending",
}


def readable_pos(tag: str) -> str:
    return READABLE_POS.get(tag, tag or "Other")


def penn_to_wordnet(tag: str) -> str:
    """Map a Penn Treebank tag onto a WordNet part of speech."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


@lru_cache(maxsize=1)
def _tools():
    ensure_nltk()
    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, digits, and punctuation.

    Digits are removed on purpose, so a sum such as "6bn" becomes the token "bn".
    That leftover is rare, but it is a useful example of what cleaning throws away.
    """
    text = _URL.sub(" ", text or "")
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _lemmas(tagged: list[tuple[str, str]], stops: set[str], lemmatizer: WordNetLemmatizer) -> list[str]:
    lemmas = []
    for word, tag in tagged:
        if word in stops or len(word) < 2:
            continue
        lemmas.append(lemmatizer.lemmatize(word, penn_to_wordnet(tag)))
    return lemmas


def preprocess_document(text: str) -> str:
    """Return the lemma string the Naive Bayes model was trained on."""
    stops, lemmatizer = _tools()
    tokens = word_tokenize(clean_text(text))
    tagged = pos_tag(tokens) if tokens else []
    return " ".join(_lemmas(tagged, stops, lemmatizer))


def preprocess_many(texts: list[str]) -> list[str]:
    """Preprocess a batch. Tagging sentences together is much faster."""
    stops, lemmatizer = _tools()
    token_lists = [word_tokenize(clean_text(text)) for text in texts]
    tagged_sents = pos_tag_sents(token_lists) if token_lists else []
    return [" ".join(_lemmas(tagged, stops, lemmatizer)) for tagged in tagged_sents]


def linguistic_analysis(text: str, preview: int = 30) -> dict:
    """Build the preprocessing view shown for one article.

    Classification uses ``processed``. The token table is only for display,
    but it comes from the same cleaned tokens and the same tagger.
    """
    stops, lemmatizer = _tools()
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tagged = pos_tag(tokens) if tokens else []
    lemmas = _lemmas(tagged, stops, lemmatizer)

    rows = []
    for word, tag in tagged[:preview]:
        removed = word in stops or len(word) < 2
        lemma = "" if removed else lemmatizer.lemmatize(word, penn_to_wordnet(tag))
        rows.append(
            {
                "token": word,
                "pos": tag,
                "readable_pos": readable_pos(tag),
                "decision": "Removed" if removed else "Kept",
                "lemma": lemma or "—",
            }
        )

    return {
        "cleaned": cleaned,
        "processed": " ".join(lemmas),
        "content_lemmas": lemmas,
        "token_count": len(tokens),
        "lemma_count": len(lemmas),
        "token_rows": rows,
        "token_rows_truncated": len(tagged) > preview,
    }

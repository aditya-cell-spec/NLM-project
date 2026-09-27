"""Small WordNet lookups for a few important nouns and verbs.

WordNet (Miller, 1995) is a lexical database. A synset is a set of words
that share a meaning. Hypernyms are more general concepts
("chip" → "equipment" is the kind of relation a viva answer can use).

Many words have several senses. A tiny Lesk-style score picks one:
overlap with the article, plus a bonus if the definition matches the
category Naive Bayes just predicted (so "chip" in a technology story
is the electronic chip, not a potato crisp).
"""

from __future__ import annotations

import re

from nltk.corpus import wordnet as wn

from src.bootstrap import ensure_nltk
from src.preprocess import penn_to_wordnet

_TOKEN = re.compile(r"[a-z]+")

# Extra gloss words that should win when the article itself does not
# repeat the dictionary's wording.
DOMAIN_HINTS = {
    "Technology": ("computer", "electronic", "software", "machine", "device", "circuit", "digital", "processor"),
    "Sports": ("game", "sport", "ball", "player", "team", "score", "match"),
    "Politics": ("government", "political", "minister", "election", "parliament"),
    "Business": ("business", "company", "money", "market", "bank", "financial"),
    "Entertainment": ("film", "music", "perform", "actor", "show", "song"),
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def _overlap(gloss: set[str], words: set[str]) -> int:
    score = 0
    for token in gloss:
        for hint in words:
            if token == hint or (len(hint) >= 5 and len(token) >= 5 and (hint in token or token in hint)):
                score += 1
                break
    return score


def _choose_synset(synsets, word: str, context: set[str], category: str | None):
    """Prefer a sense that fits this article and the predicted category."""
    hints = set(DOMAIN_HINTS.get(category or "", ()))
    # The word itself appears in almost every gloss, so it is not evidence.
    context = {token for token in context if token != word}
    best = synsets[0]
    best_score = -1
    for synset in synsets[:12]:
        gloss = _tokens(synset.definition())
        for lemma in synset.lemmas():
            gloss |= _tokens(lemma.name().replace("_", " "))
        gloss.discard(word)
        score = _overlap(gloss, context) + 3 * _overlap(gloss, hints)
        if score > best_score:
            best_score = score
            best = synset
    return best


def _entry(word: str, penn_tag: str | None, context: set[str], category: str | None) -> dict | None:
    pos = penn_to_wordnet(penn_tag) if penn_tag else None
    synsets = wn.synsets(word, pos=pos) if pos else []
    if not synsets:
        synsets = wn.synsets(word)
    if not synsets:
        return None

    synset = _choose_synset(synsets, word, context, category)
    synonyms: list[str] = []
    for lemma in synset.lemmas():
        name = lemma.name().replace("_", " ")
        if name.lower() != word.lower() and name not in synonyms:
            synonyms.append(name)

    hypernyms = []
    for hyper in synset.hypernyms()[:3]:
        hypernyms.append(hyper.lemmas()[0].name().replace("_", " "))

    pos_name = {"n": "Noun", "v": "Verb", "a": "Adjective", "r": "Adverb", "s": "Adjective"}.get(
        synset.pos(), synset.pos()
    )
    return {
        "word": word,
        "pos": pos_name,
        "definition": synset.definition(),
        "synonyms": synonyms[:5],
        "hypernyms": hypernyms,
    }


def wordnet_insights(
    lemmas: list[str],
    penn_by_lemma: dict[str, str],
    limit: int = 3,
    category: str | None = None,
    context: list[str] | None = None,
) -> list[dict]:
    """Look up a few lemmas that exist in WordNet, using the article as context."""
    ensure_nltk()
    context_words = _tokens(" ".join(context or lemmas))
    found = []
    seen = set()
    for lemma in lemmas:
        if lemma in seen or len(lemma) < 3:
            continue
        seen.add(lemma)
        # Skip obvious bigram leftovers; WordNet is queried with single words.
        if " " in lemma:
            continue
        entry = _entry(lemma, penn_by_lemma.get(lemma), context_words, category)
        if entry is None:
            continue
        found.append(entry)
        if len(found) >= limit:
            break
    return found

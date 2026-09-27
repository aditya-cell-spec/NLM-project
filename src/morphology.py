"""Morphology and a small classroom parser.

Two views, both meant to be explained in a few sentences:

1. Word → lemma → part of speech, using the WordNet lemmatizer and the
   Penn tagger. This is the morphological analysis.
2. A chunker (shallow parse) plus a tiny context-free grammar. Real news
   sentences are long, so the CFG is only applied to short sentences that
   fit the grammar. The chunker still runs on a longer sentence.
"""

from __future__ import annotations

import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree

from src.preprocess import penn_to_wordnet, readable_pos
from src.bootstrap import ensure_nltk

# Teaching grammar: determiner, adjective, noun, verb. Nothing else.
# Productions are fixed. The lexicon is built from the sentence itself.
GRAMMAR_TEMPLATE = """
S -> NP VP
NP -> Det Nom | Nom
Nom -> Adj Nom | N
VP -> V NP | V
""".strip()

GRAMMAR_NOTES = (
    "S = sentence, NP = noun phrase, VP = verb phrase, "
    "Det = determiner, Adj = adjective, N = noun, V = verb."
)

CHUNK_GRAMMAR = r"""
NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
PP: {<IN><NP>}
VP: {<MD>?<VB.*><NP|PP|RB>+}
"""

_ALPHA = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[^a-z]")

# Penn tags accepted by the classroom grammar.
_SYMBOL = {
    "DT": "Det",
    "JJ": "Adj",
    "JJR": "Adj",
    "JJS": "Adj",
    "NN": "N",
    "NNS": "N",
    "NNP": "N",
    "NNPS": "N",
    "VB": "V",
    "VBD": "V",
    "VBG": "V",
    "VBN": "V",
    "VBP": "V",
    "VBZ": "V",
}


def morphology_rows(text: str, limit: int = 12) -> list[dict]:
    """Content words, with lemma changes listed before unchanged words."""
    ensure_nltk()
    from src.preprocess import _tools

    stops, lemmatizer = _tools()
    cleaned_tokens = [
        _WORD.sub("", token.lower())
        for token in word_tokenize(text or "")
        if _ALPHA.search(token)
    ]
    cleaned_tokens = [token for token in cleaned_tokens if token]
    tagged = pos_tag(cleaned_tokens) if cleaned_tokens else []

    rows = []
    for word, tag in tagged:
        if word in stops or len(word) < 2:
            continue
        lemma = lemmatizer.lemmatize(word, penn_to_wordnet(tag))
        rows.append(
            {
                "word": word,
                "lemma": lemma,
                "pos": readable_pos(tag),
                "penn": tag,
                "changed": lemma != word,
            }
        )

    changed = [row for row in rows if row["changed"]]
    unchanged = [row for row in rows if not row["changed"]]
    return (changed + unchanged)[:limit]


def _alpha_tokens(sentence: str) -> list[str]:
    tokens = []
    for token in word_tokenize(sentence):
        word = _WORD.sub("", token.lower())
        if len(word) >= 1 and _ALPHA.search(token):
            tokens.append(word)
    return tokens


def _parse_tokens(tokens: list[tuple[str, str]]) -> Tree | None:
    """Parse a fully mapped token sequence. Return None if it does not fit."""
    if not (3 <= len(tokens) <= 12):
        return None
    if not any(symbol == "V" for _word, symbol in tokens):
        return None
    if not any(symbol == "N" for _word, symbol in tokens):
        return None

    lexicon: dict[str, list[str]] = {}
    for word, symbol in tokens:
        bucket = lexicon.setdefault(symbol, [])
        if word not in bucket:
            bucket.append(word)

    lines = [GRAMMAR_TEMPLATE]
    for symbol, words in lexicon.items():
        rhs = " | ".join(f"'{word}'" for word in words)
        lines.append(f"{symbol} -> {rhs}")

    try:
        grammar = nltk.CFG.fromstring("\n".join(lines))
        parser = nltk.ChartParser(grammar)
        trees = list(parser.parse([word for word, _symbol in tokens]))
    except ValueError:
        return None
    return trees[0] if trees else None


def _tag_symbols(words: list[str]) -> list[tuple[str, str, str]]:
    tagged = pos_tag(words) if words else []
    mapped = []
    for word, tag in tagged:
        symbol = _SYMBOL.get(tag)
        mapped.append((word, tag, symbol))
    return mapped


def parse_cfg(text: str) -> dict:
    """Find a short sentence that the teaching grammar can parse."""
    ensure_nltk()
    sentences = nltk.sent_tokenize(text or "")
    simplified_hit = None

    for sentence in sentences[:8]:
        words = _alpha_tokens(sentence)
        if not (3 <= len(words) <= 14):
            continue
        mapped = _tag_symbols(words)
        if all(symbol for _word, _tag, symbol in mapped):
            tree = _parse_tokens([(word, symbol) for word, _tag, symbol in mapped])
            if tree is not None:
                return {
                    "ok": True,
                    "simplified": False,
                    "sentence": " ".join(words),
                    "tree_text": tree.pformat(margin=50),
                    "tree": tree,
                    "grammar": GRAMMAR_TEMPLATE,
                    "note": "This sentence uses only the word classes in the teaching grammar, so it was parsed in full.",
                }

        reduced = [(word, symbol) for word, _tag, symbol in mapped if symbol]
        if simplified_hit is None and 3 <= len(reduced) <= 12:
            tree = _parse_tokens(reduced)
            if tree is not None:
                simplified_hit = {
                    "ok": True,
                    "simplified": True,
                    "sentence": " ".join(word for word, _symbol in reduced),
                    "tree_text": tree.pformat(margin=50),
                    "tree": tree,
                    "grammar": GRAMMAR_TEMPLATE,
                    "note": (
                        "The original sentence contained words outside this small grammar "
                        "(prepositions, pronouns, and so on). Those words were set aside, "
                        "and the remaining pattern was parsed."
                    ),
                }

    if simplified_hit:
        return simplified_hit

    return {
        "ok": False,
        "simplified": False,
        "sentence": "",
        "tree_text": "",
        "tree": None,
        "grammar": GRAMMAR_TEMPLATE,
        "note": (
            "None of the opening sentences fit this classroom grammar. "
            "The chunk tree above is the shallow parse. A sample article includes "
            "a short sentence written so the CFG can succeed."
        ),
    }


def chunk_sentence(text: str) -> dict:
    """Shallow phrase chunks for the first usable sentence."""
    ensure_nltk()
    sentences = nltk.sent_tokenize(text or "")
    chosen = ""
    tokens: list[str] = []
    for sentence in sentences[:6]:
        tokens = [token for token in word_tokenize(sentence) if _ALPHA.search(token)]
        if len(tokens) >= 4:
            chosen = sentence.strip()
            break
    tokens = tokens[:22]
    if len(tokens) < 4:
        return {"ok": False, "sentence": chosen, "tree_text": "", "tree": None}

    tagged = pos_tag(tokens)
    chunker = nltk.RegexpParser(CHUNK_GRAMMAR)
    tree = chunker.parse(tagged)
    return {
        "ok": True,
        "sentence": " ".join(tokens),
        "tree_text": tree.pformat(margin=60),
        "tree": tree,
    }

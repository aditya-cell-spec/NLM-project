"""Download the small NLTK resources this project needs.

The tagger, tokenizer, stop-word list, WordNet, and VADER lexicon are
classroom resources. They are stored in ./nltk_data and are not committed.
"""

from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parents[1]
NLTK_DIR = ROOT / "nltk_data"

# (nltk.data path, download name)
RESOURCES = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
)


def ensure_nltk() -> Path:
    """Make sure every required NLTK resource can be found."""
    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    path = str(NLTK_DIR)
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)

    missing = []
    for resource_path, _name in RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(resource_path)

    if not missing:
        return NLTK_DIR

    for resource_path, download_name in RESOURCES:
        if resource_path not in missing:
            continue
        ok = nltk.download(download_name, download_dir=path, quiet=True)
        if not ok:
            raise RuntimeError(
                f"Could not download NLTK resource '{download_name}'. "
                "Check your network connection and run: python -m src.bootstrap"
            )
    return NLTK_DIR


if __name__ == "__main__":
    location = ensure_nltk()
    print(f"NLTK data is ready in {location}")

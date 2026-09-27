"""Paths, labels, and the split used everywhere in the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bbc_news.csv"
MODEL_PATH = ROOT / "models" / "newslens.joblib"

# Bump this if preprocessing or the classifier changes, so an old file is retrained.
MODEL_VERSION = 1

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Display labels. The CSV uses the short BBC names on the right.
LABEL_MAP = {
    "business": "Business",
    "sport": "Sports",
    "tech": "Technology",
    "politics": "Politics",
    "entertainment": "Entertainment",
}
CATEGORIES = ["Business", "Sports", "Technology", "Politics", "Entertainment"]

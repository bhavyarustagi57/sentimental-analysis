"""TF-IDF feature construction shared by every classifier."""

from __future__ import annotations

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import MAX_FEATURES, MIN_DF, NGRAM_RANGE


_NEGATIONS = {"no", "nor", "not", "never", "neither", "cannot"}
NEGATION_SAFE_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - _NEGATIONS)


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words=NEGATION_SAFE_STOP_WORDS,
        sublinear_tf=True,
        strip_accents="unicode",
    )

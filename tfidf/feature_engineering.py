"""
TF-IDF Feature Engineering
-----------------------------
Reusable TF-IDF pipeline for the Logistic Regression models.
Wrapped as functions (not one-off notebook code) so they can be
re-applied identically to new data during Week 4 retraining.

The DistilBERT tokenizer equivalent lives in distilbert/tokenization.py.
"""

import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_tfidf_features(texts: pd.Series, max_features: int = 5000,
                          vectorizer: TfidfVectorizer = None):
    """
    Fit (or reuse) a TF-IDF vectorizer on heavy-cleaned text.

    Pass an existing `vectorizer` when transforming new data (e.g. at
    inference time or during retraining) so the vocabulary stays consistent.

    Returns: (feature_matrix, fitted_vectorizer)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        matrix = vectorizer.fit_transform(texts)
        logger.info(f"Fitted new TF-IDF vectorizer: {len(vectorizer.vocabulary_)} vocab terms")
    else:
        matrix = vectorizer.transform(texts)
        logger.info("Applied existing TF-IDF vectorizer to new data")

    return matrix, vectorizer


def save_tfidf_vectorizer(vectorizer: TfidfVectorizer, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved TF-IDF vectorizer to {filepath}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))
    from ingestion import ingest_raw_data
    from cleaning import clean_and_label

    df = ingest_raw_data("data/raw/Reviews.csv", text_col="Text", rating_col="Score")
    cleaned = clean_and_label(df)

    tfidf_matrix, vectorizer = build_tfidf_features(cleaned["text_heavy_clean"])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

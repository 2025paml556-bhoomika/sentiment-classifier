"""
Cleaning & Labeling Module
----------------------------
Produces TWO cleaned versions of the text, because the two models
in this project expect different inputs:

  - heavy_clean  -> for TF-IDF (Logistic Regression)   : stopwords removed
  - light_clean  -> for DistilBERT tokenizer            : stopwords KEPT
                                                           (transformers need
                                                           full sentence context)

Also converts star ratings into a binary sentiment label and drops
neutral (3-star) rows, per the 2-class design decision.
"""

import re
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "on", "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "its",
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _basic_clean(text: str) -> str:
    text = str(text).lower()
    text = _strip_html(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def heavy_clean(text: str) -> str:
    """Aggressive cleaning for TF-IDF: lowercase, strip punctuation, remove stopwords."""
    text = _basic_clean(text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)


def light_clean(text: str) -> str:
    """Light cleaning for DistilBERT: keep sentence structure and stopwords."""
    text = str(text)
    text = _strip_html(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Map star rating -> Positive/Negative, drop neutral (3-star) rows."""
    df = df.copy()
    df = df[df["rating"] != 3]  # drop neutral
    df["label"] = (df["rating"] >= 4).astype(int)  # 1 = Positive, 0 = Negative
    return df


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning step: drop nulls/empties/duplicates, create labels,
    and produce both cleaned text versions.
    """
    df = df.copy()
    before = len(df)

    df = df.dropna(subset=["review_text", "rating"])
    df = df[df["review_text"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["review_text"])

    df = label_sentiment(df)

    df["text_heavy_clean"] = df["review_text"].apply(heavy_clean)
    df["text_light_clean"] = df["review_text"].apply(light_clean)

    df = df[df["text_heavy_clean"].str.strip() != ""]

    after = len(df)
    logger.info(f"Cleaning complete: {before} -> {after} rows "
                f"({before - after} dropped: nulls/empty/duplicate/neutral)")

    return df[["review_text", "rating", "label", "text_heavy_clean", "text_light_clean"]]


if __name__ == "__main__":
    from ingestion import ingest_raw_data

    df = ingest_raw_data("data/raw/Reviews.csv", text_col="Text", rating_col="Score")
    cleaned = clean_and_label(df)
    print(cleaned.head())

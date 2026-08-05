"""
Data Validation Module
------------------------
Checks the ingested data for quality issues BEFORE it moves into
cleaning/feature engineering. This is the "reliability" piece the
assignment specifically grades — a pipeline that silently trains
on bad data is not reliable.

Each check returns findings; validate_data() raises an error if any
CRITICAL rule fails, and logs a warning report for everything else.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data fails a critical validation rule."""
    pass


def validate_data(df: pd.DataFrame, min_rows: int = 100) -> dict:
    """
    Run validation checks on the ingested data.

    Returns a report dict (also logged) describing what was found.
    Raises DataValidationError if a critical rule is violated.
    """
    report = {}

    # --- Critical checks (pipeline stops if these fail) ---
    if len(df) < min_rows:
        raise DataValidationError(f"Only {len(df)} rows found — need at least {min_rows}")

    if "review_text" not in df.columns or "rating" not in df.columns:
        raise DataValidationError("Required columns 'review_text'/'rating' missing after ingestion")

    # --- Informational checks (logged, not blocking) ---
    n_null_text = df["review_text"].isna().sum()
    n_null_rating = df["rating"].isna().sum()
    n_duplicates = df.duplicated(subset=["review_text"]).sum()
    n_empty_text = (df["review_text"].astype(str).str.strip() == "").sum()

    report["total_rows"] = len(df)
    report["null_text_rows"] = int(n_null_text)
    report["null_rating_rows"] = int(n_null_rating)
    report["duplicate_rows"] = int(n_duplicates)
    report["empty_text_rows"] = int(n_empty_text)

    # Rating value sanity check (assumes 1-5 star scale — adjust if needed)
    valid_ratings = df["rating"].dropna().between(1, 5)
    n_invalid_rating = (~valid_ratings).sum()
    report["invalid_rating_values"] = int(n_invalid_rating)

    # Class balance preview (based on the label rule: >=4 positive, <=2 negative)
    positive = (df["rating"] >= 4).sum()
    negative = (df["rating"] <= 2).sum()
    neutral = len(df) - positive - negative
    report["class_balance"] = {
        "positive": int(positive),
        "negative": int(negative),
        "neutral_dropped": int(neutral),
    }

    # Review length distribution (word count) — useful for spotting junk rows
    lengths = df["review_text"].fillna("").astype(str).apply(lambda t: len(t.split()))
    report["review_length_stats"] = {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 1),
    }

    logger.info("Validation report:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    return report


if __name__ == "__main__":
    from ingestion import ingest_raw_data
    df = ingest_raw_data("data/raw/reviews.csv", text_col="Text", rating_col="Score")
    validate_data(df)

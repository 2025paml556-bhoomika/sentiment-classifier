"""
Data Ingestion Module
----------------------
Loads raw review data from a CSV file into a pandas DataFrame.
Kept separate from cleaning/validation so it can be swapped out
(e.g. for a database or API source) without touching downstream code.
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ingest_raw_data(filepath: str, text_col: str, rating_col: str) -> pd.DataFrame:
    """
    Load raw CSV data and return a DataFrame with just the columns we need.

    Args:
        filepath: path to the raw CSV file
        text_col: name of the column containing review text
        rating_col: name of the column containing the star rating / score

    Returns:
        DataFrame with columns [text_col, rating_col]

    Raises:
        FileNotFoundError: if the file doesn't exist
        ValueError: if expected columns are missing
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    logger.info(f"Ingesting raw data from {filepath}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    missing = [c for c in (text_col, rating_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Expected column(s) not found in data: {missing}. "
                          f"Available columns: {list(df.columns)}")

    df = df[[text_col, rating_col]].copy()
    df.columns = ["review_text", "rating"]

    logger.info(f"Ingestion complete: {len(df)} rows ready for validation")
    return df


if __name__ == "__main__":
    # Quick manual test
    df = ingest_raw_data("data/raw/reviews.csv", text_col="Text", rating_col="Score")
    print(df.head())

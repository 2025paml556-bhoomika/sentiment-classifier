"""
Train/test split builder
--------------------------
Writes data/processed/split.parquet with two columns: review_id and split.
This file is the single source of truth for row assignment. Both model
paths (tfidf/ and distilbert/) read it, so every run is scored on the
same 72,765 test rows.

The split used to live inside feature_store/features.parquet, which made
the DistilBERT path depend on a TF-IDF artifact. This script moves that
ownership into the shared data stage.

Row assignment must stay identical to the original, or old reported
metrics lose comparability. So:
  - if feature_store/features.parquet exists locally, the split is copied
    from it, byte for byte
  - otherwise it is recomputed with the exact logic the original
    build_features.py used (same columns, same dropna, same seed)

Run from the repo root, then version the output:
    ./venv/bin/python data_pipeline/make_split.py
    dvc add data/processed/split.parquet
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STORE_PARQUET = Path("feature_store/features.parquet")
CLEANED_CSV = Path("data/processed/cleaned_reviews.csv")
OUT_PATH = Path("data/processed/split.parquet")
TEST_SIZE = 0.2
SEED = 42
EXPECTED = {"train": 291_060, "test": 72_765}


def main():
    if STORE_PARQUET.exists():
        logger.info(f"Copying existing split from {STORE_PARQUET}")
        split = pd.read_parquet(STORE_PARQUET, columns=["review_id", "split"])
    else:
        from sklearn.model_selection import train_test_split

        logger.info(f"Store not present, recomputing split from {CLEANED_CSV} "
                    "with the original build_features.py logic")
        df = pd.read_csv(CLEANED_CSV,
                         usecols=["review_text", "text_heavy_clean", "label"]).dropna()
        df = df.reset_index(drop=True)
        _, test_idx = train_test_split(df.index, test_size=TEST_SIZE,
                                       stratify=df["label"], random_state=SEED)
        split = pd.DataFrame({"review_id": df.index, "split": "train"})
        split.loc[test_idx, "split"] = "test"

    counts = split["split"].value_counts().to_dict()
    if counts != EXPECTED:
        raise SystemExit(f"Split counts {counts} do not match the published "
                         f"{EXPECTED}. Not writing anything.")

    split.to_parquet(OUT_PATH, index=False)
    logger.info(f"Wrote {len(split):,} rows to {OUT_PATH} ({counts})")


if __name__ == "__main__":
    main()

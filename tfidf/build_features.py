"""
Feature store builder
-----------------------
Computes features ONCE and writes them to a Parquet-backed feature store.
Training reads from the store and never recomputes. Serving reuses the
saved transformer, so training and serving cannot disagree.

    cleaned_reviews.csv
        -> TF-IDF (20k, unigram+bigram, sublinear) -> TruncatedSVD (600 dims)
        -> feature_store/features.parquet   (fixed 600-column schema)
        -> feature_store/transformer.pkl    (fitted, for serving)
        -> feature_store/schema.json        (what the columns are)

Why SVD instead of picking the top N TF-IDF terms: SVD compresses all
20,000 features into combinations rather than discarding most of them.
Config chosen by one-knob ablation (see MLflow logreg-store-* runs):
going 300 -> 600 dims was worth +0.0255 macro F1 and sublinear_tf
another +0.0078. Compression still costs accuracy against inline
TF-IDF; that is the price of the store's training-serving consistency.

Why the transformer is fitted on the TRAIN split only: fitting it on all
rows would leak test-set information into the features. The store keeps
every row, with a `split` column saying which is which.

The split itself is NOT created here. It comes from the shared
data/processed/split.parquet (see data_pipeline/make_split.py), so both
model paths score on identical rows. The store keeps a copy of the
split column for convenience.

Run from the repo root:
    ./venv/bin/python tfidf/build_features.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
SPLIT_PARQUET = "data/processed/split.parquet"
STORE_DIR = Path("feature_store")
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
SVD_DIMS = 600
SEED = 42


def build_transformer():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
                                  sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=SVD_DIMS, random_state=SEED)),
    ])


def main():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEANED_CSV, usecols=["review_text", "text_heavy_clean", "label"]).dropna()
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} cleaned rows")

    split = pd.read_parquet(SPLIT_PARQUET).sort_values("review_id")
    if len(split) != len(df) or not (split["review_id"].values == df.index.values).all():
        raise SystemExit(f"{SPLIT_PARQUET} does not line up with {CLEANED_CSV}. "
                         "Re-run data_pipeline/make_split.py.")
    train_idx = split.loc[split["split"] == "train", "review_id"]

    transformer = build_transformer()
    logger.info(f"Fitting transformer on {len(train_idx):,} train rows "
                f"({MAX_FEATURES} tfidf features -> {SVD_DIMS} SVD dims)")
    transformer.fit(df.loc[train_idx, "text_heavy_clean"])

    logger.info(f"Transforming all {len(df):,} rows into the store")
    features = transformer.transform(df["text_heavy_clean"]).astype("float32")

    feature_cols = [f"f_{i}" for i in range(SVD_DIMS)]
    store = pd.DataFrame(features, columns=feature_cols)
    store.insert(0, "review_id", df.index)
    store.insert(1, "label", df["label"].values)
    store.insert(2, "text_length", df["review_text"].str.split().str.len().values)
    store.insert(3, "split", split["split"].values)

    store_path = STORE_DIR / "features.parquet"
    store.to_parquet(store_path, index=False, compression="snappy")
    logger.info(f"Wrote {len(store):,} rows x {len(store.columns)} cols to {store_path} "
                f"({store_path.stat().st_size / 1e6:.0f} MB)")

    pd.to_pickle(transformer, STORE_DIR / "transformer.pkl")
    logger.info(f"Saved fitted transformer to {STORE_DIR / 'transformer.pkl'}")

    schema = {
        "feature_columns": feature_cols,
        "n_features": SVD_DIMS,
        "metadata_columns": ["review_id", "label", "text_length", "split"],
        "tfidf_max_features": MAX_FEATURES,
        "tfidf_ngram_range": list(NGRAM_RANGE),
        "tfidf_sublinear_tf": True,
        "svd_dims": SVD_DIMS,
        "fitted_on": "train split only, to avoid test-set leakage",
        "split_source": SPLIT_PARQUET,
        "random_state": SEED,
        "rows": split["split"].value_counts().astype(int).to_dict(),
    }
    with open(STORE_DIR / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Schema written to {STORE_DIR / 'schema.json'}")


if __name__ == "__main__":
    main()

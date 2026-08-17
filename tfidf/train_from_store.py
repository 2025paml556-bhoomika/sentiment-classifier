"""
Train from the feature store
------------------------------
Reads features from feature_store/features.parquet and never recomputes
them. The store's `split` column is a copy of the shared
data/processed/split.parquet, so this run is scored on the same test
rows as every other model.

Saves two things to MLflow:
  - the classifier, which expects the store's feature columns
  - a serving pipeline (transformer + classifier) that maps raw cleaned
    text straight to a prediction, so serving cannot use different
    features from training

Run from the repo root:
    ./venv/bin/python tfidf/train_from_store.py
"""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STORE_DIR = Path("feature_store")
EXPERIMENT = "sentiment-classifier"
SEED = 42

# None trains on the full train split. A number draws a stratified sample of
# that size, same seed and ratios as comparison/train_controlled_50k.py, so
# runs stay comparable.
TRAIN_SAMPLE = None

# The store config this name reflects: TF-IDF 20k sublinear -> SVD 600.
# The SVD-300 era runs (logreg-feature-store-svd300, -50k) stay in MLflow.
RUN_NAME = ("logreg-feature-store-svd600" if TRAIN_SAMPLE is None
            else f"logreg-feature-store-svd600-{TRAIN_SAMPLE // 1000}k")


def main():
    schema = json.loads((STORE_DIR / "schema.json").read_text())
    feature_cols = schema["feature_columns"]

    store = pd.read_parquet(STORE_DIR / "features.parquet")
    train = store[store["split"] == "train"]
    test = store[store["split"] == "test"]
    logger.info(f"Read store: {len(train):,} train / {len(test):,} test, "
                f"{len(feature_cols)} features")

    if TRAIN_SAMPLE is not None:
        pool = len(train)
        train = train.groupby("label", group_keys=False).apply(
            lambda g: g.sample(int(round(TRAIN_SAMPLE * len(g) / pool)),
                               random_state=SEED))
        logger.info(f"Sampled {len(train):,} of {pool:,} train rows, "
                    f"labels {train['label'].value_counts().to_dict()}")

    # As numpy, not DataFrames: the serving pipeline feeds arrays straight
    # out of SVD, so fitting on named columns would warn on every request.
    Xtr, ytr = train[feature_cols].to_numpy(), train["label"]
    Xte, yte = test[feature_cols].to_numpy(), test["label"]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
        model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=SEED)
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        metrics = {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "roc_auc": roc_auc_score(yte, prob),
            "negative_precision": precision_score(yte, pred, pos_label=0),
            "negative_recall": recall_score(yte, pred, pos_label=0),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "features": "feature_store/features.parquet",
            "n_features": schema["n_features"],
            "tfidf_max_features": schema["tfidf_max_features"],
            "tfidf_ngram_range": str(tuple(schema["tfidf_ngram_range"])),
            "svd_dims": schema["svd_dims"],
            "class_weight": "balanced",
            "random_state": SEED,
            "train_rows": len(train),
            "test_rows": len(test),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(yte, pred, target_names=["negative", "positive"]),
            "classification_report.txt")
        mlflow.log_artifact(str(STORE_DIR / "schema.json"))
        mlflow.sklearn.log_model(model, name="model")

        # The transformer that built the store, plus this classifier, as one
        # artifact. Serving loads this and cannot compute different features.
        transformer = pd.read_pickle(STORE_DIR / "transformer.pkl")
        serving = Pipeline(transformer.steps + [("clf", model)])
        mlflow.sklearn.log_model(serving, name="serving_pipeline")

        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")

    logger.info("compare all runs with: ./venv/bin/mlflow ui")


if __name__ == "__main__":
    main()

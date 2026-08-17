"""
Feature store ablation
------------------------
The store model (logreg-feature-store-svd300) trails the plain bigram
baseline by 0.085 macro F1. This finds out which store parameter is to
blame: every config changes exactly ONE thing against the store's
setup (TF-IDF 20k bigrams -> SVD 300 -> LogReg C=1).

Runs in memory, without rewriting the 581 MB store. Same train/test
rows as every other run (data/processed/split.parquet). Rebuild the
store with the winning config afterwards via tfidf/build_features.py.

Run from the repo root:
    ./venv/bin/python tfidf/store_ablation.py
"""

import logging
import time

import mlflow
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
SPLIT_PARQUET = "data/processed/split.parquet"
EXPERIMENT = "sentiment-classifier"
SEED = 42

# Store baseline: tfidf {max_features 20000, ngram (1,2)}, svd_dims 300, C 1.0.
# The one-knob runs (svd600 +0.0255, sublinear +0.0078, tfidf50k +0.0020,
# C4 -0.0001) are logged in MLflow. This combines the two winners.
CONFIGS = [
    {"name": "logreg-store-svd600-sublinear", "svd_dims": 600,
     "tfidf": {"max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True}},
]


def load_split():
    # Same columns and dropna as make_split.py, so the index lines up
    # with the split file's review_id.
    df = pd.read_csv(CLEANED_CSV,
                     usecols=["review_text", "text_heavy_clean", "label"]).dropna()
    df = df.reset_index(drop=True)
    split = pd.read_parquet(SPLIT_PARQUET)
    train = df.loc[split.loc[split["split"] == "train", "review_id"]]
    test = df.loc[split.loc[split["split"] == "test", "review_id"]]
    return (train["text_heavy_clean"], test["text_heavy_clean"],
            train["label"], test["label"])


def train_one(cfg, Xtr_text, Xte_text, ytr, yte):
    t0 = time.time()
    with mlflow.start_run(run_name=cfg["name"]):
        transformer = Pipeline([
            ("tfidf", TfidfVectorizer(**cfg["tfidf"])),
            ("svd", TruncatedSVD(n_components=cfg["svd_dims"], random_state=SEED)),
        ])
        Xtr = transformer.fit_transform(Xtr_text)
        Xte = transformer.transform(Xte_text)
        logger.info(f"{cfg['name']}: transformed to {Xtr.shape[1]} dims "
                    f"in {time.time() - t0:.0f}s")

        model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                   C=cfg.get("C", 1.0), random_state=SEED)
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
            "features": "tfidf+svd, in-memory store ablation",
            **{k: str(v) for k, v in cfg["tfidf"].items()},
            "svd_dims": cfg["svd_dims"],
            "C": cfg.get("C", 1.0),
            "class_weight": "balanced",
            "split_source": SPLIT_PARQUET,
            "random_state": SEED,
            "train_rows": len(ytr),
            "test_rows": len(yte),
        })
        mlflow.log_metrics(metrics)
        logger.info(f"{cfg['name']}: done in {time.time() - t0:.0f}s")
        return metrics


def main():
    mlflow.set_experiment(EXPERIMENT)
    Xtr_text, Xte_text, ytr, yte = load_split()
    print(f"\ntrain {len(ytr):,} rows   test {len(yte):,} rows\n")
    print(f"{'run':<26} {'accuracy':>9} {'macro F1':>9} {'ROC AUC':>8} {'neg recall':>11}")
    print("-" * 66)

    for cfg in CONFIGS:
        m = train_one(cfg, Xtr_text, Xte_text, ytr, yte)
        print(f"{cfg['name']:<26} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
              f"{m['roc_auc']:>8.4f} {m['negative_recall']:>11.4f}")

    print("\nstore baseline logreg-feature-store-svd300: macro F1 0.7878")
    print("browse runs with: ./venv/bin/mlflow ui\n")


if __name__ == "__main__":
    main()

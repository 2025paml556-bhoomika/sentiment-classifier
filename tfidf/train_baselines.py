"""
Week 2 (M3) — Logistic Regression baseline on TF-IDF features
---------------------------------------------------------------
Trains one MLflow run per feature configuration and logs parameters,
metrics, the classification report, and the fitted model.

Model and vectorizer are wrapped in a single sklearn Pipeline, so the
two can never drift apart, and Week 3 serving loads one artifact.

The train/test split comes from the shared data/processed/split.parquet,
so these runs score on the same test rows as every other model.

Run from the repo root:
    ./venv/bin/python tfidf/train_baselines.py

Then browse the runs:
    ./venv/bin/mlflow ui

Note: the pipeline expects text that heavy_clean() has already
processed. It does not clean text itself.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
SPLIT_PARQUET = "data/processed/split.parquet"
EXPERIMENT = "sentiment-classifier"
SEED = 42

# One MLflow run per entry.
CONFIGS = [
    {"name": "logreg-tfidf-unigram", "max_features": 5000, "ngram_range": (1, 1)},
    {"name": "logreg-tfidf-bigram", "max_features": 20000, "ngram_range": (1, 2)},
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


def train_one(cfg, Xtr, Xte, ytr, yte):
    with mlflow.start_run(run_name=cfg["name"]):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=cfg["max_features"],
                                      ngram_range=cfg["ngram_range"])),
            ("clf", LogisticRegression(class_weight="balanced", solver="liblinear")),
        ])
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]

        metrics = {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "roc_auc": roc_auc_score(yte, prob),
            # Negative is the minority class at 15.7%, so these two
            # matter more than accuracy for judging the model.
            "negative_precision": precision_score(yte, pred, pos_label=0),
            "negative_recall": recall_score(yte, pred, pos_label=0),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "features": "tfidf",
            "max_features": cfg["max_features"],
            "ngram_range": str(cfg["ngram_range"]),
            "class_weight": "balanced",
            "solver": "liblinear",
            "split_source": SPLIT_PARQUET,
            "random_state": SEED,
            "train_rows": len(Xtr),
            "test_rows": len(Xte),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(yte, pred, target_names=["negative", "positive"]),
            "classification_report.txt")
        mlflow.sklearn.log_model(model, name="model")

        return metrics


def main():
    mlflow.set_experiment(EXPERIMENT)
    Xtr, Xte, ytr, yte = load_split()
    print(f"\ntrain {len(Xtr):,} rows   test {len(Xte):,} rows\n")
    print(f"{'run':<22} {'accuracy':>9} {'macro F1':>9} {'ROC AUC':>8} {'neg recall':>11}")
    print("-" * 62)

    results = {}
    for cfg in CONFIGS:
        m = train_one(cfg, Xtr, Xte, ytr, yte)
        results[cfg["name"]] = m
        print(f"{cfg['name']:<22} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
              f"{m['roc_auc']:>8.4f} {m['negative_recall']:>11.4f}")

    best = max(results, key=lambda name: results[name]["macro_f1"])
    print(f"\nbest by macro F1: {best} ({results[best]['macro_f1']:.4f})")
    print("browse runs with: ./venv/bin/mlflow ui\n")


if __name__ == "__main__":
    main()

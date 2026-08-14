"""Controlled comparison: LogReg and DistilBERT on the SAME 50,000 rows.

The existing runs are confounded. distilbert-20k trained on 20,000 rows and
the baselines on 291,060, so the comparison mixes architecture with training
size. Here both models see one identical 50,000-row sample, drawn from the
feature store's train split and evaluated on its 72,765 test rows.

Two Logistic Regression variants run, with and without class weighting,
because DistilBERT uses plain cross-entropy. That isolates weighting from
architecture.

Text differs by design: DistilBERT reads text_light_clean (keeps sentence
structure), LogReg reads text_heavy_clean (stopwords removed).

Run from the repo root:
    ./venv/bin/python training/train_controlled_50k.py
"""

import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_distilbert import ReviewDataset, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
STORE_PARQUET = "feature_store/features.parquet"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "sentiment-classifier"
OUT_DIR = Path("model_store/distilbert-50k")

TRAIN_SAMPLE = 50_000
SEED = 42

# Matches the winning baseline config, so the only change is training size.
TFIDF_FEATURES = 20000
TFIDF_NGRAMS = (1, 2)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH = 16
EVAL_BATCH = 64
EPOCHS = 1
LR = 2e-5
DEVICE = "mps"


def scores(ytrue, pred, prob):
    return {
        "accuracy": accuracy_score(ytrue, pred),
        "macro_f1": f1_score(ytrue, pred, average="macro"),
        "roc_auc": roc_auc_score(ytrue, prob),
        "negative_precision": precision_score(ytrue, pred, pos_label=0),
        "negative_recall": recall_score(ytrue, pred, pos_label=0),
    }


def load_data():
    """One stratified sample, shared by both models, from the store's train split."""
    df = pd.read_csv(CLEANED_CSV,
                     usecols=["review_text", "text_heavy_clean",
                              "text_light_clean", "label"])
    df = df.dropna(subset=["review_text", "text_heavy_clean", "label"])
    df = df.reset_index(drop=True)

    split = pd.read_parquet(STORE_PARQUET, columns=["review_id", "split"])
    test = df.loc[split.loc[split["split"] == "test", "review_id"]]
    pool = df.loc[split.loc[split["split"] == "train", "review_id"]]

    sample = pool.groupby("label", group_keys=False).apply(
        lambda g: g.sample(int(round(TRAIN_SAMPLE * len(g) / len(pool))),
                           random_state=SEED))
    logger.info(f"train {len(sample):,} of {len(pool):,} available   test {len(test):,}")
    logger.info(f"train labels: {sample['label'].value_counts().to_dict()}")
    return sample, test


def run_logreg(name, class_weight, train_df, test_df):
    t0 = time.time()
    with mlflow.start_run(run_name=name):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=TFIDF_FEATURES,
                                      ngram_range=TFIDF_NGRAMS)),
            ("clf", LogisticRegression(class_weight=class_weight,
                                       solver="liblinear")),
        ])
        model.fit(train_df["text_heavy_clean"], train_df["label"])
        prob = model.predict_proba(test_df["text_heavy_clean"])[:, 1]
        pred = (prob > 0.5).astype(int)
        metrics = scores(test_df["label"], pred, prob)

        mlflow.log_params({
            "model": "LogisticRegression",
            "features": "tfidf",
            "max_features": TFIDF_FEATURES,
            "ngram_range": str(TFIDF_NGRAMS),
            "class_weight": str(class_weight),
            "solver": "liblinear",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split_source": STORE_PARQUET,
            "random_state": SEED,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(test_df["label"], pred,
                                  target_names=["negative", "positive"]),
            "classification_report.txt")
        mlflow.sklearn.log_model(model, name="model")

    logger.info(f"{name} done in {time.time() - t0:.0f}s")
    return metrics


def run_distilbert(name, train_df, test_df):
    t0 = time.time()
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = ReviewDataset(train_df["text_light_clean"], train_df["label"], tokenizer)
    test_ds = ReviewDataset(test_df["text_light_clean"], test_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              collate_fn=train_ds.collate)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, collate_fn=test_ds.collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    with mlflow.start_run(run_name=name):
        model.train()
        for epoch in range(EPOCHS):
            running = 0.0
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running += float(loss)
                if step % 100 == 0:
                    logger.info(f"step {step}/{len(train_loader)} "
                                f"loss {running / 100:.4f} "
                                f"({(time.time() - t0) / step:.2f}s/step)")
                    mlflow.log_metric("train_loss", running / 100,
                                      step=(epoch * len(train_loader)) + step)
                    running = 0.0

        logger.info("evaluating on the store's test split")
        ytrue, prob = evaluate(model, test_loader)
        pred = (prob > 0.5).astype(int)
        metrics = scores(ytrue, pred, prob)

        mlflow.log_params({
            "model": MODEL_NAME,
            "features": "text_light_clean, tokenized",
            "max_length": MAX_LENGTH,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "epochs": EPOCHS,
            "batch_size": TRAIN_BATCH,
            "learning_rate": LR,
            "class_weight": "none",
            "device": DEVICE,
            "split_source": STORE_PARQUET,
            "random_state": SEED,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(ytrue, pred, target_names=["negative", "positive"]),
            "classification_report.txt")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        logger.info(f"saved model and tokenizer to {OUT_DIR}")

    logger.info(f"{name} done in {(time.time() - t0) / 60:.1f}min")
    return metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    train_df, test_df = load_data()

    results = {
        "logreg-50k-balanced": run_logreg("logreg-50k-balanced", "balanced",
                                          train_df, test_df),
        "logreg-50k-unweighted": run_logreg("logreg-50k-unweighted", None,
                                            train_df, test_df),
        "distilbert-50k": run_distilbert("distilbert-50k", train_df, test_df),
    }

    print(f"\nall models trained on the same {len(train_df):,} rows, "
          f"tested on {len(test_df):,}\n")
    print(f"{'run':<24}{'acc':>8}{'macroF1':>9}{'rocAUC':>8}{'negP':>7}{'negR':>7}")
    print("-" * 63)
    for name, m in results.items():
        print(f"{name:<24}{m['accuracy']:>8.4f}{m['macro_f1']:>9.4f}"
              f"{m['roc_auc']:>8.4f}{m['negative_precision']:>7.3f}"
              f"{m['negative_recall']:>7.3f}")


if __name__ == "__main__":
    main()

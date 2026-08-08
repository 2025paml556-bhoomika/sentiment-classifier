"""
Week 2 (M3) — DistilBERT fine-tuned on light-cleaned text
-----------------------------------------------------------
The advanced model for task 2.2, trained on the Apple GPU via MPS.

Scored on the SAME test rows as every other run, taken from the feature
store's `split` column, so task 2.4 compares models rather than splits.

Training uses a stratified subsample, because 8 GB of shared memory makes
the full 291k rows impractical. The sample size is logged to MLflow.

Run from the repo root:
    ./venv/bin/python training/train_distilbert.py
"""

import logging
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
STORE_PARQUET = "feature_store/features.parquet"
OUT_DIR = Path("model_store/distilbert-20k")
EXPERIMENT = "sentiment-classifier"
RUN_NAME = "distilbert-20k"

MODEL_NAME = "distilbert-base-uncased"
TRAIN_SAMPLE = 20_000
MAX_LENGTH = 128
TRAIN_BATCH = 16
EVAL_BATCH = 64
EPOCHS = 1
LR = 2e-5
SEED = 42
DEVICE = "mps"


class ReviewDataset(Dataset):
    """Tokenizes one row at a time, so no big tensor is held in memory."""

    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

    def collate(self, batch):
        texts, labels = zip(*batch)
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=MAX_LENGTH, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc


def load_data():
    """Load rows and reuse the feature store's train/test split."""
    df = pd.read_csv(CLEANED_CSV,
                     usecols=["review_text", "text_heavy_clean",
                              "text_light_clean", "label"])
    # build_features.py read those first three columns then dropped NaNs, so
    # this index matches the store's review_id exactly.
    df = df.dropna(subset=["review_text", "text_heavy_clean", "label"])
    df = df.reset_index(drop=True)

    split = pd.read_parquet(STORE_PARQUET, columns=["review_id", "split"])
    test_ids = split.loc[split["split"] == "test", "review_id"]
    train_ids = split.loc[split["split"] == "train", "review_id"]

    test = df.loc[test_ids]
    pool = df.loc[train_ids]

    # Stratified subsample: keep the real class ratio rather than balancing it,
    # since class_weight in the other models handles imbalance at training time.
    sample = pool.groupby("label", group_keys=False).apply(
        lambda g: g.sample(int(round(TRAIN_SAMPLE * len(g) / len(pool))),
                           random_state=SEED))
    logger.info(f"train {len(sample):,} of {len(pool):,} available   "
                f"test {len(test):,}")
    logger.info(f"train labels: {sample['label'].value_counts().to_dict()}")
    return sample, test


def evaluate(model, loader):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            trues.append(labels.numpy())
            if i % 200 == 0:
                logger.info(f"  eval batch {i}/{len(loader)}")
    return np.concatenate(trues), np.concatenate(probs)


def main():
    torch.manual_seed(SEED)
    train_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = ReviewDataset(train_df["text_light_clean"], train_df["label"], tokenizer)
    test_ds = ReviewDataset(test_df["text_light_clean"], test_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              collate_fn=train_ds.collate)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH,
                             collate_fn=test_ds.collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
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
                    logger.info(f"epoch {epoch + 1} step {step}/{len(train_loader)} "
                                f"loss {running / 100:.4f}")
                    mlflow.log_metric("train_loss", running / 100,
                                      step=(epoch * len(train_loader)) + step)
                    running = 0.0

        logger.info("evaluating on the store's test split")
        ytrue, prob = evaluate(model, test_loader)
        pred = (prob > 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(ytrue, pred),
            "macro_f1": f1_score(ytrue, pred, average="macro"),
            "roc_auc": roc_auc_score(ytrue, prob),
            "negative_precision": precision_score(ytrue, pred, pos_label=0),
            "negative_recall": recall_score(ytrue, pred, pos_label=0),
        }

        mlflow.log_params({
            "model": MODEL_NAME,
            "features": "text_light_clean, tokenized",
            "max_length": MAX_LENGTH,
            "train_sample": len(train_df),
            "train_pool": 291060,
            "test_rows": len(test_df),
            "epochs": EPOCHS,
            "batch_size": TRAIN_BATCH,
            "learning_rate": LR,
            "device": DEVICE,
            "split_source": "feature_store/features.parquet",
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

        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")

    logger.info("compare all runs with: ./venv/bin/mlflow ui "
                "--backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()

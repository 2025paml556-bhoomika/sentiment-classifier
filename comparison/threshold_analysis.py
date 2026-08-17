"""Threshold analysis: does distilbert-50k beat logreg-tfidf-bigram?

The model comparison scores every run at a 0.5 decision threshold. That is
an arbitrary default, not a property of a model. A support-ticket classifier
should pick its threshold from the recall the business needs.

This compares the two candidates across the whole precision-recall curve,
on the shared split's 72,765 test rows, and reports precision at matched
recall. Probabilities are cached, so the sweep can be re-run cheaply.

Run from the repo root:
    ./venv/bin/python comparison/threshold_analysis.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "distilbert"))
from train import ReviewDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/processed/cleaned_reviews.csv"
SPLIT_PARQUET = "data/processed/split.parquet"
BERT_DIR = Path("model_store/distilbert-50k")
CACHE = Path("/tmp/threshold_cache")

SEED = 42
DEVICE = "mps"
EVAL_BATCH = 64
MAX_LENGTH = 128

# Recall levels to compare at. 0.9094 is logreg-tfidf-bigram's recall at 0.5.
TARGET_RECALLS = [0.80, 0.85, 0.90, 0.9094, 0.93, 0.95]


def load_data():
    df = pd.read_csv(CLEANED_CSV,
                     usecols=["review_text", "text_heavy_clean",
                              "text_light_clean", "label"])
    df = df.dropna(subset=["review_text", "text_heavy_clean", "label"])
    df = df.reset_index(drop=True)

    split = pd.read_parquet(SPLIT_PARQUET)
    test = df.loc[split.loc[split["split"] == "test", "review_id"]]
    train = df.loc[split.loc[split["split"] == "train", "review_id"]]
    logger.info(f"train {len(train):,}   test {len(test):,}")
    return train, test


def logreg_probs(train, test):
    """P(negative) from a retrained logreg-tfidf-bigram."""
    path = CACHE / "logreg.npy"
    if path.exists():
        logger.info("using cached logreg probabilities")
        return np.load(path)

    logger.info("training logreg-tfidf-bigram on 291,060 rows")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear")),
    ])
    model.fit(train["text_heavy_clean"], train["label"])
    # Column 0 is the negative class, which is the one we care about.
    probs = model.predict_proba(test["text_heavy_clean"])[:, 0]
    np.save(path, probs)
    return probs


def bert_probs(test):
    """P(negative) from the saved distilbert-50k weights."""
    path = CACHE / "bert.npy"
    if path.exists():
        logger.info("using cached distilbert probabilities")
        return np.load(path)

    logger.info(f"running distilbert-50k inference on {len(test):,} rows")
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE)
    model.eval()

    ds = ReviewDataset(test["text_light_clean"], test["label"], tokenizer)
    loader = DataLoader(ds, batch_size=EVAL_BATCH, collate_fn=ds.collate)

    out = []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            out.append(torch.softmax(logits, dim=-1)[:, 0].cpu().numpy())
            if i % 200 == 0:
                logger.info(f"  batch {i}/{len(loader)}")
    probs = np.concatenate(out)
    np.save(path, probs)
    return probs


def curve(y_neg, probs):
    """Precision and recall for the negative class at every threshold."""
    order = np.argsort(-probs)
    hits = y_neg[order].cumsum()
    flagged = np.arange(1, len(probs) + 1)
    return hits / flagged, hits / y_neg.sum(), probs[order]


def at_recall(precision, recall, thresholds, target):
    i = np.searchsorted(recall, target)
    if i >= len(recall):
        return None
    return precision[i], recall[i], thresholds[i]


def main():
    CACHE.mkdir(exist_ok=True)
    train, test = load_data()
    # Negative is the minority class, so treat label 0 as the positive event.
    y_neg = (test["label"].values == 0).astype(int)
    n_neg = int(y_neg.sum())

    models = {
        "logreg-tfidf-bigram": logreg_probs(train, test),
        "distilbert-50k": bert_probs(test),
    }

    print(f"\ntest rows {len(test):,}   real negatives {n_neg:,}\n")
    print("At the default 0.5 threshold")
    print(f"{'model':<22}{'prec':>8}{'recall':>8}{'caught':>9}{'missed':>8}{'falseAl':>9}")
    print("-" * 64)
    for name, probs in models.items():
        pred = probs > 0.5
        caught = int((pred & (y_neg == 1)).sum())
        flagged = int(pred.sum())
        print(f"{name:<22}{caught / flagged:>8.3f}{caught / n_neg:>8.3f}"
              f"{caught:>9,}{n_neg - caught:>8,}{flagged - caught:>9,}")

    print("\nPrecision at matched recall")
    header = f"{'recall':>8}" + "".join(f"{n:>24}" for n in models)
    print(header)
    print("-" * len(header))
    curves = {n: curve(y_neg, p) for n, p in models.items()}
    for target in TARGET_RECALLS:
        row = f"{target:>8.3f}"
        for name in models:
            p, r, t = curves[name]
            got = at_recall(p, r, t, target)
            row += "" if got is None else f"{got[0]:>15.3f} @ t={got[2]:.3f}"
        print(row)

    print("\nFalse alarms needed to reach each recall")
    header = f"{'recall':>8}" + "".join(f"{n:>22}" for n in models)
    print(header)
    print("-" * len(header))
    for target in TARGET_RECALLS:
        row = f"{target:>8.3f}"
        for name in models:
            p, r, t = curves[name]
            got = at_recall(p, r, t, target)
            if got is None:
                row += f"{'-':>22}"
            else:
                caught = got[1] * n_neg
                row += f"{caught / got[0] - caught:>22,.0f}"
        print(row)


if __name__ == "__main__":
    main()

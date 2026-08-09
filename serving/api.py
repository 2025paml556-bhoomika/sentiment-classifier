"""
Week 3 (M4) — prediction API
------------------------------
Serves the fine-tuned DistilBERT model, which won on macro F1.

Runs on CPU, not MPS, because the Docker image in task 3.4 has no Apple
GPU. Serving on CPU everywhere keeps local and container behaviour the same.

Input text goes through light_clean() before tokenizing, the same function
used in training. Skipping it would feed the model differently shaped text
than it learned from.

Weights live in model_store/, which is DVC-tracked, so run `dvc pull` first
on a fresh clone.

Start it from the repo root:
    ./venv/bin/uvicorn serving.api:app --reload

Then try it:
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"text": "arrived stale and tasted awful"}'
"""

import re
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "features"))

from cleaning import light_clean  # noqa: E402

MODEL_DIR = ROOT / "model_store" / "distilbert-20k"
MAX_LENGTH = 128
MAX_CHARS = 5000
LABELS = {0: "negative", 1: "positive"}

app = FastAPI(title="Review Sentiment Classifier", version="1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


class ReviewRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def must_hold_words(cls, v: str) -> str:
        # light_clean keeps punctuation, so "!!!???" would pass an emptiness
        # check and the model would return a meaningless confident answer.
        if not re.search(r"[A-Za-z0-9]", light_clean(v)):
            raise ValueError("text must contain at least one letter or digit")
        return v


class Prediction(BaseModel):
    label: str
    confidence: float
    cleaned_text: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR.name}


@app.post("/predict", response_model=Prediction)
def predict(request: ReviewRequest):
    cleaned = light_clean(request.text)
    batch = tokenizer(cleaned, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt")

    with torch.no_grad():
        probs = torch.softmax(model(**batch).logits, dim=-1)[0]

    index = int(probs.argmax())
    return Prediction(label=LABELS[index],
                      confidence=round(float(probs[index]), 4),
                      cleaned_text=cleaned)

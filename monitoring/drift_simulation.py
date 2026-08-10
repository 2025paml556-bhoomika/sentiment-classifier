"""
Week 4 (M5) — drift simulation
--------------------------------
Sends two batches of reviews to the running API and compares the results:

  BASELINE : real reviews from the test split, the kind the model trained on
  DRIFTED  : hand-written reviews using recent slang and different products

Both batches carry known labels, so this measures a real accuracy drop and
not just a confidence wobble.

Calls the live API rather than loading the model directly, so the requests
also land in logs/predictions.jsonl and feed the 4.3 monitoring work.

Start the API first, then run this from the repo root:
    ./venv/bin/uvicorn serving.api:app --port 8000
    ./venv/bin/python monitoring/drift_simulation.py
"""

import json
import logging
import urllib.error
import urllib.request

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API = "http://localhost:8000/predict"
CLEANED_CSV = "data/processed/cleaned_reviews.csv"
STORE_PARQUET = "feature_store/features.parquet"
BASELINE_N = 200
LOW_CONFIDENCE = 0.8
SEED = 42

# Slang the model never saw, plus products that are not food. 1 = positive.
DRIFTED = [
    ("this snack absolutely slaps no cap, bussin fr", 1),
    ("lowkey goated, been eating it every day", 1),
    ("straight up mid, would not cop again", 0),
    ("took the L on this one, total waste", 0),
    ("it ate and left no crumbs, obsessed", 1),
    ("the flavour is giving nothing honestly", 0),
    ("this is peak, chefs kiss", 1),
    ("mad sus texture, threw it out", 0),
    ("big W purchase, my whole family is hooked", 1),
    ("kinda cheugy packaging but tastes elite", 1),
    ("bruh this ain't it chief", 0),
    ("slept on gem, tell your friends", 1),
    ("caught me lacking, expired on arrival", 0),
    ("no thoughts just vibes, ten out of ten", 1),
    ("this ate down, absolute banger", 1),
    ("cheap build quality, the hinge snapped in a week", 0),
    ("battery lasts two full days, genuinely impressive", 1),
    ("the app crashes every time I open the settings page", 0),
    ("installation took four minutes and it just worked", 1),
    ("driver support is nonexistent on newer systems", 0),
    ("screen is crisp and the colours are accurate", 1),
    ("customer service left my ticket open for three weeks", 0),
    ("fits perfectly and the stitching is solid", 1),
    ("the zip broke on the second use, poor quality", 0),
    ("sound isolation is excellent for the price", 1),
    ("firmware update bricked the whole unit", 0),
    ("assembly instructions were clear and complete", 1),
    ("arrived with a cracked casing and no packaging", 0),
    ("runs quiet and cool even under load", 1),
    ("the subscription auto renewed without any warning", 0),
]


def call_api(text):
    request = urllib.request.Request(
        API, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        return body["label"], body["confidence"]
    except urllib.error.HTTPError as e:
        # 422 means validation rejected it, which is a valid outcome to count.
        return f"rejected-{e.code}", None


def load_baseline():
    """Real test-split reviews, so the baseline is measured not assumed."""
    df = pd.read_csv(CLEANED_CSV,
                     usecols=["review_text", "text_heavy_clean", "label"])
    df = df.dropna(subset=["review_text", "text_heavy_clean", "label"])
    df = df.reset_index(drop=True)

    split = pd.read_parquet(STORE_PARQUET, columns=["review_id", "split"])
    test_ids = split.loc[split["split"] == "test", "review_id"]
    sample = df.loc[test_ids].sample(BASELINE_N, random_state=SEED)
    return list(zip(sample["review_text"], sample["label"]))


def run_batch(name, cases):
    correct = 0
    confidences = []
    rejected = 0

    for text, truth in cases:
        label, confidence = call_api(text)
        if confidence is None:
            rejected += 1
            continue
        confidences.append(confidence)
        if (label == "positive") == bool(truth):
            correct += 1

    scored = len(cases) - rejected
    mean_confidence = sum(confidences) / len(confidences)
    low = sum(1 for c in confidences if c < LOW_CONFIDENCE) / len(confidences)
    accuracy = correct / scored

    logger.info(f"{name}: {scored} scored, accuracy {accuracy:.3f}, "
                f"mean confidence {mean_confidence:.3f}, "
                f"below {LOW_CONFIDENCE} in {low:.1%} of cases")
    return {"accuracy": accuracy, "mean_confidence": mean_confidence,
            "low_confidence_rate": low, "scored": scored, "rejected": rejected}


def main():
    logger.info("checking the API is up")
    label, _ = call_api("this is a test review and it tastes fine")
    if label.startswith("rejected"):
        raise SystemExit("API is not answering, start uvicorn first")

    baseline = run_batch("BASELINE (real test reviews)", load_baseline())
    drifted = run_batch("DRIFTED  (slang and new products)", DRIFTED)

    print()
    print(f"{'metric':<22}{'baseline':>10}{'drifted':>10}{'change':>10}")
    print("-" * 52)
    for key in ["accuracy", "mean_confidence", "low_confidence_rate"]:
        b, d = baseline[key], drifted[key]
        print(f"{key:<22}{b:>10.3f}{d:>10.3f}{d - b:>+10.3f}")
    print()
    print(f"accuracy fell by {(baseline['accuracy'] - drifted['accuracy']):.1%} "
          f"on text the model was never trained on")


if __name__ == "__main__":
    main()

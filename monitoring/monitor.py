"""
Week 4 (M5) — monitoring signals and the retraining trigger
-------------------------------------------------------------
Reads logs/predictions.jsonl and reports the health of the live model.

Every signal here works WITHOUT true labels, because production never tells
you whether a prediction was right. That rules out accuracy and leaves
confidence, prediction mix, and input shape as the things we can watch.

Thresholds come from the drift experiment in reports/drift_report.md, not
from guesswork:

    low-confidence rate   10% on normal traffic, 40% under drift  -> alert at 25%
    mean confidence       0.948 normal, 0.816 under drift         -> alert below 0.90
    positive share        84.3% in the training data              -> alert if 15+ points off

Run from the repo root:
    ./venv/bin/python monitoring/monitor.py
"""

import json
import statistics as stats
from pathlib import Path

LOG_PATH = Path("logs/predictions.jsonl")
WINDOW = 100

LOW_CONFIDENCE = 0.8
MAX_LOW_CONFIDENCE_RATE = 0.25
MIN_MEAN_CONFIDENCE = 0.90
TRAINING_POSITIVE_SHARE = 0.843
MAX_POSITIVE_SHARE_DRIFT = 0.15
MAX_P95_LATENCY_MS = 200.0


def load_window(path=LOG_PATH, window=WINDOW):
    """Most recent `window` predictions. Old traffic should not mask today's."""
    if not path.exists():
        raise SystemExit(f"no predictions logged yet at {path}")
    rows = [json.loads(line) for line in path.open()]
    return rows[-window:]


def signals(rows):
    confidence = [r["confidence"] for r in rows]
    words = [r["word_count"] for r in rows]
    latency = sorted(r["latency_ms"] for r in rows)
    positive = sum(1 for r in rows if r["label"] == "positive")

    return {
        "predictions": len(rows),
        "mean_confidence": stats.mean(confidence),
        "low_confidence_rate": sum(1 for c in confidence if c < LOW_CONFIDENCE) / len(rows),
        "positive_share": positive / len(rows),
        "median_word_count": stats.median(words),
        "p95_latency_ms": latency[min(int(len(latency) * 0.95), len(latency) - 1)],
    }


def checks(s):
    """Each check returns (name, value, threshold text, ok)."""
    share_drift = abs(s["positive_share"] - TRAINING_POSITIVE_SHARE)
    return [
        ("low confidence rate", f"{s['low_confidence_rate']:.1%}",
         f"below {MAX_LOW_CONFIDENCE_RATE:.0%}",
         s["low_confidence_rate"] <= MAX_LOW_CONFIDENCE_RATE),
        ("mean confidence", f"{s['mean_confidence']:.3f}",
         f"above {MIN_MEAN_CONFIDENCE}",
         s["mean_confidence"] >= MIN_MEAN_CONFIDENCE),
        ("positive share", f"{s['positive_share']:.1%}",
         f"within {MAX_POSITIVE_SHARE_DRIFT:.0%} of {TRAINING_POSITIVE_SHARE:.1%}",
         share_drift <= MAX_POSITIVE_SHARE_DRIFT),
        ("p95 latency", f"{s['p95_latency_ms']:.1f}ms",
         f"below {MAX_P95_LATENCY_MS:.0f}ms",
         s["p95_latency_ms"] <= MAX_P95_LATENCY_MS),
    ]


def should_retrain(s):
    """
    Task 4.4 trigger.

    Retrain when more than 25% of the last 100 predictions fall below 0.8
    confidence. Normal traffic sits at 10% and drifted traffic reached 40%,
    so 25% clears everyday variation while catching real drift.

    Confidence is the trigger rather than accuracy because live traffic
    carries no labels. Prediction mix and input length are reported as
    supporting evidence, not as triggers, since a genuine change in what
    customers write can move them without the model getting worse.
    """
    if s["predictions"] < WINDOW:
        return False, (f"only {s['predictions']} predictions, need {WINDOW} "
                       f"before judging")
    if s["low_confidence_rate"] > MAX_LOW_CONFIDENCE_RATE:
        return True, (f"{s['low_confidence_rate']:.1%} of the last {WINDOW} "
                      f"predictions fell below {LOW_CONFIDENCE} confidence")
    return False, (f"low confidence rate {s['low_confidence_rate']:.1%} is within "
                   f"the {MAX_LOW_CONFIDENCE_RATE:.0%} limit")


def main():
    rows = load_window()
    s = signals(rows)

    print(f"\nlast {s['predictions']} predictions\n")
    print(f"{'signal':<22}{'value':>12}   {'healthy range':<34}{'state'}")
    print("-" * 82)
    for name, value, threshold, ok in checks(s):
        print(f"{name:<22}{value:>12}   {threshold:<34}{'OK' if ok else 'ALERT'}")

    print(f"\nmedian input length: {s['median_word_count']:.0f} words")

    retrain, reason = should_retrain(s)
    print(f"\nretrain now: {'YES' if retrain else 'no'} — {reason}\n")


if __name__ == "__main__":
    main()

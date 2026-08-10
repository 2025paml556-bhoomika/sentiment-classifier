# Monitoring and retraining design

Tasks 4.3 and 4.4. Implemented in `monitoring/monitor.py`, reading
`logs/predictions.jsonl`.

## The constraint that shapes everything

Production never tells you whether a prediction was right. No customer confirms that a
review really was negative. So **accuracy cannot be measured on live traffic**, and any
signal that needs true labels is unusable here.

That leaves three things we can observe: how confident the model is, what mix of answers
it produces, and what the incoming text looks like.

## Signals

Computed over the most recent 100 predictions, so yesterday's traffic cannot mask today's.

| Signal | Healthy range | Why it is watched |
|---|---|---|
| Low confidence rate, share below 0.8 | below 25% | The primary drift signal. Failures under drift were consistently low-confidence, so this stands in for the accuracy we cannot see. |
| Mean confidence | above 0.90 | Slower moving than the rate above, and it confirms a broad shift rather than a few odd inputs. |
| Positive share | within 15 points of 84.3% | Training data was 84.3% positive. A large swing means the incoming mix changed, or the model started favouring one class. |
| p95 latency | below 200 ms | Operational health, not drift. Catches a slow or overloaded service. |
| Median input length | reported, not alerted | Supporting context. Sharply shorter text often means a new channel or new writing style. |

Every threshold comes from the drift experiment in `reports/drift_report.md` rather than
from guesswork. Normal traffic showed a 10% low-confidence rate and 0.948 mean confidence.
Drifted traffic showed 40% and 0.816.

## The retraining trigger

> **Retrain when more than 25% of the last 100 predictions fall below 0.8 confidence.**

Why 25%: normal traffic sits near 10% and drifted traffic reached 40%. A limit of 25% clears
everyday variation while still catching real drift.

Why 100 predictions: large enough that a handful of odd inputs cannot trip the alarm, small
enough to react within a day of normal traffic.

Why confidence and not accuracy: live traffic carries no labels, as explained above.

Prediction mix and input length are deliberately **reported but not used as triggers**. A
genuine change in what customers write can move both without the model getting any worse.
Retraining on those alone would waste effort.

## Validation

The monitor was run against two windows drawn from the same log, to check it discriminates
rather than always alerting.

| Signal | Normal traffic | Drifted traffic |
|---|---|---|
| Low confidence rate | 15.0% OK | 27.0% ALERT |
| Mean confidence | 0.930 OK | 0.875 ALERT |
| Positive share | 76.0% OK | 65.0% ALERT |
| p95 latency | 30.6 ms OK | 23.8 ms OK |
| **Trigger fires** | **no** | **yes** |

Latency stayed healthy in both, which is correct: drift changes what the model reads, not
how fast it runs. That the same code passes one window and alerts on the other is the
evidence that these thresholds are usable.

## What happens when the trigger fires

1. Collect the recent low-confidence inputs from `logs/predictions.jsonl`.
2. Have them labelled by hand. They are the examples the current model finds hardest.
3. Add them to `data/processed/cleaned_reviews.csv` and rebuild the feature store, so
   training and serving features stay in step.
4. Re-run `training/train_distilbert.py`, then compare against the current model in MLflow
   on the same held-out test split.
5. Promote the new model only if macro F1 improves. Keep the previous weights in
   `model_store/` so a rollback is possible.

Step 5 matters: a trigger that retrains automatically without a comparison gate can quietly
ship a worse model.

## Limitations

- Thresholds rest on a single drift experiment with 30 hand-written reviews. They should be
  revisited once real traffic exists.
- A model can be confidently wrong. Confidence tracks drift well here, but it is a proxy,
  not a guarantee.
- The window is a fixed count rather than a time period, so during quiet traffic it may span
  several days.

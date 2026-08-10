# Drift simulation report

Task 4.2. Produced by `monitoring/drift_simulation.py` on 10 August 2026, against the
live API serving `distilbert-20k`.

## Method

Two batches of labelled reviews were sent through the running API, so both took the real
serving path rather than calling the model directly.

- **Baseline**: 200 real reviews sampled from the held-out test split. This is the kind of
  text the model trained on, so it sets the honest reference point rather than an assumed one.
- **Drifted**: 30 hand-written reviews in two halves. Fifteen use recent slang about food,
  the familiar topic. Fifteen use plain English about electronics and software, an unfamiliar
  topic.

Both batches carry known labels, so this measures a genuine accuracy drop rather than only
a shift in confidence.

## Results

| Metric | Baseline | Drifted | Change |
|---|---|---|---|
| Accuracy | 0.925 | 0.867 | −0.058 |
| Mean confidence | 0.948 | 0.816 | −0.132 |
| Share below 0.8 confidence | 10.0% | 40.0% | +30.0 points |

Accuracy fell by 5.8 points. Confidence fell more than twice as far, and the share of
low-confidence answers quadrupled.

## Which kind of drift hurts more

| Drift type | Accuracy | Mean confidence |
|---|---|---|
| Slang, familiar topic | 0.800 | 0.775 |
| New topic, plain English | 0.933 | 0.857 |

**New vocabulary hurts roughly twice as much as a new topic.** Electronics reviews written
in plain English still classify well, because DistilBERT's pretraining covers general
English usage. Slang was absent from both its pretraining and this project's fine-tuning
data, so phrases like "this ate down" and "absolute banger" carry no learned sentiment.

## The finding that matters for monitoring

Every one of the four failures arrived with low confidence:

| Confidence | Said | Truth | Text |
|---|---|---|---|
| 0.5004 | negative | positive | no thoughts just vibes, ten out of ten |
| 0.5612 | positive | negative | bruh this ain't it chief |
| 0.6587 | negative | positive | this ate down, absolute banger |
| 0.6715 | positive | negative | cheap build quality, the hinge snapped in a week |

The 0.5004 case is effectively a coin flip, and the model reports that honestly.

This matters because **production has no true labels**. Nobody tells you a live prediction
was wrong, so accuracy cannot be measured on live traffic. Confidence can. Since failures
here are consistently low-confidence, the share of answers below a confidence threshold is
a usable stand-in for the accuracy you cannot see.

That is the basis for the retraining trigger in task 4.4. The low-confidence rate moved from
10% to 40% under drift, a far clearer signal than the 5.8-point accuracy drop, and it is
measurable without labels.

## Limitations

- 30 drifted reviews is a small sample, so treat each rate as indicative rather than precise.
- The drifted text was written by hand for this test, so it reflects an assumption about how
  real drift would look.
- The model under test trained on a 20,000-row sample for one epoch. A stronger model might
  absorb some of this drift.

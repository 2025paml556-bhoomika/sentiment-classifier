# Model comparison

Task 2.4. Seven runs, all logged to MLflow (`sqlite:///mlflow.db`).

## Words used in this report

- **Negative** means a bad review, which is what we want to catch.
- **Precision** is how often the model is right when it says "negative".
- **Recall** is the share of real complaints the model finds.
- **Macro F1** averages the score for both classes equally. It does not let the
  large positive class hide a weak negative class.
- **ROC AUC** measures how well a model sorts reviews from worst to best. It
  does not depend on any cut-off value.
- **Threshold** is the cut-off on the model's score. Above it, we call a review
  negative. The default is 0.5.

## How we kept the comparison fair

Every model is scored on the **same 72,765 test rows**. We read the rows from the
feature store's `split` column. We do not split the data again per model. Without
this, each model would face a different test set. The numbers could not be compared.

We checked this holds. Both `training.py` and `build_features.py` produce exactly
the same test rows as the feature store.

The test set holds **11,413 real negative reviews**, or 15.7% of the total. This
imbalance is why macro F1 leads the table instead of accuracy. A model that always
answers "positive" would score 84.3% accuracy and be useless.

## Results

| Run | Train rows | Accuracy | Macro F1 | ROC AUC | Neg. precision | Neg. recall | Run time |
|---|---|---|---|---|---|---|---|
| `distilbert-50k` | 50,000 | **0.9510** | **0.9064** | **0.9799** | 0.8517 | 0.8323 | 26.2 min |
| `distilbert-20k` | 20,000 | 0.9457 | 0.8917 | 0.9762 | 0.8752 | 0.7628 | 14.7 min |
| `logreg-tfidf-bigram` | 291,060 | 0.9249 | 0.8729 | 0.9750 | 0.7009 | **0.9094** | 34 s |
| `logreg-50k-balanced` | 50,000 | 0.9141 | 0.8547 | 0.9650 | 0.6741 | 0.8758 | 16 s |
| `logreg-tfidf-unigram` | 291,060 | 0.8977 | 0.8347 | 0.9608 | 0.6209 | 0.8934 | 23 s |
| `logreg-50k-unweighted` | 50,000 | 0.9246 | 0.8339 | 0.9642 | **0.8917** | 0.5914 | 13 s |
| `logreg-feature-store-svd300` | 291,060 | 0.8615 | 0.7878 | 0.9387 | 0.5362 | 0.8675 | 16 s |

Run times come from the MLflow start and end timestamps. Read them with care:

- The two DistilBERT times include scoring the 72,765 test rows, which alone takes
  about 7 minutes. They ran on the Apple GPU through MPS. Every other run used CPU.
- The Logistic Regression times include saving the model to MLflow.
- `logreg-feature-store-svd300` looks cheap at 16 seconds. That is misleading. It
  reads features that `build_features.py` already computed, and that build is a
  separate and much larger cost.

The gap is the useful part. Two ways to read the cost:

- Against the best baseline, `logreg-tfidf-bigram`, DistilBERT costs about **46 times**
  more time and buys 0.034 macro F1.
- At equal data, against `logreg-50k-balanced`, it costs about **98 times** more time
  and buys 0.052 macro F1.

So DistilBERT is the better model, but it is never the cheap one. On a machine without
a GPU the gap grows much wider, because it cannot fall back on fast matrix hardware.

`distilbert-50k` wins every overall score, meaning accuracy, macro F1 and ROC AUC.
Two per-class scores go elsewhere. `logreg-tfidf-bigram` has the best negative
recall, and `logreg-50k-unweighted` the best negative precision. The sections below
explain why neither fact makes those models the better choice.

## Both models on the same 50,000 rows

The first comparison was not fair on training size. DistilBERT trained on 20,000
rows. The baselines trained on 291,060. So the result mixed two things at once:
the model type and the amount of data.

We fixed this. We drew one sample of 50,000 rows and trained both models on it.
The sample comes from the train split only, so no test rows leak in. See
`training/train_controlled_50k.py`.

| Comparison | Gap in macro F1 |
|---|---|
| Old, unequal data: `distilbert-20k` vs `logreg-tfidf-bigram` | +0.0188 |
| New, equal data at 50,000 rows | **+0.0517** |

**The unequal data was hiding how good DistilBERT is.** With equal data, its lead
almost triples. The baseline had been getting 5.8 times more data.

More data also fixed DistilBERT's weak spot. Going from 20,000 to 50,000 rows:

| Metric | 20k | 50k | Change |
|---|---|---|---|
| Negative recall | 0.7628 | 0.8323 | **+0.0695** |
| Negative precision | 0.8752 | 0.8517 | −0.0235 |
| Macro F1 | 0.8917 | 0.9064 | +0.0147 |

Recall improved a lot. Precision dropped a little. Note that we did not add class
weighting. The extra negative examples were enough on their own.

## Why DistilBERT wins

**Pretraining.** DistilBERT already understood English before this project. Other
people trained it on a large body of text first. Fine-tuning only had to teach it
what a bad review looks like. It did not have to learn what words mean. The
Logistic Regression models start from zero. They learn everything from these
reviews alone.

**Word order.** DistilBERT reads text as a sequence. TF-IDF only counts words. For
example, "not good" and "good" share the same `good` count. So TF-IDF cannot tell
them apart. Bigrams, which are word pairs, recover part of this. But pairs are a
weaker tool than reading the whole sentence.

## What bigrams actually add

The bigram run beats the unigram run by 0.0383 macro F1. Two changes cause this at
once. We added word pairs. We also grew the vocabulary from 5,000 to 20,000 terms.

To separate them, we ran all three settings together on one Linux machine. The
missing setting is a 20,000-term run using single words only. It scored 0.8409.

| Change | Gain in macro F1 | Share of the gap |
|---|---|---|
| Vocabulary 5,000 to 20,000, words only | +0.0062 | 16% |
| Adding word pairs at 20,000 terms | **+0.0322** | **84%** |

So word pairs do most of the work. A bigger vocabulary adds little.

Those three runs total 0.0384, against 0.0383 in the table above. The gap differs in
the fourth decimal because the Linux machine had different library versions. We ran
them outside MLflow, so they are not logged as experiments.

## The threshold matters more than the model choice

Every score above uses a threshold of 0.5. That is just a default. It is not a
property of a model. A support-ticket classifier should pick its threshold from the
recall the business needs.

So we compared the two best candidates at matched recall. See
`training/threshold_analysis.py`.

| Target recall | LogReg precision | DistilBERT precision | LogReg false alarms | DistilBERT false alarms |
|---|---|---|---|---|
| 0.800 | 0.835 | **0.877** | 1,798 | **1,279** |
| 0.850 | 0.785 | **0.837** | 2,657 | **1,896** |
| 0.900 | 0.719 | **0.772** | 4,023 | **3,033** |
| 0.909 | 0.701 | **0.758** | 4,428 | **3,313** |
| 0.930 | 0.658 | **0.709** | 5,529 | **4,354** |
| 0.950 | 0.601 | **0.646** | 7,191 | **5,930** |

**DistilBERT wins at every recall level.** There is no cut-off where the baseline
is better.

Look at the 0.909 row. That is the baseline's own operating point. Both models
catch 10,379 complaints and miss 1,034. But DistilBERT raises **1,115 fewer false
alarms**, which is 3,313 against 4,428. That saves about a quarter of the wasted
agent time.

DistilBERT needs a threshold of 0.265 to reach that recall, not 0.5. The reason is
simple. It trained without class weighting on data that is 5.4 to 1 positive. So
its scores lean toward positive. At 0.5 it flags too few complaints. This is a
calibration problem, not a weakness in the model.

## Which model to ship

**Ship `distilbert-50k`, and set the threshold on purpose.**

For support tickets, a missed complaint costs more than a false alarm. An agent
dismisses a false alarm in seconds. A missed complaint can lose a customer. So
recall should be high.

Counting reviews at the default 0.5 threshold:

| Model | Complaints caught | Complaints missed | False alarms |
|---|---|---|---|
| `logreg-tfidf-bigram` | 10,379 | 1,034 | 4,429 |
| `logreg-50k-balanced` | 9,996 | 1,417 | 4,833 |
| `distilbert-50k` | 9,499 | 1,914 | 1,654 |
| `distilbert-20k` | 8,706 | 2,707 | 1,242 |
| `logreg-50k-unweighted` | 6,750 | 4,663 | 820 |

At 0.5 the baseline catches the most complaints. That is why an earlier version of
this report argued for the baseline. **The threshold test above shows that
conclusion was wrong.** The baseline only looked better because 0.5 suits it and
not DistilBERT. Once we tune the threshold, DistilBERT is better everywhere.

Rule out `logreg-50k-unweighted` for this use case. It misses 4,663 complaints,
which is 41% of them.

Two changes follow for serving. `serving/api.py` currently loads `distilbert-20k`
and uses 0.5. It should load the 50k weights. The threshold should become a setting.

## Why class weighting matters

The two 50,000-row Logistic Regression runs differ in one setting only, which is
`class_weight`. The effect is large.

| Run | Accuracy | Macro F1 | Neg. recall | Complaints caught | False alarms |
|---|---|---|---|---|---|
| `logreg-50k-balanced` | 0.9141 | **0.8547** | **0.8758** | **9,996** | 4,833 |
| `logreg-50k-unweighted` | **0.9246** | 0.8339 | 0.5914 | 6,750 | **820** |

Weighting catches 3,246 more complaints. It also raises about six times more false
alarms.

This pair also shows why we lead with macro F1. The unweighted run has **higher
accuracy** but much **lower macro F1**. Accuracy rewards it for being right about
the large positive class. It hides the fact that the model misses 41% of complaints.

## The feature store costs accuracy

`logreg-feature-store-svd300` comes last, at 0.7878 macro F1. The bigram baseline
scores 0.8729. Both start from the same 20,000 TF-IDF features. The only difference
is SVD, which compresses them to 300 columns. That compression costs **0.085 macro
F1**.

This is expected, not a defect. Squeezing 20,000 features into 300 loses
information. What it buys is a fixed 300-column layout and one saved transformer
shared by training and serving. So the two cannot compute features differently.

SVD is still the right way to reach 300 columns. It blends all 20,000 features into
combinations. The alternative is keeping only the top 300 terms and throwing away
19,700. SVD is worth 0.063 macro F1 over that alternative.

## Caveats

- **One epoch.** Both DistilBERT runs saw the data once. They likely have more room
  to improve.
- **No class weighting on DistilBERT.** Most Logistic Regression runs use
  `class_weight="balanced"`. DistilBERT uses plain cross-entropy. Weighting the loss
  is still worth testing, though the threshold fix already handles the symptom.
- **Single split.** We used one train/test split and no cross-validation. Small
  differences should not be read too closely.
- **Different text cleaning.** DistilBERT reads `text_light_clean`, which keeps
  sentence structure. The Logistic Regression models read `text_heavy_clean`, which
  removes stopwords. This is on purpose. Each model needs its own preparation.
- **Thresholds are tuned on the test set.** The table above picks cut-offs using the
  same rows we score on. A separate validation split would be stricter.

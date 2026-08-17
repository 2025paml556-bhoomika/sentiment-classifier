# Week 1 (M2) — Data Validation Report

Flavor C, Support Ticket / Review Sentiment Classifier. Dataset: Amazon Fine
Food Reviews. All figures come from a full pipeline run on 5 August 2026.

Reproduce with:

```bash
./venv/bin/python src/pipeline.py --input data/raw/Reviews.csv --text-col Text --rating-col Score
```

## Summary

We ingested 568,454 reviews and kept 363,825, dropping 36.0% of the data.
Almost all of that loss is duplicate review text. The data has no nulls, no
empty fields, and no invalid ratings. The one serious quality problem is class
imbalance: the surviving reviews are 84.3% positive.

## Row accounting

Each step runs in the order shown, so the counts are sequential rather than
independent.

| Step | Rows removed | Rows remaining |
|---|---|---|
| Ingested from CSV | — | 568,454 |
| Drop null text or rating | 0 | 568,454 |
| Drop empty text | 0 | 568,454 |
| Drop duplicate review text | 174,875 | 393,579 |
| Drop neutral (3-star) reviews | 29,754 | 363,825 |
| Drop rows left empty by cleaning | 0 | 363,825 |
| **Total** | **204,629 (36.0%)** | **363,825** |

Deduplication accounts for 85.5% of everything dropped. The Amazon dataset
repeats the same review text across different product IDs, so this is expected
rather than a sign of corruption.

## Class balance

| Label | Rows | Share |
|---|---|---|
| Positive (4-5 stars) | 306,758 | 84.3% |
| Negative (1-2 stars) | 57,067 | 15.7% |

The imbalance ratio is 5.38 to 1.

**This is the most important finding in this report.** A model that ignores the
text and always answers "Positive" would score about 84.3% accuracy. Any
accuracy figure near that number means nothing. Week 2 must therefore report
macro F1, which averages the score across both classes, and must train with
balanced class weights. We confirmed this empirically: a Logistic Regression
baseline on 5,000 TF-IDF features reaches 89.8% accuracy but only 0.835 macro
F1, so accuracy overstates quality by a wide margin.

## Data quality findings

**No missing or malformed data.** Null text, null ratings, empty strings, and
out-of-range star values all came to zero. No repair logic was needed.

**Heavy duplication.** 174,875 reviews share text with another review, which is
30.8% of the raw file. We deduplicate on `review_text` alone, ignoring product
and user, because identical text carries identical sentiment and would
otherwise leak between the training and test splits.

**Wide length range.** Reviews run from 3 words to 3,432 words, averaging 80.3.
The long tail matters for the transformer model, because DistilBERT truncates at
128 tokens, so the longest reviews lose most of their text.

**The JSON report measures class balance before deduplication.** Reading
`data/processed/validation_report.json`, the `class_balance` block shows 443,777
positive, 82,037 negative, and 42,640 neutral. Those are raw counts taken during
validation, which runs before cleaning. After deduplication only 29,754 of the
3-star rows remain to be dropped, because 12,886 of them were already removed as
duplicates. Use the post-cleaning figures above when reporting model results.

## Design decisions

**Two classes, not three.** We drop 3-star reviews rather than modelling a
neutral class. Neutral sentiment in star ratings is ambiguous, and a clean
binary target makes the model comparison in Week 2 easier to interpret. Cost:
29,754 discarded reviews.

**Threshold at 4 and above.** Ratings of 4 and 5 map to Positive, 1 and 2 to
Negative.

**Two cleaned versions of every review.** The classical and transformer models
need different input, so `cleaning.py` produces both.

| Version | Treatment | Consumer |
|---|---|---|
| `text_heavy_clean` | Lowercased, punctuation stripped, stopwords removed | TF-IDF |
| `text_light_clean` | HTML stripped, whitespace collapsed, structure kept | DistilBERT |

DistilBERT needs full sentence structure, including stopwords, because it reads
word order. TF-IDF ignores order, so removing filler words there only helps.

**Validation can halt the pipeline.** `validate_data()` raises
`DataValidationError` on critical failures, currently a row count below 100 or
missing required columns. Everything else is reported but does not block, so
that quality drift is visible without stopping a run.

## Artifacts produced

| File | Size | Contents |
|---|---|---|
| `data/processed/cleaned_reviews.csv` | 422 MB | 363,825 rows, 5 columns, DVC-tracked |
| `data/processed/validation_report.json` | 338 B | Machine-readable validation output |

TF-IDF features are not built here. They come later, from
`tfidf/build_features.py`, which fits on the train split only.

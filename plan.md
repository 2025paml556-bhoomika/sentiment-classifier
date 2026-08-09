# Flavor C — Support Ticket / Review Sentiment Classifier

ML Engineering Mini-Project (PCAM ZG412). Group submission, deadline **24 August 2026**.

Evaluation is holistic at the end, not week by week. All three final artifacts are
mandatory. The assignment is not evaluated at all without the video.

## Status snapshot

| Week | Milestone | Focus | State |
|---|---|---|---|
| 1 | M2 | Data ingestion, validation, features | 10 of 10 tasks done |
| 2 | M3 | Model training, experiment tracking | 4 of 6 tasks done |
| 3 | M4 | Packaging, deployment | 2 of 5 tasks done |
| 4 | M5 | Monitoring, drift, retraining | Not started |
| — | added | Feature store (professor's request) | Built and versioned |

Verified end to end on 8 Aug 2026. The pipeline takes about 23 seconds on the full
568k-row dataset. Re-running produced a byte-identical `cleaned_reviews.csv`
(md5 `7a6af401b4782822d0606ac2c96c3084`), so the data stages are reproducible.

## Repo layout

The project follows the reference folder structure. There is no `src/` directory.

```
pipeline.py                           end-to-end Week 1 run, writes artifacts
harness_m2.py                         same stages, prints input/output, writes nothing
validation/ingestion.py               load raw CSV, rename columns
validation/validate_data.py           quality checks, halts on critical failure
features/cleaning.py                  heavy_clean (TF-IDF) + light_clean (DistilBERT)
features/feature_engineering.py       TF-IDF builder, DistilBERT tokenizer
features/build_features.py            builds the feature store
training/training.py                  baseline models from cleaned CSV
training/train_model_from_store.py    model trained from the feature store
training/train_distilbert.py          DistilBERT fine-tune on the Apple GPU
feature_store/                        features.parquet, transformer.pkl, schema.json
model_store/distilbert-20k/           fine-tuned weights, 256 MB, DVC-tracked
serving/  ui/                         placeholders, not yet written
reports/validation_report.md          the 1.10 write-up
```

Reproduce Week 1 with:

```bash
venv/bin/python pipeline.py --input data/raw/Reviews.csv --text-col Text --rating-col Score
```

Step through the same stages one at a time, writing nothing:

```bash
venv/bin/python harness_m2.py
```

## Verified data numbers (8 Aug 2026)

| Stage | Result |
|---|---|
| Ingested | 568,454 rows from a 301 MB CSV |
| Validation | 0 nulls, 0 empty, 0 invalid ratings, 174,875 duplicates found |
| After cleaning | 363,825 rows, 204,629 dropped (duplicates, then 3-star neutrals) |
| Labels | 306,758 positive / 57,067 negative, a 5.4-to-1 split |
| TF-IDF | 363,825 x 5,000 sparse matrix |

Drops are sequential, so the per-step counts in `reports/validation_report.md`
supersede the raw figures in `data/processed/validation_report.json`.

## Week 1 (M2) — Data ingestion, validation & feature pipeline

- [x] **1.1 Select dataset.** Amazon Fine Food Reviews, 568,454 rows at `data/raw/Reviews.csv`. The `Text` and `Score` columns match what the code expects. Still needs registering in the group spreadsheet.
- [x] **1.2 Data ingestion.** `validation/ingestion.py` loads raw CSV into a DataFrame and logs row counts. Scripted, no manual steps.
- [x] **1.3 Data validation.** `validation/validate_data.py` checks nulls, duplicates, empty text, rating range, class balance, and review-length distribution. Raises `DataValidationError` on critical failures, which halts the pipeline.
- [x] **1.4 Label creation.** `label_sentiment()` in `features/cleaning.py` drops 3-star rows and maps rating >= 4 to 1 (Positive), <= 2 to 0 (Negative).
- [x] **1.5 Text cleaning, two versions.** `heavy_clean()` removes stopwords for TF-IDF; `light_clean()` keeps sentence structure for DistilBERT.
- [x] **1.6 Feature pipeline A.** `build_tfidf_features()` fits or reuses a TF-IDF vectorizer. Accepts an existing vectorizer so Week 4 retraining reuses the same vocabulary.
- [x] **1.7 Feature pipeline B.** `build_bert_tokenized_features()` returns `input_ids` and `attention_mask` for DistilBERT.
- [x] **1.8 Dataset versioning.** DVC 3.67.1. Five things tracked: the two datasets, the two feature-store artifacts, and the DistilBERT weights directory. Git holds only the `.dvc` pointer files, each carrying an md5 and a byte size. The data lives in `.dvc/cache` and is ignored by git. The `.gitignore` entries for the two CSVs were removed, because DVC refuses to track files git already ignores; DVC then wrote its own ignore rules.
- [x] **1.9 Repo setup.** Git repo using the reference folder layout. `.gitignore` excludes pickles, `venv/`, `mlruns/`, and `mlflow.db`.
- [x] **1.10 Validation report.** Written to `reports/validation_report.md`: row accounting, class balance, data-quality findings, and the design decisions behind them.

## Feature store (added requirement)

Built by `features/build_features.py`. Not part of `pipeline.py`, which never touches it.

- Input is `data/processed/cleaned_reviews.csv`.
- TF-IDF with 20,000 features and unigrams plus bigrams, then TruncatedSVD to 300 dense dimensions.
- The transformer is **fitted on the 291,060 train rows only**, then applied to all 363,825 rows. Fitting on everything would bake test-set word statistics into every feature value. Transforming all rows is safe, because it only applies rules already learned.
- Output: `features.parquet` (581 MB, fixed 300-column schema), `transformer.pkl` (47 MB, reused at serving time), `schema.json` (what the columns mean).
- Split is recorded in a `split` column, 291,060 train and 72,765 test.

Rebuild it with:

```bash
venv/bin/python features/build_features.py
```

The store's purpose is consistency: serving reuses the saved transformer, so training
and serving features cannot disagree. It costs accuracy, which is covered under open
items below.

## Week 2 (M3) — Model training & experiment tracking

Four distinct runs are logged to MLflow, backed by `sqlite:///mlflow.db`. All are scored
on the same 72,765 test rows, taken from the feature store's `split` column, so these
numbers compare models and not splits.

| Run | Accuracy | Macro F1 | ROC AUC | Neg. precision | Neg. recall | Train rows |
|---|---|---|---|---|---|---|
| `distilbert-20k` | **0.9457** | **0.8917** | **0.9762** | **0.875** | 0.763 | 20,000 |
| `logreg-tfidf-bigram` | 0.9249 | 0.8729 | 0.9750 | 0.701 | **0.909** | 291,060 |
| `logreg-tfidf-unigram` | 0.8977 | 0.8347 | 0.9608 | 0.621 | 0.893 | 291,060 |
| `logreg-feature-store-svd300` | 0.8615 | 0.7878 | 0.9387 | 0.536 | 0.868 | 291,060 |

The last two columns matter more than the winner. DistilBERT is the more careful model:
when it calls a review negative it is right 87.5% of the time, against 70.1% for the
baseline. But it catches fewer negatives, 76.3% against 90.9%. For support tickets,
missing an unhappy customer usually costs more than a false alarm a human can dismiss,
so the baseline may still be the better product choice despite the lower macro F1.

- [x] **2.0 Handle class imbalance.** The cleaned set is about 84% positive, a 5.4-to-1 split. A model that always predicts Positive would score about 84% accuracy while being useless. Both Logistic Regression trainers use `class_weight="balanced"` and every run reports macro F1 rather than accuracy. **Not yet applied to DistilBERT**, which trains on plain cross-entropy with the natural class ratio. That is the likely cause of its weak negative recall, so weighting the loss is the obvious next experiment.
- [x] **2.1 Baseline model.** Logistic Regression on TF-IDF, in unigram (5k) and bigram (20k) variants, via `training/training.py`. The bigram run wins by 0.038 macro F1, though both bigrams and the larger vocabulary contribute, so the credit is shared.
- [x] **2.2 Advanced model.** DistilBERT fine-tuned via `training/train_distilbert.py`, on the Apple GPU through MPS. No Colab needed. One epoch, 20,000-row stratified subsample, batch 16, `max_length` 128, about 15 minutes. Loss fell from 0.38 to 0.15. Weights are in `model_store/distilbert-20k`, DVC-tracked because `model.safetensors` is 256 MB and GitHub rejects anything over 100 MB.
- [x] **2.3 Experiment tracking.** Parameters, metrics, and a classification report logged for every run. `train_model_from_store.py` also logs a combined transformer-plus-classifier pipeline, so a served sklearn model carries its own feature logic.
- [ ] **2.4 Model comparison.** The numbers are above and the precision-versus-recall trade-off is understood. The written comparison for the report is still to draft.
- [ ] **2.5 Reproducibility check.** A teammate must reproduce the winning run from the logged config alone. Blocked on pinning versions, see open items.

## Week 3 (M4) — Packaging & deployment

- [ ] **3.1 Model serialization.** Half done. DistilBERT weights are saved to `model_store/distilbert-20k` and DVC-tracked. The sklearn models still live only inside MLflow, so export the chosen one if it is ever served.
- [x] **3.2 REST API.** `serving/api.py`, FastAPI. `POST /predict` takes text and returns label, confidence, and the cleaned text. `GET /health` reports the loaded model. Serves DistilBERT on CPU, not MPS, because the Docker image in 3.4 has no Apple GPU, and matching devices keeps local and container behaviour identical. Input is passed through `light_clean()` first, the same function used in training.
- [x] **3.3 Input validation.** Verified against 10 cases. Rejected with HTTP 422: empty string, whitespace only, punctuation only, emoji only, HTML only, non-string types, null, missing field, wrong field name, and text over 5,000 characters. The punctuation-only case needed a real fix, since `light_clean` keeps punctuation, so `!!!???` first slipped through and returned a meaningless "positive" at 0.63 confidence. The check now requires at least one letter or digit.
- [ ] **3.4 Containerize.** Write a Dockerfile, then build and test the image locally. Note the image must `dvc pull` the weights, or copy them in.
- [ ] **3.5 API testing.** Postman collection or curl commands with sample requests and responses.

## Week 4 (M5) — Monitoring, drift & retraining

- [ ] **4.1 Prediction logging.** Log input, output, confidence, and timestamp for every prediction.
- [ ] **4.2 Drift simulation.** Feed text with unseen slang and topics, then observe the accuracy drop.
- [ ] **4.3 Monitoring setup.** Define signals such as rolling confidence average and prediction distribution.
- [ ] **4.4 Retraining trigger design.** Write the rule down, for example "retrain if confidence < X for Y% of predictions".
- [ ] **4.5 Architecture diagram.** Finalize the pipeline diagram for the report.
- [ ] **4.6 Video script and recording.** Script and record the walkthrough to the full spec in artifact 3 below, which lists every stage the narration must cover.

## Final submission — 3 required artifacts

1. **GitHub repository.** DVC-versioned dataset plus all pipeline code. Commit history must reflect weekly progress, so commit as you go rather than in one batch. Must be accessible to evaluators.
2. **Report**, uploaded to the course page. Covers design decisions, validation results, experiment tracking logs, model comparison, monitoring log, drift-simulation report, retraining design, architecture diagram, and setup instructions.
3. **Video demonstration.** Max 10 minutes, voiceover narration required. A silent screen recording does not count. Cover the full chain: data collection, preprocessing, feature engineering, model building, tracking, deployment, predictions, monitoring, retraining.

## Open items to resolve

- **No README.** The repo has none at all. This is a 20% rubric criterion and the
  largest single scoring gap. It needs setup instructions and an architecture diagram.
- **The feature store model is the weakest.** Macro F1 of 0.7878 against 0.8729 for
  plain bigram TF-IDF, so the store costs 0.085 macro F1. This is expected, because
  compressing 20,000 features into 300 loses information. The docstring in
  `build_features.py` cites "+0.063 macro F1" for SVD, but that compares SVD against
  picking the top 300 TF-IDF terms, not against the full 20,000. Reword it, and state
  the trade-off in the report: the store buys training-serving consistency and pays in
  accuracy. Otherwise a reviewer reading the MLflow runs will ask why the weakest model
  is the one wired for serving.
- **Three duplicate MLflow runs.** `logreg-feature-store-svd300` is logged three times
  with identical metrics. Delete two before presenting the comparison.
- **No DVC remote.** Nothing is configured, so evaluators cannot run `dvc pull`. The
  cache is local only. Artifact 1 expects the dataset to be reachable.
- **Environment now pinned.** `requirements.txt` holds exact `==` versions, and the
  missing `pyarrow` was added, without which a fresh install cannot write the feature
  store's Parquet file. Built on Python 3.13.9. What remains for task 2.5 is a teammate
  actually reproducing a run from the logged config.
- **`pipeline.py` uses argparse**, against project convention. Worth replacing with a
  plain config before the tree grows.
- **Stage 4 of `pipeline.py` overlaps the feature store, and it leaks.** It fits TF-IDF
  on all 363,825 rows with no train/test split, so `tfidf_vectorizer.pkl` has seen the
  test set. Nothing trains from it today, so the leak is harmless, but two feature paths
  invite the question of which one counts. Decide whether to drop the stage.
- **Task owners are unassigned.** The source document has a "Suggested owner" blank for
  each week. Fill these in with the team.

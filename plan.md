# Flavor C — Support Ticket / Review Sentiment Classifier

ML Engineering Mini-Project (PCAM ZG412). Group submission, deadline **24 August 2026**.

Evaluation is holistic at the end, not week by week. All three final artifacts are
mandatory. The assignment is not evaluated at all without the video.

## Status snapshot

| Week | Milestone | Focus | State |
|---|---|---|---|
| 1 | M2 | Data ingestion, validation, features | 10 of 10 tasks done |
| 2 | M3 | Model training, experiment tracking | Not started |
| 3 | M4 | Packaging, deployment | Not started |
| 4 | M5 | Monitoring, drift, retraining | Not started |

Week 1 code lives in `src/` and was verified end to end on 5 Aug 2026 via
`src/pipeline.py`, taking about 23 seconds on the full 568k-row dataset.

Reproduce the run with:

```bash
./venv/bin/python src/pipeline.py --input data/raw/Reviews.csv --text-col Text --rating-col Score
```

## Week 1 (M2) — Data ingestion, validation & feature pipeline

- [x] **1.1 Select dataset.** Amazon Fine Food Reviews, 568,454 rows at `data/raw/Reviews.csv`. The `Text` and `Score` columns match what the code expects. Still needs registering in the group spreadsheet.
- [x] **1.2 Data ingestion.** `src/ingestion.py` loads raw CSV into a DataFrame and logs row counts. Scripted, no manual steps.
- [x] **1.3 Data validation.** `src/validation.py` checks nulls, duplicates, empty text, rating range, class balance, and review-length distribution. Raises `DataValidationError` on critical failures, which halts the pipeline.
- [x] **1.4 Label creation.** `label_sentiment()` in `src/cleaning.py` drops 3-star rows and maps rating >= 4 to 1 (Positive), <= 2 to 0 (Negative).
- [x] **1.5 Text cleaning, two versions.** `heavy_clean()` removes stopwords for TF-IDF; `light_clean()` keeps sentence structure for DistilBERT.
- [x] **1.6 Feature pipeline A.** `build_tfidf_features()` fits or reuses a TF-IDF vectorizer. Accepts an existing vectorizer so Week 4 retraining reuses the same vocabulary.
- [x] **1.7 Feature pipeline B.** `build_bert_tokenized_features()` returns `input_ids` and `attention_mask` for DistilBERT.
- [x] **1.8 Dataset versioning.** DVC 3.67.1 initialized on 5 Aug 2026. Both datasets are tracked: `data/raw/Reviews.csv` (301 MB) and `data/processed/cleaned_reviews.csv` (422 MB). Git holds only the two `.dvc` pointer files, which carry an md5 and a byte size. The data itself lives in `.dvc/cache` (690 MB) and is ignored by git. The `.gitignore` entries for those two CSVs were removed, because DVC refuses to track files git already ignores; DVC then wrote its own ignore rules into `data/raw/` and `data/processed/`.
- [x] **1.9 Repo setup.** Git repo initialized with `src/`, `data/raw/`, `data/processed/`. `.gitignore` excludes CSVs, pickles, `venv/`, and `mlruns/`.
- [x] **1.10 Validation report.** Written to `reports/validation_report.md`: row accounting, class balance, data-quality findings, and the design decisions behind them. Numbers come from a verified full run on 5 Aug 2026. Drops in this project are sequential, so the per-step counts there supersede the raw figures in `validation_report.json`.

## Week 2 (M3) — Model training & experiment tracking

- [ ] **2.0 Handle class imbalance.** Not in the source document, but the data forces it. The cleaned set is roughly 84% positive, a 5.4-to-1 split. A model that always predicts Positive scores about 84% accuracy while being useless. Decide on class weights, resampling, or both, and report macro F1 rather than accuracy.
- [ ] **2.1 Baseline model.** Logistic Regression (or SVM) on the TF-IDF features from 1.6. Use `class_weight="balanced"` given 2.0.
- [ ] **2.2 Advanced model.** Fine-tune DistilBERT on light-cleaned text. GPU via Colab or local.
- [ ] **2.3 Experiment tracking.** Log both runs to MLflow: parameters, metrics, artifacts, code version.
- [ ] **2.4 Model comparison.** Compare accuracy, F1, and precision-recall side by side. Document why the winner wins.
- [ ] **2.5 Reproducibility check.** A teammate must reproduce the winning run from the logged config alone.

## Week 3 (M4) — Packaging & deployment

- [ ] **3.1 Model serialization.** Pickle for sklearn, saved weights for DistilBERT.
- [ ] **3.2 REST API.** FastAPI endpoint taking text, returning predicted label plus confidence.
- [ ] **3.3 Input validation.** Handle empty strings, non-text input, and oversized text gracefully.
- [ ] **3.4 Containerize.** Write a Dockerfile, then build and test the image locally.
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

- **Filename case mismatch.** The dataset is `data/raw/Reviews.csv`, but the
  `__main__` test blocks in all four modules use `data/raw/reviews.csv`. macOS
  ignores case, so those self-tests pass locally and would fail on Linux,
  including Docker in Week 3. `pipeline.py` is unaffected, since it takes the path
  as an argument.
- **No pinned environment.** `requirements.txt` uses open `>=` ranges, and the
  installed versions have already drifted far past them. `transformers` resolved to
  5.14.1 against a `>=4.35` floor, a major version jump, and `torch` to 2.13 against
  `>=2.0`. The DistilBERT code still imports fine today, but task 2.5 grades whether
  a teammate can reproduce your run, so pin exact versions now.
- **`src/pipeline.py` uses argparse**, against project convention. Worth replacing
  with a plain config or environment variables before the tree grows.
- **Task owners are unassigned.** The source document has a "Suggested owner" blank
  for each week. Fill these in with the team.

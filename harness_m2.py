"""
M2 Harness — step through the Week 1 pipeline, one stage at a time
--------------------------------------------------------------------
Calls the same functions as src/pipeline.py, but prints the INPUT and
the OUTPUT of every stage and pauses in between, so you can read what
each step produced before moving on.

Run from the repo root:
    ./venv/bin/python src/harness_m2.py

Press Enter to advance, Ctrl-C to stop and keep whatever is on screen.
Pauses are skipped when stdin is not a terminal, so piping to a file
or a log still works.

This harness only inspects. It does not write to data/processed —
run src/pipeline.py for that.
"""

import json
import logging
import sys
from pathlib import Path

# The pipeline stages live in their own folders, so make them importable.
ROOT = Path(__file__).parent
sys.path[:0] = [str(ROOT / "validation"), str(ROOT / "features")]

from ingestion import ingest_raw_data  # noqa: E402
from validate_data import validate_data  # noqa: E402
from cleaning import clean_and_label  # noqa: E402
from feature_engineering import build_tfidf_features, build_bert_tokenized_features  # noqa: E402

RAW_CSV = "data/raw/Reviews.csv"
TEXT_COL = "Text"
RATING_COL = "Score"
PREVIEW_ROWS = 5
SNIPPET = 88

# The pipeline modules log each step at INFO. The harness prints richer
# versions of the same numbers, so silence them to keep the output readable.
logging.disable(logging.INFO)


def stage(num, title):
    print(f"\n{'=' * 72}")
    print(f"  STAGE {num} — {title}")
    print("=" * 72, flush=True)


def show(label, lines):
    print(f"\n  {label}")
    for line in lines:
        print(f"    {line}")
    # Flush so stage output stays in order when piped, since library
    # warnings go to unbuffered stderr.
    sys.stdout.flush()


def pause():
    if sys.stdin.isatty():
        input("\n  ── Enter to continue, Ctrl-C to stop here ──")


def snip(text):
    text = " ".join(str(text).split())
    return text if len(text) <= SNIPPET else text[:SNIPPET] + " ..."


def main():
    print("\nM2 HARNESS — data ingestion, validation & feature pipeline")

    stage(1, "INGESTION")
    show("INPUT", [
        f"file   : {RAW_CSV} ({Path(RAW_CSV).stat().st_size / 1e6:.0f} MB on disk)",
        f"columns: {TEXT_COL!r} as text, {RATING_COL!r} as rating",
    ])
    df_raw = ingest_raw_data(RAW_CSV, text_col=TEXT_COL, rating_col=RATING_COL)
    show("OUTPUT", [
        f"{df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols, renamed to {list(df_raw.columns)}",
        "",
        *[f"rating={r.rating}  {snip(r.review_text)}"
          for r in df_raw.head(PREVIEW_ROWS).itertuples()],
    ])
    pause()

    stage(2, "VALIDATION")
    show("INPUT", [f"{len(df_raw):,} ingested rows",
                   "critical rules halt the pipeline; the rest are reported"])
    report = validate_data(df_raw)
    show("OUTPUT", json.dumps(report, indent=2).splitlines())
    pause()

    stage(3, "CLEANING & LABELING")
    show("INPUT", [f"{len(df_raw):,} validated rows",
                   f"example: {snip(df_raw.iloc[0].review_text)}"])
    df_clean = clean_and_label(df_raw)
    labels = df_clean["label"].value_counts()
    first = df_clean.iloc[0]
    show("OUTPUT", [
        f"{len(df_raw):,} -> {len(df_clean):,} rows "
        f"({len(df_raw) - len(df_clean):,} dropped: null, empty, duplicate, 3-star)",
        f"labels: {labels.get(1, 0):,} positive / {labels.get(0, 0):,} negative",
        "",
        "one review, both cleaning versions:",
        f"  raw   {snip(first.review_text)}",
        f"  heavy {snip(first.text_heavy_clean)}",
        "        ^ stopwords removed, feeds TF-IDF",
        f"  light {snip(first.text_light_clean)}",
        "        ^ structure kept, feeds DistilBERT",
    ])
    pause()

    stage(4, "FEATURE ENGINEERING A — TF-IDF")
    show("INPUT", [f"{len(df_clean):,} heavy-cleaned texts"])
    matrix, vectorizer = build_tfidf_features(df_clean["text_heavy_clean"])
    terms = vectorizer.get_feature_names_out()
    row0 = matrix[0]
    heaviest = sorted(zip(row0.indices, row0.data), key=lambda pair: -pair[1])[:8]
    show("OUTPUT", [
        f"sparse matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]:,} features",
        f"vocabulary   : {len(terms):,} terms",
        f"row 0 has {row0.nnz} nonzero features out of {matrix.shape[1]:,}",
        "row 0 top weights: " + ", ".join(f"{terms[i]}={w:.2f}" for i, w in heaviest),
    ])
    pause()

    stage(5, "FEATURE ENGINEERING B — DistilBERT tokenizer")
    sample = df_clean["text_light_clean"].head(PREVIEW_ROWS)
    show("INPUT", [
        f"{len(sample)} light-cleaned texts",
        f"a subset on purpose — tokenizing all {len(df_clean):,} rows is slow here",
    ])
    encoded = build_bert_tokenized_features(sample)
    ids = encoded["input_ids"]
    show("OUTPUT", [
        f"input_ids     : {tuple(ids.shape)}  (rows x padded length)",
        f"attention_mask: {tuple(encoded['attention_mask'].shape)}",
        f"row 0 real tokens: {int(encoded['attention_mask'][0].sum())} of {ids.shape[1]} "
        f"(rest is padding)",
        f"row 0 tokens: {' '.join(encoded.tokens(0)[:14])} ...",
        f"row 0 ids   : {ids[0][:14].tolist()} ...",
    ])

    print(f"\n{'=' * 72}")
    print("  M2 COMPLETE — nothing written to disk")
    print("=" * 72)
    print("\n  To produce the real artifacts, run:")
    print(f"    ./venv/bin/python src/pipeline.py --input {RAW_CSV} "
          f"--text-col {TEXT_COL} --rating-col {RATING_COL}\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Stopped. Output above is what the pipeline produced up to here.\n")

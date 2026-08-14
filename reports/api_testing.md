# API testing

Task 3.5. Every request and response below was captured from the running API serving
`distilbert-20k`. Nothing here is invented.

Start the API first:

```bash
venv/bin/uvicorn serving.api:app --port 8000
```

An interactive version of all of this is at `http://localhost:8000/docs`.

## Health check

```bash
curl localhost:8000/health
```

```json
{"status": "ok", "model": "distilbert-20k"}
```

## Successful predictions

### Clearly positive

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This coffee is wonderful, smooth and rich."}'
```

```json
{"label": "positive", "confidence": 0.9983,
 "cleaned_text": "This coffee is wonderful, smooth and rich."}
```

### Clearly negative

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Arrived stale, tasted awful, waste of money."}'
```

```json
{"label": "negative", "confidence": 0.9879,
 "cleaned_text": "Arrived stale, tasted awful, waste of money."}
```

### Negation, which needs word order

A model that only counted words would see "good" and answer positive. This one reads the
order and gets it right.

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "not good at all, very disappointed"}'
```

```json
{"label": "negative", "confidence": 0.9573,
 "cleaned_text": "not good at all, very disappointed"}
```

### HTML and extra whitespace are cleaned

The `cleaned_text` field shows exactly what the model read, which makes the cleaning step
visible rather than hidden.

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "<br />The packaging   was   fine.<br /> Tastes great!"}'
```

```json
{"label": "positive", "confidence": 0.9968,
 "cleaned_text": "The packaging was fine. Tastes great!"}
```

### A mixed review, where confidence drops

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Tastes good. Overpriced though."}'
```

```json
{"label": "positive", "confidence": 0.9369,
 "cleaned_text": "Tastes good. Overpriced though."}
```

## Rejected requests

All return HTTP **422**. FastAPI wraps the reason in a `detail` array; the `msg` field is
quoted below for brevity.

| Input | HTTP | Message |
|---|---|---|
| `{"text": ""}` | 422 | text must contain at least one letter or digit |
| `{"text": "     "}` | 422 | text must contain at least one letter or digit |
| `{"text": "!!!???"}` | 422 | text must contain at least one letter or digit |
| `{"text": "🙂🙂🙂"}` | 422 | text must contain at least one letter or digit |
| `{"text": "<br /><br />"}` | 422 | text must contain at least one letter or digit |
| `{"text": 12345}` | 422 | Input should be a valid string |
| `{"text": null}` | 422 | Input should be a valid string |
| `{}` | 422 | Field required |
| `{"review": "great"}` | 422 | Field required |
| 6,000 characters | 422 | String should have at most 5000 characters |

Full response shape for a rejection:

```bash
curl -i -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "!!!???"}'
```

```json
{"detail": [{"type": "value_error",
             "loc": ["body", "text"],
             "msg": "Value error, text must contain at least one letter or digit"}]}
```

The punctuation case is the one that needed a real fix. `light_clean()` deliberately keeps
punctuation for DistilBERT, so `"!!!???"` first passed an emptiness check and came back as
`positive` with 0.63 confidence — a confident-looking answer about nothing. The rule now
requires at least one letter or digit.

## Edge cases that are accepted on purpose

| Input | Result | Reasoning |
|---|---|---|
| `{"text": "awful"}` | negative, 0.8825 | One word is still a review. |
| `{"text": "5"}` | positive, 0.5975 | A digit is allowed. Confidence near 0.5 correctly signals that the model has almost nothing to go on. |
| 4,000 characters | positive, 0.9792 | Under the 5,000 limit, so accepted. Text beyond 128 tokens is truncated. |

## Latency

Measured from the `latency_ms` field the API writes to `logs/predictions.jsonl`.

| | Milliseconds |
|---|---|
| First request after startup | 365.6 |
| Median thereafter | 16.9 |
| 95th percentile | 25.6 |

The first request is slow because PyTorch initialises lazily. Send one throwaway request
after startup if you need the first real user request to be fast.

## What is not covered

- No load or concurrency testing. Latency figures come from sequential requests.
- Only the DistilBERT model is exposed. The three Logistic Regression models are not served.
- No authentication. The API is open, which is fine locally and not fine in production.

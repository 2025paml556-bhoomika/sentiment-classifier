# Task 3.4 — container for the prediction API.
#
# Build and run from the repo root:
#     docker build -t sentiment-api .
#     docker run -p 8000:8000 sentiment-api
#
# The model weights are DVC-tracked, so they must exist on disk before
# building. Run `dvc pull` first if model_store/distilbert-50k is empty.
# They are copied in rather than pulled during the build, because this
# project has no DVC remote yet.

FROM python:3.13-slim

WORKDIR /app

# CPU-only torch. The default wheel bundles CUDA libraries that add gigabytes
# and cannot be used here, since containers have no access to the Apple GPU.
COPY serving/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Only what the API needs at runtime.
COPY serving/ ./serving/
COPY data_pipeline/cleaning.py ./data_pipeline/cleaning.py
COPY ui/index.html ./ui/index.html
COPY model_store/distilbert-50k/ ./model_store/distilbert-50k/

# Predictions are logged to logs/predictions.jsonl. Mount a volume over this
# to keep them after the container stops:
#     docker run -p 8000:8000 -v "$PWD/logs:/app/logs" sentiment-api
RUN mkdir -p logs

EXPOSE 8000

# No --reload: that watches files for changes and belongs in development only.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

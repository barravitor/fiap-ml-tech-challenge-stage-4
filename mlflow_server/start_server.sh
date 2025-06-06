#!/bin/bash

echo "[MLFLOW] Starting MLflow server..."

mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

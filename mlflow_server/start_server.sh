#!/bin/bash

echo "[MLFLOW] Iniciando servidor MLflow..."

mlflow server \
  --backend-store-uri file:/app/mlruns \
  --default-artifact-root file:/app/mlruns \
  --host 0.0.0.0 \
  --port 5000

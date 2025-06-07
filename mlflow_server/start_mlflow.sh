#!/bin/bash

echo "⏳ Aguardando MinIO iniciar..."
until curl -s http://localhost:9000/minio/health/ready | grep -q "OK"; do
  sleep 1
done

echo "✅ MinIO está pronto. Iniciando MLflow..."

DB_PATH=$(pwd)/data/mlflow.db
ARTIFACT_PATH=$(pwd)/data/mlartifacts

# Cria banco se não existir
touch $DB_PATH

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root s3://mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
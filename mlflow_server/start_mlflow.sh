#!/bin/bash

DB_PATH=$(pwd)/data/mlflow.db
ARTIFACT_PATH=$(pwd)/data/mlartifacts

mkdir -p "$(dirname "$DB_PATH")"
mkdir -p "$ARTIFACT_PATH"

echo "Banco de dados vai em: $DB_PATH"
echo "Artefatos vão em: $ARTIFACT_PATH"

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root file:///$ARTIFACT_PATH \
  --host 0.0.0.0 \
  --port 5000
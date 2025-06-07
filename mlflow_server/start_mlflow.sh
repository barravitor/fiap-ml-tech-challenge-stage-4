#!/bin/bash

DB_PATH=$(pwd)/data/mlflow.db
ARTIFACT_PATH=$(pwd)/data/mlartifacts

# Cria banco se não existir
touch $DB_PATH

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root s3://mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
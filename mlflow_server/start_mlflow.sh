#!/bin/bash

mkdir -p data mlartifacts

DB_PATH=$(pwd)/data/mlflow.db
ARTIFACT_PATH=$(pwd)/mlartifacts

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root file:///$ARTIFACT_PATH \
  --host 0.0.0.0 \
  --port 5000
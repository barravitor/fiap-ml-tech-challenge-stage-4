#!/bin/bash

# Aguarda até que o NGINX esteja escutando na porta 80
echo "Aguardando o NGINX ficar disponível..."
while ! nc -z localhost 80; do
  sleep 1
done

echo "NGINX disponível. Iniciando MLflow..."

DB_PATH=$(pwd)/data/mlflow.db
ARTIFACT_PATH=$(pwd)/data/mlartifacts

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root file:///$ARTIFACT_PATH \
  --host 0.0.0.0 \
  --port 5000
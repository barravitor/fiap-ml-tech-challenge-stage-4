#!/bin/bash

# Cria o arquivo de credenciais a partir da variável de ambiente
echo "$GOOGLE_CREDENTIALS_JSON" > ./gcp-storage-service-account.json

# (opcional) Verifica se o arquivo foi criado
ls -l "$GOOGLE_APPLICATION_CREDENTIALS"

mlflow server \
  --backend-store-uri sqlite:///$MLFLOW_DB_PATH \
  --default-artifact-root $MLFLOW_ARTIFACT_PATH \
  --host 0.0.0.0 \
  --port 5000
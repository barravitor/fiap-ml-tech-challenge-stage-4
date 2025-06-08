#!/bin/bash

printf '%b' "$GOOGLE_CREDENTIALS_JSON" > ./gcp-storage-service-account.json

ls -l "$GOOGLE_APPLICATION_CREDENTIALS"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
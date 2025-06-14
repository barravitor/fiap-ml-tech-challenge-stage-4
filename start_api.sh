# Load .env
set -a
source .env
set +a

newrelic-admin run-program uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir api --reload-dir shared
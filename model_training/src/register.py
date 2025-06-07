import sys
import mlflow
from mlflow.tracking import MlflowClient

TICKER=''
EXPETIMENTAL_RUN_ID=''

if not TICKER or not EXPETIMENTAL_RUN_ID:
    print("Erro: TICKER or EXPETIMENTAL_RUN_ID not found.")
    sys.exit(1)

MODEL_NAME=f"LSTM-{TICKER}"

result = mlflow.register_model(
    model_uri=f"runs:/{EXPETIMENTAL_RUN_ID}/model",
    name=MODEL_NAME
)

print(f"Model registered: {result.version}")

client = MlflowClient()
client.set_registered_model_alias(
    name=MODEL_NAME,
    version=result.version,
    alias="Production",
)
# client.set_model_version_tag(MODEL_NAME, result.version, 'stage', 'Production')
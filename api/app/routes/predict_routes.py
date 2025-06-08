# app/routes/predict_routes.py

import os
import io
import torch
from typing import List
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from shared.utils import preprocess_dataframe
from ..schemas.predict_schemas import PredictBodySchema, PredictResponseSchema, DataItem
from shared.config import FEATURES_COLS_DEFAULT, MINIMUM_NUMBER_OF_DATA, NUMBER_OF_DAYS_TO_FORECAST
from shared.forecast_service import ForecastService
from google.auth import default

predict_router = APIRouter()

FEATURES_COLS_DEFAULT_AND_DATE = ["Date"] + FEATURES_COLS_DEFAULT

@predict_router.post("/stocks",
    response_model=PredictResponseSchema,
    responses={
        200: {
            "description": "Return a list with forecast",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [31.12],
                        "data": [
                            {
                                "Close": 31.11992073059082,
                                "High": 32.03924146833904,
                                "Low": 31.5513799266813,
                                "Open": 31.844098881908675,
                                "Volume": 20776200.0,
                                "SMA_10": 31.63815269470215,
                                "RSI_14": 31.14500599404569,
                                "Volume_MA_10": 38548940.0,
                                "Volume_Relative": 0.5389564537961355
                            }
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Bad Request.",
            "content": {
                "application/json": {
                    "examples": {
                        "error_to_read_file": {
                            "summary": "Invalid CSV format",
                            "value": {
                                "detail": "Error to read CSV data"
                            }
                        },
                        "invalid_csv_rows": {
                            "summary": "Invalid CSV format",
                            "value": {
                                "detail": f"CSV report must have at least {MINIMUM_NUMBER_OF_DATA} rows"
                            }
                        },
                        "invalid_csv_columns": {
                            "summary": "Invalid CSV format",
                            "value": {
                                "detail": f"CSV data must be {len(FEATURES_COLS_DEFAULT_AND_DATE)} columns"
                            }
                        },
                    }
                }
            }
        },
        500: {
            "description": "Bad Request.",
            "content": {
                "application/json": {
                    "examples": {
                        "internal_error": {
                            "summary": "Internal error server",
                            "value": {
                                "detail": "Internal error server"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def predict(request: PredictBodySchema):
    try:
        df_raw = pd.read_csv(io.StringIO(request.csv), skiprows=0)
    except Exception:
        raise HTTPException(status_code=400, detail="Error to read CSV data")

    if len(df_raw) < MINIMUM_NUMBER_OF_DATA:
        raise HTTPException(status_code=400, detail=f"CSV report must have at least {MINIMUM_NUMBER_OF_DATA} rows")

    if df_raw.shape[1] < len(FEATURES_COLS_DEFAULT_AND_DATE):
        raise HTTPException(status_code=400, detail=f"CSV data must be {len(FEATURES_COLS_DEFAULT_AND_DATE)} columns")
    

    creds, proj = default()
    print("[DEBUG] GCP project:", proj)


    print("[DEBUG] GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    client = MlflowClient()
    MODEL_NAME = f"LSTM-{request.ticker.upper()}"

    print('MODEL_NAME', MODEL_NAME)

    prod_version = client.get_model_version_by_alias(MODEL_NAME, 'Production')

    print('prod_version', prod_version)

    mlflow_model_uri = f"models:/{MODEL_NAME}/{prod_version.version}"

    print('mlflow_model_uri', mlflow_model_uri)

    artifacts = client.list_artifacts(prod_version.run_id)
    print([a.path for a in artifacts])
    print("MLFLOW_TRACKING_URI:", mlflow.get_tracking_uri())

    mlflow_model = mlflow.pytorch.load_model(mlflow_model_uri, map_location=torch.device('cpu'))

    print('mlflow_model', mlflow_model)

    alias_info = client.get_model_version_by_alias(MODEL_NAME, 'Production')

    forecastService = ForecastService(request.ticker)

    df_treated = preprocess_dataframe(df_raw)
    df_with_prediction, predictions = forecastService.predict(
        df_treated=df_treated,
        number_of_forecast=NUMBER_OF_DAYS_TO_FORECAST,
        run_id=alias_info.run_id,
        model_version=alias_info.version,
    )

    predictions = [round(float(val), 3) for val in predictions]

    df_slice = df_with_prediction[-NUMBER_OF_DAYS_TO_FORECAST:]
    data_items: List[DataItem] = [DataItem(**row) for row in df_slice.to_dict(orient="records")]

    return PredictResponseSchema(predictions=predictions, data=data_items)
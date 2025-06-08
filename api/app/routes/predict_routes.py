# app/routes/predict_routes.py

import io
from typing import List
import pandas as pd
from mlflow.tracking import MlflowClient
from fastapi import APIRouter, HTTPException

from shared.utils import preprocess_dataframe, load_mlflow_model
from ..schemas.predict_schemas import PredictBodySchema, PredictResponseSchema, DataItem
from shared.config import (FEATURES_COLS_DEFAULT, MINIMUM_NUMBER_OF_DATA, NUMBER_OF_DAYS_TO_FORECAST)
from shared.forecast_service import ForecastService

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
def predict(request: PredictBodySchema):
    try:
        df_raw = pd.read_csv(io.StringIO(request.csv), skiprows=0)
    except Exception:
        raise HTTPException(status_code=400, detail="Error to read CSV data")

    if len(df_raw) < MINIMUM_NUMBER_OF_DATA:
        raise HTTPException(status_code=400, detail=f"CSV report must have at least {MINIMUM_NUMBER_OF_DATA} rows")

    if df_raw.shape[1] < len(FEATURES_COLS_DEFAULT_AND_DATE):
        raise HTTPException(status_code=400, detail=f"CSV data must be {len(FEATURES_COLS_DEFAULT_AND_DATE)} columns")

    client = MlflowClient()
    model_name = f"LSTM-{request.ticker.upper()}"

    prod_version = client.get_model_version_by_alias(model_name, 'Production')
    model = load_mlflow_model(model_name, prod_version.version)

    df_treated = preprocess_dataframe(df_raw)

    forecastService = ForecastService()

    df_with_prediction, predictions = forecastService.forecast_n_steps(
        model=model,
        df_treated=df_treated,
        number_of_forecast=NUMBER_OF_DAYS_TO_FORECAST
    )

    df_slice = df_with_prediction[-NUMBER_OF_DAYS_TO_FORECAST:]
    data_items: List[DataItem] = [DataItem(**row) for row in df_slice.to_dict(orient="records")]

    return PredictResponseSchema(predictions=[round(float(val), 3) for val in predictions], data=data_items)
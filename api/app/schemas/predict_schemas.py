# app/schemas/predict_schemas.py

from typing import List
from pydantic import BaseModel, field_validator
from shared.config import LIST_OF_ALLOWED_TICKETS

class PredictBodySchema(BaseModel):
    ticker: str
    csv: str

    @field_validator('ticker')
    @classmethod
    def check_ticker(cls, v):
        if v not in LIST_OF_ALLOWED_TICKETS:
            allowed = ', '.join(LIST_OF_ALLOWED_TICKETS)
            raise ValueError(f'Ticker "{v}" is not allowed. Allowed values: [{allowed}]')
        return v

    @field_validator('csv')
    @classmethod
    def check_csv_contains_all_columns(cls, v):
        missing = [col for col in ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'] if col not in v]
        if missing:
            raise ValueError(f"CSV está faltando as colunas: {', '.join(missing)}")
        return v

    class Config:
        from_attributes = True

class DataItem(BaseModel):
    Close: float
    # High: float
    # Low: float
    # Open: float
    Volume: float
    SMA_10: float
    RSI_14: float
    Volume_MA_10: float
    Volume_Relative: float

class PredictResponseSchema(BaseModel):
    predictions: List[float]
    data: List[DataItem]

    model_config = {
        "from_attributes": True
    }
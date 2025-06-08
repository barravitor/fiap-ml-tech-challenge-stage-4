# shared/utils.py

import os
import joblib
import mlflow
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlflow.models.signature import infer_signature

from shared.lstm_model import LSTMModel
from .config import (
    SMA_WINDOW, RSI_WINDOW, FEATURES_COLS, DEVICE
)

def SMA(data: pd.Series, window: int = 10) -> pd.Series:
    return data.rolling(window=window, min_periods=window).mean()

def RSI(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_dataframe(ticker: str):
    return pd.read_csv(f"data/{ticker.replace('.', '_')}_historical.csv", skiprows=0)

def create_folder(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def preprocess_dataframe(df: pd.Series):
    df = df.rename(columns={ "Price": "Date" })
    df = df.iloc[2:]
    df = df.set_index("Date")

    df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')

    valid_cols = [col for col in FEATURES_COLS if col in df.columns]

    for col in valid_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=valid_cols, inplace=True)

    df = df[df['Volume'] > 0] # Negative Trading Volume Remover

    # Creates new technical indicators
    if f"SMA_{SMA_WINDOW}" in FEATURES_COLS:
        df[f'SMA_{SMA_WINDOW}'] = SMA(df['Close'], window=SMA_WINDOW)

    if f"RSI_{RSI_WINDOW}" in FEATURES_COLS:
        df[f'RSI_{RSI_WINDOW}'] = RSI(df['Close'], window=RSI_WINDOW)

    if f"Volume_MA_{SMA_WINDOW}" in FEATURES_COLS:
        df[f'Volume_MA_{SMA_WINDOW}'] = SMA(df['Volume'], window=SMA_WINDOW)

    if f"Volume_Relative" in FEATURES_COLS:
        df['Volume_Relative'] = df['Volume'] / df[f'Volume_MA_{SMA_WINDOW}']

    df.dropna(inplace=True)

    df_filtered = {col: df[col] for col in FEATURES_COLS if col in df.columns}

    df = pd.DataFrame(df_filtered)

    return df

def save_scaler(ticker: str, scaler_name: str, scaler: MinMaxScaler):
    model_dir = f"models/{ticker.replace('.', '_')}/0/scalers"
    create_folder(model_dir)
    joblib.dump(scaler, f"{model_dir}/{scaler_name}.pkl")
    mlflow.log_artifact(f"{model_dir}/{scaler_name}.pkl", artifact_path="scalers")

def save_model(ticker: str, model,  X_train: np.ndarray):
    """
    Saves the trained model using MLflow and as a local .pth file.
    """
    model_dir = f"models/{ticker.replace('.', '_')}/0/model/data"
    create_folder(model_dir)

    # Create input example and infer signature
    input_example = X_train[:1]
    input_tensor = torch.tensor(input_example, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()

    signature = infer_signature(input_example, output)

    # Save with MLflow
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        pip_requirements="requirements.txt",
        signature=signature,
    )

    # Save locally
    torch.save(model.state_dict(), f"{model_dir}/model.pth")
    print(f"[INFO] Model saved to: {model_dir}/model.pth")

def load_mlflow_model(model_name: str, version: int = None) -> LSTMModel:
    mlflow_model_uri = f"models:/{model_name}/{version}"
    mlflow_model: LSTMModel = mlflow.pytorch.load_model(mlflow_model_uri, map_location=DEVICE)
    return mlflow_model
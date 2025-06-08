# shared/utils.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow
import torch
from mlflow.models.signature import infer_signature
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
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f"{model_dir}/{scaler_name}.pkl")
    mlflow.log_artifact(f"{model_dir}/{scaler_name}.pkl", artifact_path="scalers")

def load_scaler(ticker: str, scaler_name: str, run_id=None, model_version: int = 1):
    scale_path = f"models/{ticker.replace('.', '_')}"

    if run_id is not None:
        if os.path.exists(f"{scale_path}/{model_version}/scalers/{scaler_name}.pkl"):
            scale_path = f"{scale_path}/{model_version}/scalers/{scaler_name}.pkl"
        else:
            print("Download scaler...")
            scale_path = mlflow.artifacts.download_artifacts(
                artifact_path=f"scalers/{scaler_name}.pkl",
                run_id=run_id,
                dst_path=f"{scale_path}/{model_version}"
            )
    else:
        scale_path = f"models/{ticker.replace('.', '_')}/0/scalers/{scaler_name}.pkl"

    return joblib.load(scale_path)

def save_model(ticker: str, model,  X_train: np.ndarray):
    """
    Saves the trained model using MLflow and as a local .pth file.
    """
    model_dir = f"models/{ticker.replace('.', '_')}/0/model/data"
    os.makedirs(model_dir, exist_ok=True)

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

def load_model(ticker: str, run_id: str = None, model_version: int = 1):
    """
    Saves the trained model using MLflow and as a local .pth file.
    """

    model_path = f"models/{ticker.replace('.', '_')}"

    if run_id is not None:
        if os.path.exists(f"{model_path}/{model_version}/model/data/model.pth"):
            model_path = f"{model_path}/{model_version}/model/data/model.pth"
        else:
            print("Download model...")
            model_path = mlflow.artifacts.download_artifacts(
                artifact_path=f"model/data/model.pth",
                run_id=run_id,
                dst_path=f"{model_path}/{model_version}"
            )
    else: 
        model_path = f"models/{ticker.replace('.', '_')}/0/model/data/model.pth"

    model = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if run_id is not None:
        return model.state_dict()

    return model
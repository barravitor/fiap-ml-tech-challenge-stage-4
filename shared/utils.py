# shared/utils.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import SMA_WINDOW, RSI_WINDOW, FEATURES_COLS

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
    print(df.tail())
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
    model_dir = f"models/{ticker.replace('.', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f"{model_dir}/{scaler_name}.pkl")

def get_scaler(ticker: str, scaler_name: str):
    model_dir = f"models/{ticker.replace('.', '_')}"
    return joblib.load(f"{model_dir}/{scaler_name}.pkl")
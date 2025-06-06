# model_training/src/data_loader.py

import yfinance as yf
import os
import pandas as pd

def extract_stock_data(ticker: str, start: str = "2018-01-01", end: str = "2024-07-20") -> pd.DataFrame:
    """
    Downloads the historical data of a stock and saves it to the /data folder.

    Args:
        ticker (str): Stock symbol (e.g., 'PETR4.SA').
        start (str): Start date in the format 'YYYY-MM-DD'.
        end (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        DataFrame containing the downloaded data.
    """
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker.replace('.', '_')}_historical.csv"

    if os.path.exists(filename):
        print(f"[INFO] File already exists: {filename}")
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        return df

    print(f"[INFO] Downloading data from {ticker}...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    data.to_csv(filename)
    print(f"[INFO] Data saved in: {filename}")
    return data
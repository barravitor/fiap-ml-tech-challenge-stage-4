# shared/forecast_service.py

import torch
import mlflow
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from mlflow.data import from_pandas

from .config import (
    DEVICE, FEATURES_COLS, TARGET, WINDOW_SIZE,
    LEARNING_RATE, NUM_EPOCHS, SMA_WINDOW, RSI_WINDOW
)
from .lstm_model import LSTMModel
from .utils import SMA, RSI, save_scaler, save_model

class ForecastService:
    def forecast_n_steps(self, model: LSTMModel, df_treated: pd.DataFrame, number_of_forecast: int = 1) -> tuple[pd.DataFrame, list[float]]:
        """
        Predicts future values and appends them to the treated DataFrame.

        Parameters:
            df_treated (pd.DataFrame): Preprocessed DataFrame containing features.
            number_of_forecast (int): Number of future steps to predict.

        Returns:
            tuple: (Updated DataFrame, list of predicted values)
        """
        predictions = []
        for _ in range(number_of_forecast):
            next_value = model.predict(df_treated)
            predictions.append(next_value)

            # Append prediction to the DataFrame
            new_row = df_treated.iloc[-1].copy()
            new_row[TARGET] = next_value
            df_treated = pd.concat([df_treated, new_row.to_frame().T], ignore_index=True)

            # Recalculate indicators if needed
            start_idx = max(0, len(df_treated) - max(SMA_WINDOW, RSI_WINDOW) - 1)
            subset = df_treated.iloc[start_idx:].copy()

            if f"SMA_{SMA_WINDOW}" in FEATURES_COLS:
                subset[f'SMA_{SMA_WINDOW}'] = SMA(subset['Close'], window=SMA_WINDOW)

            if f"RSI_{RSI_WINDOW}" in FEATURES_COLS:
                subset[f'RSI_{RSI_WINDOW}'] = RSI(subset['Close'], window=RSI_WINDOW)

            # Update last row with calculated indicators
            for col in [f'SMA_{SMA_WINDOW}', f'RSI_{RSI_WINDOW}']:
                if col in subset.columns:
                    df_treated.at[df_treated.index[-1], col] = subset[col].iloc[-1]

        return df_treated, predictions

    def generate_scaler(self, ticker: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
        """
        Scales the dataset and generates input-output sequences for training.

        Parameters:
            df (pd.DataFrame): Input DataFrame with features and target.

        Returns:
            tuple: (X, y, scaler_X, scaler_y)
        """
        X_raw = df[FEATURES_COLS].copy()
        y_raw = df[[TARGET]].copy()

        # Fit scalers
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_raw.values)
        y_scaled = scaler_y.fit_transform(y_raw)

        # Save scalers
        save_scaler(ticker, 'scaler_X', scaler_X)
        save_scaler(ticker, 'scaler_y', scaler_y)

        # Generate sliding window sequences
        X, y = [], []
        for i in range(len(X_scaled) - WINDOW_SIZE):
            X.append(X_scaled[i:i + WINDOW_SIZE])
            y.append(y_scaled[i + WINDOW_SIZE][0])

        return np.array(X), np.array(y), scaler_X, scaler_y

    def train(self, model: LSTMModel, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> LSTMModel:
        """
        Trains the LSTM model.

        Returns:
            Trained LSTMModel instance.
        """
        print(f"[INFO] Using device: {DEVICE}")

        # Log inputs to MLflow
        mlflow.log_input(from_pandas(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)), name="X_train"), context="training")
        mlflow.log_input(from_pandas(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)), name="X_test"), context="testing")

        # Prepare tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # DataLoader
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

        # Model training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_tensor.to(DEVICE))
                val_loss = criterion(val_pred, y_test_tensor.to(DEVICE)).item()

            avg_train_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(f"[{epoch + 1:02d}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        save_model(ticker=model.ticker, model=model, X_train=X_train)
        return model

    def evaluate_model(self, model: LSTMModel, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Evaluates the trained model on the test set.

        Returns:
            np.ndarray: Model predictions on test set.
        """
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy().squeeze()

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)

        print("\n===== Model Evaluation =====")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2%}")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)

        return preds

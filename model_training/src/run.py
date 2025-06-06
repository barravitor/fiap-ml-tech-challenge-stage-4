# model_training/src/index.py

import mlflow
from sklearn.model_selection import train_test_split

from .data_loader import extract_stock_data
from shared.utils import load_dataframe, preprocess_dataframe
from shared.config import (
    MLFLOW_TRACKING_URI, FEATURES_COLS, INPUT_SIZE, TARGET, SMA_WINDOW,
    RSI_WINDOW, LEARNING_RATE, TEST_SIZE_SPLIT, NUM_EPOCHS, HIDDEN_SIZE,
    NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE
)
from shared.forecast_service import ForecastService


if __name__ == "__main__":
    ticker = "PETR4.SA"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Start MLflow experiment
    with mlflow.start_run(run_name=f"LSTM Training - {ticker}", description="Tech Challenge 4"):
        if mlflow.active_run():
            # Log experiment metadata
            mlflow.set_tag("ticker", ticker)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("features", FEATURES_COLS)
            mlflow.log_param("sma_window", SMA_WINDOW)
            mlflow.log_param("rsi_window", RSI_WINDOW)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("test_size_split", TEST_SIZE_SPLIT)
            mlflow.log_param("num_epochs", NUM_EPOCHS)
            mlflow.log_param("hidden_size", HIDDEN_SIZE)
            mlflow.log_param("num_layers", NUM_LAYERS)
            mlflow.log_param("output_size", OUTPUT_SIZE)
            mlflow.log_param("window_size", WINDOW_SIZE)
            mlflow.log_param("input_size", INPUT_SIZE)

            # Initialize forecasting service
            forecastService = ForecastService(ticker)

            # Load stock data and preprocess it
            extract_stock_data(ticker)
            df_raw = load_dataframe(ticker)
            df_treated = preprocess_dataframe(df_raw)

            # Scale and prepare sequences
            X, y, scaler_X, scaler_y = forecastService.generate_scaler(df_treated)

            # Train/test split (no shuffling to preserve time series order)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_SPLIT, shuffle=False)

            # Train model
            forecastService.train(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            # Evaluate model
            forecastService.evaluate_model(X_test, y_test)

            # Use the last 30 records for forecasting
            last_30 = df_treated[-30:]
            df_with_prediction, predictions = forecastService.predict(last_30)

            print([round(float(val), 3) for val in predictions])

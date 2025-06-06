# shared/predict.py

import torch
import pandas as pd
import torch.nn as nn
from .config import FEATURES_COLS, DEVICE

def predict(model: nn.Module, last_data: pd.DataFrame, scaler_X, scaler_y):
    """
    Predicts the next value in a time series using a trained LSTM model.
    
    Parameters:
        model (nn.Module): Trained LSTM model.
        last_data (pd.DataFrame): DataFrame containing the rows of processed feature data.
        scaler_X: Scaler used to normalize input features.
        scaler_y: Scaler used to normalize target values (used here to inverse-transform the prediction).
    
    Returns:
        float: The predicted next value in the original scale.
    """

    model.eval()

    # Extract and scale input data
    input_features = last_data[FEATURES_COLS].values
    input_scaled = scaler_X.transform(input_features)

    # Format as tensor: (1, seq_len, input_size)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor).cpu().numpy()

    # Inverse scale the prediction
    prediction_unscaled = scaler_y.inverse_transform(prediction_scaled)

    return prediction_unscaled[0][0]

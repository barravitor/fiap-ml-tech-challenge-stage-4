# model_training/src/model.py

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from .config import FEATURES_COLS, DEVICE, TARGET

class LSTMModel(nn.Module):
    def __init__(self, ticker: str, input_size, hidden_size: int, num_layers: int, output_size: int, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ticker = ticker
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def predict(self, input_data: DataFrame):
        self.eval()

        # Extract and scale input data
        input_features = input_data[FEATURES_COLS].values
        input_scaled = self.scaler_X.transform(input_features)

        # Format as tensor: (1, seq_len, input_size)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Make prediction
        with torch.no_grad():
            prediction_scaled = self(input_tensor).cpu().numpy()

        # Inverse scale the prediction
        prediction_unscaled = self.scaler_y.inverse_transform(prediction_scaled)

        next_value = prediction_unscaled[0][0]

        print(f"[PREDICTION] Next value for '{TARGET}': {next_value:.4f}")
        return next_value
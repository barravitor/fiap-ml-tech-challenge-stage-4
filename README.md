# FIAP TECH CHALLENGE | Creating a Deep Learning Algorithm

## Índice

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation and run](#installation-and-run)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)

## Introduction

Project responsible for training a deep learning algorithms for predict price on stocks on B3 [link](https://www.b3.com.br)

## 🚀 Features

- **🔁 Deep Learning with LSTM**  
  A model based on LSTM neural networks for time series forecasting, ideal for capturing temporal patterns in stock market data.

- **⚙️ Automated Preprocessing Pipeline**  
  Includes:
  - Calculation of technical indicators (SMA, RSI)
  - Normalization using `MinMaxScaler`
  - Time window generation
  - Train/test data split

- **📊 Experiment Tracking with MLflow**  
  - Stores trained models with input signature and example
  - Logs key metrics: MAE, RMSE, MAPE
  - Logs training parameters and model architecture
  - Model versioning and artifact storage (models and scalers)

- **🧪 RESTful API with FastAPI**  
  Interface for integration with external systems.

  **Available Endpoints:**
  - `POST /predict`: Sends input data and receives forecast output
  - `GET /health`: Checks API health status

- **🧱 Modular and Scalable Architecture**  
  Code organized into reusable modules:
  - `train`, `data_loader`, `forecast_service`, `api`
  - Facilitates maintenance, testing, and extensibility

- **💾 Scaler Persistence**  
  - Input and output scalers are saved via MLflow and locally with `joblib`
  - Ensures consistent inference and reusability in production

- **📘 Auto-Generated Documentation with Swagger UI**  
  Interactive interface available at `/docs` for testing and exploring API endpoints.

  - [API Swagger](https://fiap-ml-tech-challenge-stage-4-production.up.railway.app/docs)
  - [Full API documentation](./docs/APIDocumentation.md)

## Technologies Used

- **Python**: The project's main language, chosen for its rich library for data analysis.
- **scikit-learn**: Library used for statistical modeling and machine learning, including algorithms such as regression, classification, preprocessing, cross-validation, and performance evaluation.
- **numpy**: Fundamental library for numerical computing. Used for high-performance vector operations, matrices and mathematical functions.
- **pandas**: Library used for manipulation and analysis of tabular data, such as reading CSVs, filtering, grouping and data transformations.
- **torch**: Deep learning framework used for building and training neural networks, providing GPU acceleration and automatic differentiation.
- **mlflow**: Open-source platform for managing the machine learning lifecycle, including experiment tracking, model packaging, and deployment.
- **yfinance**: Python library to download historical market data from Yahoo Finance, including stock prices, dividends, and financial statements.
- **FastApi**: Modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Installation and run

Instructions on how to install and run the project.

Create a .env file in the project root following the example in the .env-example file

Required python version: 3.10.12

```bash
python3.10 -m venv .venv # Run to create the environment
source .venv/bin/activate # Run to start the environment
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 # Run to install PyTorch in the right version
pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt # Run to install the necessary packages
```

```bash
./start_mlflow.sh # Run to execute ML Flow on browser on url: http://localhost:5000
```

Run the command on terminal to execute model training
```bash
python -m model_training.src.run
```

Run the API to load predict data
```bash
./start_api.sh # Run to execute API REST on url: http://localhost:8000
```

## Contribution

We welcome contributions to this project! Here’s how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Make your changes and commit them (`git commit -m 'feat: Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a Pull Request.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests where necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contact

For questions, suggestions, or feedback, please contact:

* **Edson Vitor**  
  GitHub: [barravitor](https://github.com/barravitor)
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .routes.index_routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting the application...")
    yield
    print("Closing the application...")

app = FastAPI(
    lifespan=lifespan,
    title="FIAP ML API | B3 Stocks Price Prediction",
    description="""
<h2>Welcome to the <strong>FIAP ML API | B3 Stocks Price Prediction</strong> documentation.</h2>
<p>This API provides endpoints for:</p>
<ul>
    <li><b><a href="#tag/Predict">Predict</a></b>: Generate stock price forecasts for B3-listed companies.</li>
</ul>
<p>Using historical market data, this API delivers deep learning-powered predictions to help analyze and anticipate stock price movements on the B3 exchange.</p>
<p>This project is developed for educational and non-commercial purposes.</p>
<p>For more details, visit our <a href="https://github.com/barravitor/fiap-ml-tech-challenge-stage-4" target="_blank">GitHub repository</a>.</p>
    """,
    version="1.0.0",
    openapi_tags=[{
        "name": "Predict",
        "description": "Operations related to predict stocks price."
    }]
)

app.include_router(router)

@app.get("/")
def read_root():
    return { "message": "Welcome to the API!" }
# app/main.py
import time
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import logging

from fastapi.middleware.cors import CORSMiddleware
from .routes.index_routes import router

logging.basicConfig(
    filename="api.log",
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='a'
)

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
    <li><b><a href="#tag/Health">Health</a></b>: Check the status of the API to ensure it's running and available.</li>
    <li><b><a href="#tag/Predict">Predict</a></b>: Generate stock price forecasts for B3-listed companies.</li>
</ul>
<p>Using historical market data, this API delivers deep learning-powered predictions to help analyze and anticipate stock price movements on the B3 exchange.</p>
<p>This project is developed for educational and non-commercial purposes.</p>
<p>For more details, visit our <a href="https://github.com/barravitor/fiap-ml-tech-challenge-stage-4" target="_blank">GitHub repository</a>.</p>
    """,
    version="1.0.0",
    openapi_tags=[{
        "name": "Health",
        "description": "Check the status and availability of the API."
    }, {
        "name": "Predict",
        "description": "Operations related to predict stocks price."
    }]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    log_details = f"{request.method} {request.url.path} {response.status_code} {process_time:.4f}s"
    logging.info(log_details)
    return response

app.include_router(router)

@app.get("/")
def read_root():
    return { "message": "Welcome to the API!" }
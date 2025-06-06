from fastapi import APIRouter
from ..routes.predict_routes import predict_router

router = APIRouter()

router.include_router(predict_router, prefix="/predict", tags=["Predict"])

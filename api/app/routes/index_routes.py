from fastapi import APIRouter
from ..routes.status_routes import status_router
from ..routes.predict_routes import predict_router

router = APIRouter()

router.include_router(status_router, prefix="/health", tags=["Health"])
router.include_router(predict_router, prefix="/predict", tags=["Predict"])

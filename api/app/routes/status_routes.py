# app/routes/status_routes.py

from fastapi import APIRouter

status_router = APIRouter()

@status_router.get("/")
def health():
    return { "status": "ok" }
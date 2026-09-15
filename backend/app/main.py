"""
NeuroPlay-AI FastAPI application entry point.
Run: uvicorn backend.app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuroplay.db.init_db import init_db
from neuroplay.logger import get_logger

from .dependencies import get_ann_model
from .routers import analytics, explain, game, leaderboard, psychology

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuroPlay-AI backend...")
    init_db()
    get_ann_model()  # Warms the cache - model loads once here, not per-request
    logger.info("Backend ready.")
    yield
    logger.info("Shutting down NeuroPlay-AI backend...")


app = FastAPI(title="NeuroPlay-AI API", version="0.1.0", lifespan=lifespan)


ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(game.router)
app.include_router(leaderboard.router)
app.include_router(explain.router)
app.include_router(analytics.router)
app.include_router(psychology.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "NeuroPlay-AI"}

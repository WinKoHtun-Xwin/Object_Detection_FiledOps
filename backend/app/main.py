"""FastAPI app factory + route registration."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import clips, health, stream
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="Object Detection Playground", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(clips.router, prefix="/api")
    app.include_router(stream.router)  # /ws at root

    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=settings.clips_dir), name="clips")
    return app


app = create_app()

"""FastAPI app factory + route registration."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import clips, health, recognition, stream
from app.config import settings
from app.recognition import db as dbmod
from app.recognition import gallery, live_face_engine, live_recognizer, recognition_worker

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    dbmod.init_schema()
    gallery.reload()
    recognition_worker.start()
    live_recognizer.start()
    try:
        live_face_engine.warmup()
    except Exception:
        log.exception("live face engine warmup failed")
    try:
        yield
    finally:
        recognition_worker.stop()
        live_recognizer.stop()


TAGS_METADATA = [
    {"name": "System", "description": "Health and device/GPU status."},
    {"name": "Clips", "description": "Recorded motion clips."},
    {"name": "People", "description": "Person gallery — CRUD and per-person clips."},
    {"name": "Review", "description": "Unknown-face review queue — label or dismiss."},
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Object Detection Playground",
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(clips.router, prefix="/api")
    app.include_router(recognition.router, prefix="/api")
    app.include_router(recognition.review_router, prefix="/api")
    app.include_router(stream.router)

    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=settings.clips_dir), name="clips")
    app.mount("/faces", StaticFiles(directory=settings.faces_dir), name="faces")
    return app


app = create_app()

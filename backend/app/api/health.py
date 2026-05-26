"""Health + model status endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.runtime.device import probe

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/device")
def device() -> dict:
    info = probe()
    return {"kind": info.kind, "name": info.name, "vram_mb": info.vram_mb}

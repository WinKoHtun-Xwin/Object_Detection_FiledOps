"""Clip listing endpoint — exposes recorded MP4s/JPGs to the frontend."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.config import settings

router = APIRouter()


@router.get("/clips")
def list_clips(
    camera_id: str = Query(..., min_length=1, max_length=64),
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    # Reject path-traversal in camera_id.
    if "/" in camera_id or "\\" in camera_id or camera_id.startswith("."):
        raise HTTPException(status_code=400, detail="invalid camera_id")

    cam_root = settings.clips_dir / camera_id
    if not cam_root.is_dir():
        return {"camera_id": camera_id, "clips": []}

    mp4s = list(cam_root.rglob("*.mp4"))
    mp4s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    mp4s = mp4s[:limit]

    clips_out: list[dict] = []
    for mp4 in mp4s:
        rel = mp4.relative_to(settings.clips_dir).as_posix()
        jpg = mp4.with_suffix(".jpg")
        clips_out.append(
            {
                "name": mp4.stem,
                "date": mp4.parent.name,
                "mp4_url": f"/clips/{rel}",
                "jpg_url": f"/clips/{jpg.relative_to(settings.clips_dir).as_posix()}" if jpg.exists() else None,
                "size_bytes": mp4.stat().st_size,
                "mtime": mp4.stat().st_mtime,
            }
        )
    return {"camera_id": camera_id, "clips": clips_out}

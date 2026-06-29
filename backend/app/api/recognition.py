"""Recognition REST endpoints."""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.recognition import db as dbmod
from app.recognition import face_engine, gallery

log = logging.getLogger(__name__)
router = APIRouter(tags=["People"])
review_router = APIRouter(tags=["Review"])


def _person_to_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "face_count": row.get("face_count", 0),
        "latest_face_url": (
            f"/faces/{row['latest_crop_path']}"
            if row.get("latest_crop_path") else None
        ),
    }


@router.get("/people")
def list_people() -> list[dict]:
    return [_person_to_dict(p) for p in dbmod.list_persons()]


@router.post("/people")
async def create_person(
    name: Annotated[str, Form(min_length=1, max_length=64)],
    files: Annotated[list[UploadFile], File()],
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="at least one image required")

    pid = dbmod.add_person(name)
    faces_added = 0

    person_dir = settings.faces_dir / str(pid)
    person_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        data = await upload.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("could not decode upload %s", upload.filename)
            continue
        for face in face_engine.detect_and_embed(img):
            crop_name = f"upload_{uuid.uuid4().hex}.jpg"
            crop_rel = f"{pid}/{crop_name}"
            cv2.imwrite(str(settings.faces_dir / crop_rel), face.crop_bgr)
            dbmod.add_face(pid, crop_path=crop_rel, embedding=face.embedding.tobytes(), source="upload")
            gallery.add(pid, face.embedding)
            faces_added += 1

    return {"id": pid, "name": name, "face_count": faces_added}


@router.delete("/people/{person_id}")
def delete_person(person_id: int) -> dict:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    dbmod.delete_person(person_id)
    gallery.reload()
    return {"ok": True}


@router.get("/people/{person_id}")
def get_person(person_id: int) -> dict:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    faces = dbmod.list_faces_for(person_id)
    return {
        "id": person["id"],
        "name": person["name"],
        "faces": [
            {
                "id": f["id"],
                "crop_url": f"/faces/{f['crop_path']}",
                "source": f["source"],
                "created_at": f["created_at"],
            }
            for f in faces
        ],
    }


@router.get("/people/{person_id}/clips")
def list_person_clips(person_id: int) -> list[dict]:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    rows = dbmod.clips_for_person(person_id)
    out = []
    for r in rows:
        clip_rel = r["clip_path"]
        mp4_url = f"/clips/{clip_rel}"
        jpg_url = mp4_url.rsplit(".", 1)[0] + ".jpg"
        out.append(
            {
                "clip_path": clip_rel,
                "mp4_url": mp4_url,
                "jpg_url": jpg_url,
                "confidence": r["confidence"],
                "frame_ts": r["frame_ts"],
                "created_at": r["created_at"],
            }
        )
    return out


@review_router.get("/review")
def list_review(limit: int = 50) -> list[dict]:
    rows = dbmod.list_queue(status="pending", limit=limit)
    return [
        {
            "id": r["id"],
            "crop_url": f"/faces/{r['crop_path']}",
            "suggested": (
                {
                    "id": r["suggested_id"],
                    "name": r["suggested_name"],
                    "score": r["suggested_score"],
                }
                if r["suggested_id"] else None
            ),
            "source_clip": r["source_clip"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@review_router.post("/review/{queue_id}/label")
def label_review(queue_id: int, body: dict) -> dict:
    item = dbmod.get_queue_item(queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="queue item not found")

    person_id = body.get("person_id")
    new_name = body.get("new_name")
    if person_id is None and not new_name:
        raise HTTPException(status_code=400, detail="provide person_id or new_name")

    if person_id is None:
        person_id = dbmod.add_person(str(new_name))

    src = settings.faces_dir / item["crop_path"]
    dest_dir = settings.faces_dir / str(person_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"queue_{uuid.uuid4().hex}.jpg"
    dest_rel = f"{person_id}/{dest_name}"
    dest_abs = settings.faces_dir / dest_rel
    if src.exists():
        dest_abs.write_bytes(src.read_bytes())

    embedding = np.frombuffer(item["embedding"], dtype=np.float32)
    dbmod.add_face(person_id, crop_path=dest_rel, embedding=item["embedding"], source="queue")
    gallery.add(person_id, embedding)
    dbmod.update_queue_status(queue_id, "labeled")

    return {"person_id": person_id}


@review_router.post("/review/{queue_id}/dismiss")
def dismiss_review(queue_id: int) -> dict:
    item = dbmod.get_queue_item(queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="queue item not found")
    dbmod.update_queue_status(queue_id, "dismissed")
    return {"ok": True}

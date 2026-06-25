"""WebSocket inference endpoint."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.protocol.frame import FrameDecodeError, parse
from app.inference.tracking import camera_trackers
from app.recognition import live_recognizer
from app.recognition.live import FaceMatch
from app.recording import recorder_manager
from app.runtime.codec import jpeg_to_bgr
from app.runtime.registry import registry

log = logging.getLogger(__name__)
router = APIRouter()

MODE_BY_ID = {
    0: "yolo_detect",
    1: "yolo_pose",
    2: "sam3_image",
    3: "sam3_video",
    4: "yolo_seg",
    5: "yolo_obb",
    6: "yolo_cls",
}

SIZE_BY_VARIANT = {0: "n", 1: "s", 2: "m", 3: "l", 4: "x"}


def apply_live_names(boxes: list[dict[str, Any]], names: dict[int, FaceMatch]) -> None:
    """Append recognized names to box labels, matched by track_id."""
    for b in boxes:
        tid = b.get("track_id")
        if tid is None:
            continue
        m = names.get(tid)
        if m is None:
            continue
        suffix = f" · {m.name} {m.score:.2f}" if m.kind != "unknown" else " · Unknown"
        b["label"] = f"{b.get('label', 'person')}{suffix}"


def _detections_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten any inference result into a list of Box-shaped dicts.

    Only modes that surface bounding boxes contribute to motion. For modes
    without boxes (cls, sam3, obb without box equivalents), returns []."""
    if "boxes" in result:
        return result["boxes"]
    if "people" in result:
        return [p["box"] for p in result["people"]]
    return []


@router.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    log.info("ws connected: %s", ws.client)
    frame_id = 0
    current_camera_id: str | None = None
    try:
        while True:
            packet = await ws.receive_bytes()
            t0 = time.perf_counter()
            try:
                fp = parse(packet)
                img = jpeg_to_bgr(fp.jpeg)
            except (FrameDecodeError, ValueError) as e:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": str(e)})
                frame_id += 1
                continue

            mode = MODE_BY_ID.get(fp.mode_id)
            if mode is None:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": f"unknown mode {fp.mode_id}"})
                frame_id += 1
                continue

            size = SIZE_BY_VARIANT.get(fp.variant_id, "n") if mode.startswith("yolo_") else "n"
            try:
                engine = registry.get(mode, size=size)
                result = engine.infer(img, fp.header)
            except (ValueError, NotImplementedError) as e:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": str(e)})
                frame_id += 1
                continue

            ms = (time.perf_counter() - t0) * 1000.0
            result["frame_id"] = frame_id
            result["ms"] = round(ms, 2)

            camera_id = fp.header.get("camera_id") if isinstance(fp.header, dict) else None
            if camera_id:
                current_camera_id = camera_id
                recorder_manager.feed(camera_id, img, _detections_from_result(result))

            if camera_id and isinstance(fp.header, dict) and fp.header.get("recognize"):
                result_boxes = result.get("boxes") or []
                person_boxes = [b for b in result_boxes if b.get("label") == "person"]
                if person_boxes:
                    try:
                        names = live_recognizer.annotate(camera_id, img, person_boxes)
                        apply_live_names(person_boxes, names)
                    except Exception:
                        log.exception("live recognition failed")

            await ws.send_json(result)
            frame_id += 1
    except WebSocketDisconnect:
        log.info("ws disconnected: %s (after %d frames)", ws.client, frame_id)
        if current_camera_id is not None:
            recorder_manager.close(current_camera_id)
            camera_trackers.drop(current_camera_id)
            live_recognizer.clear_camera(current_camera_id)

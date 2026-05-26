"""WebSocket inference endpoint."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.protocol.frame import FrameDecodeError, parse
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


@router.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    log.info("ws connected: %s", ws.client)
    frame_id = 0
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
            await ws.send_json(result)
            frame_id += 1
    except WebSocketDisconnect:
        log.info("ws disconnected: %s (after %d frames)", ws.client, frame_id)

"""Smoke-test every YOLO26 task at size n via the WS endpoint.

Each task is exercised once (lazy-load + warmup) and then a warm frame.
"""
import asyncio
import json
import struct

import cv2
import numpy as np
import websockets

MODES = [
    (0, "detect"),
    (4, "seg"),
    (1, "pose"),
    (5, "obb"),
    (6, "cls"),
]


def make_packet(mode_id: int, variant_id: int = 0) -> bytes:
    img = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    hdr = json.dumps({"conf": 0.25}).encode()
    return struct.pack("<BBI", mode_id, variant_id, len(hdr)) + hdr + buf.tobytes()


async def main() -> None:
    async with websockets.connect("ws://localhost:8000/ws", max_size=8 * 1024 * 1024) as ws:
        for mode_id, name in MODES:
            pkt = make_packet(mode_id)
            await ws.send(pkt)
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=240))
            await ws.send(pkt)
            warm = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            summary = lambda r: {
                "type": r.get("type"),
                "ms": r.get("ms"),
                "msg": r.get("message"),
                "counts": _counts(r),
            }
            print(f"[{name}] first={summary(first)}  warm={summary(warm)}")


def _counts(r: dict) -> dict:
    return {
        "boxes": len(r.get("boxes", [])),
        "masks": len(r.get("masks", [])),
        "people": len(r.get("people", [])),
        "obb": len(r.get("obboxes", [])),
        "topk": len(r.get("topk", [])),
    }


asyncio.run(main())

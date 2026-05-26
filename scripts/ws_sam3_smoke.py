"""WS smoke test for SAM 3 mode: send 3 frames, print result summaries."""
import asyncio
import json
import struct

import cv2
import numpy as np
import websockets


async def main() -> None:
    img = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg = buf.tobytes()
    hdr = json.dumps({"text": "person"}).encode()
    pkt = struct.pack("<BBI", 2, 0, len(hdr)) + hdr + jpeg  # mode_id=2 = sam3_image

    async with websockets.connect("ws://localhost:8000/ws") as ws:
        for label in ("first", "warm1", "warm2"):
            await ws.send(pkt)
            raw = await asyncio.wait_for(ws.recv(), timeout=180)
            data = json.loads(raw)
            preview = {
                "type": data.get("type"),
                "frame_id": data.get("frame_id"),
                "ms": data.get("ms"),
                "n_masks": len(data.get("masks", [])),
                "message": data.get("message"),
            }
            print(label, preview)


asyncio.run(main())

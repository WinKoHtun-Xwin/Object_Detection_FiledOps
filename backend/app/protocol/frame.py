"""Binary frame packet parsing (client -> server).

Packet layout:
    offset  bytes  field
    0       1      mode_id        0=yolo_detect, 1=yolo_pose, 2=sam3_image, 3=sam3_video
    1       1      variant_id     YOLO size or SAM3 prompt kind (engine-specific)
    2       4      header_len     uint32 LE -- length of JSON header (0 if none)
    6       N      header_json    optional JSON (text/points/box/session_id)
    6+N     ...    jpeg_bytes     JPEG-encoded frame
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

HEADER_FIXED = 6  # 1 + 1 + 4


@dataclass(frozen=True)
class FramePacket:
    mode_id: int
    variant_id: int
    header: dict[str, Any]
    jpeg: bytes


class FrameDecodeError(ValueError):
    pass


def parse(packet: bytes) -> FramePacket:
    if len(packet) < HEADER_FIXED:
        raise FrameDecodeError(f"packet too short: {len(packet)} bytes")

    mode_id, variant_id, header_len = struct.unpack_from("<BBI", packet, 0)
    end_header = HEADER_FIXED + header_len
    if len(packet) < end_header:
        raise FrameDecodeError(f"header_len={header_len} but packet has {len(packet)} bytes")

    header: dict[str, Any] = {}
    if header_len:
        try:
            header = json.loads(packet[HEADER_FIXED:end_header].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise FrameDecodeError(f"invalid header json: {e}") from e

    jpeg = packet[end_header:]
    if not jpeg:
        raise FrameDecodeError("empty JPEG payload")

    return FramePacket(mode_id=mode_id, variant_id=variant_id, header=header, jpeg=jpeg)

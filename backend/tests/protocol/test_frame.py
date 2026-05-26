import json
import struct

import pytest

from app.protocol.frame import FrameDecodeError, parse


def build(mode: int, variant: int, header: dict | None, jpeg: bytes) -> bytes:
    hdr_bytes = json.dumps(header).encode("utf-8") if header else b""
    return struct.pack("<BBI", mode, variant, len(hdr_bytes)) + hdr_bytes + jpeg


def test_parse_minimal():
    pkt = build(0, 0, None, b"\xff\xd8\xff\xd9")
    fp = parse(pkt)
    assert fp.mode_id == 0
    assert fp.variant_id == 0
    assert fp.header == {}
    assert fp.jpeg == b"\xff\xd8\xff\xd9"


def test_parse_with_header():
    pkt = build(2, 1, {"text": "person"}, b"\xff\xd8")
    fp = parse(pkt)
    assert fp.mode_id == 2
    assert fp.header == {"text": "person"}


def test_too_short():
    with pytest.raises(FrameDecodeError):
        parse(b"\x00\x00")


def test_empty_jpeg():
    with pytest.raises(FrameDecodeError):
        parse(struct.pack("<BBI", 0, 0, 0))


def test_bad_header_json():
    bad = struct.pack("<BBI", 0, 0, 3) + b"{!}" + b"\xff\xd8"
    with pytest.raises(FrameDecodeError):
        parse(bad)

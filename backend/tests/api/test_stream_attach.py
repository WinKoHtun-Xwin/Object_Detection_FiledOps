"""apply_live_names attaches cached names to boxes by track_id (no drift)."""
from __future__ import annotations

from app.api.stream import apply_live_names
from app.recognition.live import FaceMatch


def test_attaches_high_match_with_score():
    boxes = [{"label": "person", "track_id": 2}]
    apply_live_names(boxes, {2: FaceMatch("Bob", 0.7, "high")})
    assert boxes[0]["label"] == "person · Bob 0.70"


def test_unknown_kind_gets_plain_suffix():
    boxes = [{"label": "person", "track_id": 5}]
    apply_live_names(boxes, {5: FaceMatch("Unknown", 0.0, "unknown")})
    assert boxes[0]["label"] == "person · Unknown"


def test_untracked_and_unmatched_boxes_unchanged():
    boxes = [{"label": "person"}, {"label": "dog", "track_id": 9}]
    apply_live_names(boxes, {2: FaceMatch("Bob", 0.7, "high")})
    assert boxes[0]["label"] == "person"
    assert boxes[1]["label"] == "dog"


def test_no_drift_with_multiple_people():
    boxes = [{"label": "person", "track_id": 7}, {"label": "person", "track_id": 3}]
    names = {3: FaceMatch("Cara", 0.8, "high"), 7: FaceMatch("Dan", 0.6, "mid")}
    apply_live_names(boxes, names)
    assert boxes[0]["label"] == "person · Dan 0.60"
    assert boxes[1]["label"] == "person · Cara 0.80"

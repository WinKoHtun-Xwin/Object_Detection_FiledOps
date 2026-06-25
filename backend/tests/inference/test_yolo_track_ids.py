"""_extract_boxes stamps track IDs aligned to detection order."""
from __future__ import annotations

import numpy as np

from app.inference.yolo import _extract_boxes


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)

    def __len__(self):
        return len(self.conf.arr)

    def cpu(self):
        return self

    def numpy(self):
        # Return the xyxy array as a stand-in for the raw boxes array
        return self.xyxy.numpy()


class _FakeResult:
    def __init__(self, xyxy, cls, conf, names):
        self.boxes = _FakeBoxes(xyxy, cls, conf)
        self.names = names


def _result():
    return _FakeResult(
        xyxy=[[0.0, 0.0, 50.0, 50.0], [50.0, 50.0, 100.0, 100.0]],
        cls=[0, 16],
        conf=[0.9, 0.8],
        names={0: "person", 16: "dog"},
    )


def test_track_ids_default_none():
    boxes = _extract_boxes(_result(), w=100, h=100)
    assert [b["track_id"] for b in boxes] == [None, None]


def test_track_ids_applied_in_order():
    boxes = _extract_boxes(_result(), w=100, h=100, track_ids=[7, 3])
    assert boxes[0]["track_id"] == 7
    assert boxes[1]["track_id"] == 3
    assert boxes[0]["label"] == "person"
    assert boxes[1]["label"] == "dog"


def test_infer_detect_stamps_track_ids(monkeypatch):
    from unittest.mock import MagicMock

    from app.inference.yolo import YoloEngine
    import app.inference.tracking as trackmod

    eng = object.__new__(YoloEngine)   # bypass __init__ (no model load / GPU)
    eng.task = "detect"
    eng.size = "n"
    eng.device = "cpu"
    eng.model = MagicMock()
    eng.model.predict.return_value = [_result()]

    captured = {}

    def fake_assign(camera_id, det, frame):
        captured["camera_id"] = camera_id
        return [11, 22]

    monkeypatch.setattr(trackmod.camera_trackers, "assign", fake_assign)

    out = eng.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        {"tracking": True, "camera_id": "cam-A", "conf": 0.25},
    )
    assert out["type"] == "detect"
    assert [b["track_id"] for b in out["boxes"]] == [11, 22]
    assert captured["camera_id"] == "cam-A"


def test_infer_detect_no_tracking_leaves_ids_none(monkeypatch):
    from unittest.mock import MagicMock

    from app.inference.yolo import YoloEngine
    import app.inference.tracking as trackmod

    eng = object.__new__(YoloEngine)
    eng.task = "detect"
    eng.size = "n"
    eng.device = "cpu"
    eng.model = MagicMock()
    eng.model.predict.return_value = [_result()]

    def boom(*a, **k):
        raise AssertionError("assign must not be called without tracking/recognize")

    monkeypatch.setattr(trackmod.camera_trackers, "assign", boom)

    out = eng.infer(np.zeros((100, 100, 3), dtype=np.uint8), {"conf": 0.25})
    assert [b["track_id"] for b in out["boxes"]] == [None, None]


def test_infer_detect_recognize_implies_tracking(monkeypatch):
    """Recognition pins names to track IDs, so `recognize` alone (no `tracking`
    flag, which the frontend omits for YOLO modes) must still stamp track IDs."""
    from unittest.mock import MagicMock

    from app.inference.yolo import YoloEngine
    import app.inference.tracking as trackmod

    eng = object.__new__(YoloEngine)
    eng.task = "detect"
    eng.size = "n"
    eng.device = "cpu"
    eng.model = MagicMock()
    eng.model.predict.return_value = [_result()]

    monkeypatch.setattr(trackmod.camera_trackers, "assign", lambda cam, det, frame: [5, 6])

    out = eng.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        {"recognize": True, "camera_id": "cam-A", "conf": 0.25},
    )
    assert [b["track_id"] for b in out["boxes"]] == [5, 6]

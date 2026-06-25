"""CameraTrackers: stable IDs across frames, isolation across cameras."""
from __future__ import annotations

import numpy as np

from app.inference.tracking import CameraTrackers


class _Dets:
    """Minimal stand-in for an Ultralytics numpy Boxes (what BYTETracker reads)."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])


def _two_people():
    # xywh = center-x, center-y, width, height (pixels)
    return _Dets(
        xywh=[[100.0, 100.0, 40.0, 80.0], [400.0, 300.0, 40.0, 80.0]],
        conf=[0.9, 0.9],
        cls=[0.0, 0.0],
    )


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_ids_assigned_and_stable_across_frames():
    ct = CameraTrackers()
    ids1 = ct.assign("cam-A", _two_people(), _frame())
    ids2 = ct.assign("cam-A", _two_people(), _frame())
    assert len(ids1) == 2 and all(i is not None for i in ids1)
    assert ids1 == ids2            # same detections → same IDs next frame
    assert ids1[0] != ids1[1]      # two distinct people → distinct IDs


def test_ids_are_isolated_across_cameras():
    ct = CameraTrackers()
    a = ct.assign("cam-A", _two_people(), _frame())
    b = ct.assign("cam-B", _two_people(), _frame())
    # Independent trackers, but the monotonic-counter guard keeps IDs disjoint.
    assert set(a).isdisjoint(set(b))


def test_empty_detections_returns_empty():
    ct = CameraTrackers()
    out = ct.assign("cam-A", _Dets([], [], []), _frame())
    assert out == []


def test_low_confidence_detection_does_not_shift_track_ids():
    ct = CameraTrackers()
    # Middle detection is below track_high_thresh (0.25): the tracker drops it, so the
    # two high-confidence detections must keep their ORIGINAL indices 0 and 2.
    dets = _Dets(
        xywh=[[100.0, 100.0, 40.0, 80.0], [300.0, 100.0, 40.0, 80.0], [500.0, 100.0, 40.0, 80.0]],
        conf=[0.9, 0.1, 0.9],
        cls=[0.0, 0.0, 0.0],
    )
    ids = ct.assign("cam-A", dets, _frame())
    assert len(ids) == 3
    assert ids[0] is not None
    assert ids[1] is None            # low-confidence detection gets no track id
    assert ids[2] is not None
    assert ids[0] != ids[2]

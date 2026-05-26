# Object Detection BS — Phase 1 Design

**Date:** 2026-05-25
**Phase:** 1 of 4 (Pipeline skeleton: video input → preprocess → detect → track)
**Status:** Drafted from brainstorming session, awaiting user review

---

## 1. Project overview

The full system is a real-time, person-aware video understanding pipeline that runs on a CPU-only laptop with a webcam. End state: it watches the camera continuously, recognizes people you have enrolled, infers what they are doing, stores that history, and lets you ask natural-language questions over it ("what was Alice doing yesterday?").

The system is built in **four phases**, each independently runnable:

| Phase | Pipeline stages added | Demo deliverable |
|---|---|---|
| **1** *(this spec)* | Video input · preprocessing · object detection · human tracking | Live OpenCV window: COCO objects boxed, persons additionally labeled with `track #N`, FPS overlay |
| 2 | Human re-identification (face) · enrollment storage · training review queue (Frigate-style) | Tiny FastAPI web UI for enrollment + review; live window labels persons by name |
| 3 | Pose / action recognition · event & activity storage (thumbs + clips) | Live window writes events to SQLite with timestamps; thumbnails per event, clips for significant events |
| 4 | Query engine | CLI tool: natural-language Q&A over the event log via Claude API |

This document specifies **Phase 1 only**. Phases 2-4 are referenced for context but will get their own design docs in future sessions.

## 2. Full-system pipeline (reference)

For reference, the eight runtime stages of the complete system, in data-flow order:

```
1. Video Input        → webcam / file / stream
2. Preprocessing      → resize for model, frame skip (CPU throttle)
3. Object Detection   → YOLO11n on all 80 COCO classes
4. Human Tracking     → ByteTrack assigns track IDs to person-class detections
5. Pose / Action      → MediaPipe pose → simple state machine (sit/stand/walk)        [Phase 3]
6. Event Storage      → SQLite events table + thumbs/<date>/ + clips/<date>/          [Phase 3]
7. Human Re-ID        → face detect (MediaPipe) + embed (InsightFace) + gallery       [Phase 2]
8. Query Engine       → NL question → SQL query → Claude summary                      [Phase 4]
```

Phase 1 builds stages 1–4. Re-ID logically comes "after" tracking in the pipeline, but in build order we wait for Phase 2 because it needs its own storage layer.

## 3. North-star file layout (final, all phases)

```
Object_Detection_BS/
├── obj_detect/                    # Python package — shared code across phases
│   ├── __init__.py
│   ├── video.py                   # stage 1: video input wrapper                     [Phase 1]
│   ├── preprocess.py              # stage 2: frame-skip                              [Phase 1]
│   ├── detector.py                # stage 3: YOLO11n wrapper                         [Phase 1]
│   ├── tracker.py                 # stage 4: person-class tracking via ByteTrack     [Phase 1]
│   ├── display.py                 # draw boxes/labels/FPS                            [Phase 1]
│   ├── storage.py                 # SQLite connection helper                         [Phase 1, schema grows later]
│   ├── face.py                    # face detect + embed                              [Phase 2]
│   ├── gallery.py                 # enrolled persons + training candidates           [Phase 2]
│   ├── pose.py                    # pose + action state machine                      [Phase 3]
│   ├── events.py                  # event log writer                                 [Phase 3]
│   └── query.py                   # NL → SQL via Claude                              [Phase 4]
├── apps/
│   ├── live.py                    # main webcam loop (OpenCV window)                 [Phase 1]
│   ├── enroll_web.py              # FastAPI enrollment + review UI                   [Phase 2]
│   └── ask.py                     # CLI Q&A tool                                     [Phase 4]
├── data/                          # generated, gitignored — created on first run     [Phase 2+]
│   ├── app.db                     # SQLite
│   ├── thumbs/enrolled/<name>/    # reference face crops from enrollment
│   ├── thumbs/sightings/<date>/   # live face crops on entry
│   ├── thumbs/events/<date>/      # event thumbnails (Phase 3)
│   └── clips/<date>/              # short MP4 clips (Phase 3)
├── tests/
│   ├── fixtures/                  # small sample image(s) for offline tests
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_display.py
├── requirements.txt
└── README.md
```

Phase 1 creates everything marked `[Phase 1]` plus stub files for modules built later (empty `__init__.py` entries only — no functions yet).

## 4. Phase 1 scope

### 4.1 Goal

A runnable single-command webcam demo:

```
python apps/live.py
```

This opens an OpenCV window showing the default webcam feed annotated with:
- bounding boxes for all detected COCO objects (80 classes)
- class name + confidence for every box
- for `person` detections, an additional `track #N` label that remains stable while the person stays in frame
- an FPS overlay (top-left)
- a `q=quit` hint (bottom-right)

Pressing `q` closes the window and shuts down cleanly. No persistence yet — nothing is written to disk.

### 4.2 Modules

| Module | Responsibility | Inputs → Outputs |
|---|---|---|
| `obj_detect/video.py` | Open a video source, yield frames, release on exit. Context-manager interface. | `source: int \| str` → iterator of `numpy.ndarray` BGR frames |
| `obj_detect/preprocess.py` | Frame-skip policy. Resize is handled inside YOLO. | `frame_iter, skip_n` → filtered frame iterator |
| `obj_detect/detector.py` | Holds the `Detection` dataclass and the default model-path constant. No inference logic in Phase 1 — `Tracker` owns the one YOLO call per frame. Phase 2 will add a `predict()` function here for still-image detection (used by the enrollment app). | exports `Detection`, `DEFAULT_MODEL_PATH` |
| `obj_detect/tracker.py` | Load YOLO11n. Call `.track()` once per frame (detection + ByteTrack in one pass). Split results into `(detections, person_tracks)`. | `frame` → `(list[Detection], list[Track])` |
| `obj_detect/display.py` | Draw boxes, labels, FPS overlay, quit hint. Returns a new annotated frame (does not mutate input). | `frame, detections, tracks, fps` → annotated frame |
| `obj_detect/storage.py` | SQLite connection helper. No tables yet in Phase 1, just `get_connection()` returning a connection to `data/app.db`. Exists so Phase 2 can extend with schema migrations. | `()` → `sqlite3.Connection` |
| `apps/live.py` | Wire it all together. Owns the session: FPS meter, cv2 window. | runs to completion |

Each module is small and single-purpose. `apps/live.py` is the only place that holds session state.

### 4.3 Key interfaces

```python
# obj_detect/detector.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str          # e.g. "person", "car"
    conf: float              # 0.0–1.0
    xyxy: tuple[int, int, int, int]   # pixel coords (x1, y1, x2, y2)
```

```python
# obj_detect/tracker.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Track:
    track_id: int            # ByteTrack-assigned, stable across frames in this session
    class_name: str          # always "person" in Phase 1
    conf: float
    xyxy: tuple[int, int, int, int]

class Tracker:
    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.4) -> None: ...
    def update(self, frame: np.ndarray) -> tuple[list[Detection], list[Track]]: ...
```

The tracker owns the single Ultralytics `.track()` call per frame. Detection results for all classes are returned; tracks are filtered to the `person` class only.

```python
# obj_detect/video.py
class VideoSource:
    def __init__(self, source: int | str = 0) -> None: ...
    def __enter__(self) -> "VideoSource": ...
    def __exit__(self, *exc): ...
    def __iter__(self) -> Iterator[np.ndarray]: ...
```

```python
# obj_detect/display.py
def draw(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    fps: float,
) -> np.ndarray:
    """Return a new annotated frame. Does not mutate `frame`."""
```

### 4.4 Main loop (`apps/live.py`)

```python
# pseudo-code, target ~40-60 lines
from obj_detect.video import VideoSource
from obj_detect.preprocess import iter_with_skip
from obj_detect.tracker import Tracker
from obj_detect import display
import cv2

SOURCE = 0          # default webcam
SKIP_N = 1          # 1 = no skip; raise to throttle CPU
WINDOW = "obj-detect"

tracker = Tracker()
fps_meter = FPSMeter(window=30)

with VideoSource(SOURCE) as video:
    for frame in iter_with_skip(video, skip_n=SKIP_N):
        detections, tracks = tracker.update(frame)
        fps = fps_meter.tick()
        annotated = display.draw(frame, detections, tracks, fps)
        cv2.imshow(WINDOW, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.destroyAllWindows()
```

`FPSMeter` is a small helper inside `live.py` (or `display.py`) — a sliding window over the last N frame timestamps. It's not its own module yet.

### 4.5 Visual output

```
┌──────────────────────────────────────────────────────────┐
│  FPS 24                                                  │
│                                                          │
│      ┌──────────────────┐                                │
│      │  person #3 0.87  │                                │
│      │ ┌──────────────┐ │                                │
│      │ │              │ │       ┌────────────────────┐   │
│      │ │              │ │       │  cup 0.71          │   │
│      │ │              │ │       │ ┌────────────────┐ │   │
│      │ │              │ │       │ │                │ │   │
│      │ │              │ │       │ └────────────────┘ │   │
│      │ └──────────────┘ │       └────────────────────┘   │
│      └──────────────────┘                                │
│                                                          │
│                                                  q=quit  │
└──────────────────────────────────────────────────────────┘
```

Drawing rules:
- **Never double-draw persons.** Tracker returns both a `Detection` *and* a `Track` for the same person. `display.draw()` draws:
  - All `Track` boxes (these are the persons, with `track #N` labels)
  - Only `Detection` boxes whose `class_name != "person"` (everything else)
- Persons: one distinct color per `track_id`, deterministically chosen by `track_id % len(palette)`. The same track keeps the same color across frames.
- Non-person classes: single neutral color (e.g. light gray).
- FPS text: top-left, white with a dark outline so it stays readable on any background.
- Quit hint: bottom-right, same style as FPS.

### 4.6 Dependencies (`requirements.txt`)

```
ultralytics>=8.3       # YOLO11 + built-in ByteTrack
opencv-python>=4.9
numpy>=1.26
```

On first run, `ultralytics` will auto-download `yolo11n.pt` (~6 MB) to the working directory. The implementer can pin a specific cache location later if needed.

Dev-only:

```
pytest>=8.0
```

A `.gitignore` is also part of the Phase 1 setup. Minimum entries:

```
__pycache__/
*.pyc
.pytest_cache/
*.pt              # downloaded YOLO weights
data/             # all generated runtime data (used Phase 2+)
.venv/
```

### 4.7 Configuration

All Phase 1 configuration is hard-coded constants at the top of `apps/live.py`. No CLI flags, no config file. YAGNI — once we have CLI flags in Phase 2+ we can extract these.

| Constant | Default | Purpose |
|---|---|---|
| `SOURCE` | `0` | Video source (0 = default webcam) |
| `SKIP_N` | `1` | Process every Nth frame (1 = no skip) |
| `MODEL_PATH` | `"yolo11n.pt"` | YOLO weights file (auto-downloads) |
| `CONF_THRESHOLD` | `0.4` | Minimum detection confidence |
| `FPS_WINDOW` | `30` | Frames used for the rolling FPS average |

### 4.8 Tests

Three smoke tests. CI-friendly (no webcam required). Use `tests/fixtures/sample.jpg` — a small image containing at least one person and one other COCO class (e.g. Ultralytics' bundled `bus.jpg`, copied into the fixtures dir at setup time).

1. **`tests/test_detector.py`** — instantiate `Tracker`, call `.update()` on the fixture image, assert:
   - at least one `Detection` is returned
   - every detection has a non-empty `class_name`, `0 ≤ conf ≤ 1`, and a valid `xyxy` tuple

2. **`tests/test_tracker.py`** — feed the same fixture image twice in sequence, assert:
   - at least one `Track` is returned in both calls
   - at least one `track_id` is identical across the two calls (ByteTrack should re-associate)

3. **`tests/test_display.py`** — draw on a fixed black 480×640 frame with a synthetic `Detection` and `Track`, assert:
   - output is the same shape and dtype as input
   - output is not pixel-identical to input (something was drawn)

Run with `pytest tests/`.

## 5. Success criteria

Phase 1 is "done" when **all** of the following hold:

- [ ] `python apps/live.py` opens an OpenCV window showing the webcam feed
- [ ] COCO objects are drawn with class name + confidence
- [ ] Persons additionally show `track #N` labels that stay stable while the person remains in frame
- [ ] Different track IDs render in different colors
- [ ] FPS overlay updates in real time
- [ ] Pressing `q` closes the window cleanly (no traceback)
- [ ] All three smoke tests pass under `pytest tests/`
- [ ] Throughput: ≥ 10 FPS on a modern CPU (sanity floor, not a hard SLO)
- [ ] `requirements.txt` is sufficient — a fresh `pip install -r requirements.txt` then `python apps/live.py` works without further setup
- [ ] `README.md` documents the one-line "how to run"
- [ ] `.gitignore` excludes `__pycache__/`, `*.pt`, `data/`, `.pytest_cache/`, `.venv/`

## 6. Design decisions

### 6.1 Why Ultralytics YOLO11n
Newest small YOLO from Ultralytics; one-line model load; pretrained COCO weights auto-download; works real-time on CPU at the nano size (~15–30 FPS). Easy to swap to a larger variant later by changing one string.

### 6.2 Why ByteTrack (and why bundled, not standalone)
ByteTrack is built into Ultralytics — no extra dependency. It's a strong default tracker that handles short occlusions well. Standalone trackers (DeepSORT, BoT-SORT-ReID) need their own models and add complexity that Phase 1 doesn't need.

### 6.3 Why a single combined detection + track call (not two separate passes)
Ultralytics' `.track()` runs detection internally and adds tracking. Running YOLO twice per frame (once for "all detections", once for "person tracks") roughly halves CPU FPS for no real benefit. The tracker module owns the combined call and splits results.

Trade-off accepted: `detector.py` is a thin module that only exposes the underlying model handle. The "real" inference happens inside `tracker.py`. The boundary is slightly weaker than in a pure-OO design, but for ~80 lines of code and a CPU constraint, throughput wins.

### 6.4 Why no CLI flags in v1
YAGNI. The defaults are correct for the demo. Adding `--source` / `--conf` / `--save` etc. introduces argument parsing and matrix testing for features nobody needs yet. Phase 2's web app will be where configurability arrives.

### 6.5 Why no persistence in Phase 1
SQLite schema decisions are tightly coupled to Phase 2 needs (faces, embeddings, training candidates). Designing the schema now would either over-engineer for hypotheticals or have to be redone. We ship Phase 1 stateless and add the schema with Phase 2's first migration. `storage.py` exists as a placeholder so the file structure is stable.

### 6.6 Color-by-track-id
Persons get a color derived from `track_id`. This makes it visually obvious when the tracker switches IDs (a person changes color = ID switch happened). Useful for sanity-checking Phase 1 quality and for Phase 2 debugging.

## 7. Out of scope for Phase 1 (explicit deferrals)

- No CLI flags. Hard-coded defaults only.
- No saving frames, snapshots, or video to disk.
- No face detection, recognition, or enrollment. (Phase 2)
- No training review queue or web UI. (Phase 2)
- No pose, action, or activity classification. (Phase 3)
- No event log or query engine. (Phase 4)
- No GPU / CUDA paths. CPU only. Ultralytics auto-falls-back to CPU.
- No denoise / stabilize preprocessing. Only frame-skip in v1.
- No multi-camera support. Single source.
- No remote / network sources (RTSP, HTTP) tested in v1, though `VideoSource(str_path)` should "just work" for video files because OpenCV handles them.

## 8. Open questions for future phases

These do not block Phase 1, but are noted so future sessions can pick them up:

- **Phase 2:** Frigate-style training review queue capture policy is "all detected faces". We will need a queue size cap and a "dismiss group" action to keep the UI usable.
- **Phase 2:** Whether to add body re-ID (clothes/build matching) — deferred to Phase 2.5 if face-only proves limiting.
- **Phase 3:** Definition of "significant event" for clip-recording (currently planned as duration > 5 s, but may need refinement based on real action types).
- **Phase 4:** Multi-person reference resolution in NL queries ("what was he doing yesterday" when multiple persons exist) — likely solved by requiring explicit names in queries.

## 9. Forward-pointers

- **Phase 2 design doc:** TBD, future session — covers `face.py`, `gallery.py`, `apps/enroll_web.py`, SQLite schema for persons + faces + training candidates, Frigate-style review queue UI.
- **Phase 3 design doc:** TBD — covers `pose.py`, `events.py`, schema for events + clips, thumb/clip capture policy.
- **Phase 4 design doc:** TBD — covers `query.py`, `apps/ask.py`, Claude API integration.

---

*End of Phase 1 design.*

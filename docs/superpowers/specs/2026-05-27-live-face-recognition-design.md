# Live Face Recognition Overlay — Design

**Date:** 2026-05-27
**Status:** Drafted, awaiting user review
**Depends on:** [Face recognition](./2026-05-26-face-recognition-design.md)

---

## 1. Goal

Display recognized names (or "Unknown") **on the live video feed** above each detected person's bounding box, in real time. User can toggle the feature on/off from a sidebar checkbox.

Non-goals:

- Per-frame recognition (would tank FPS — runs once per ~1s per camera).
- Tracking-based recognition (no ByteTrack in live pipeline yet — Phase B).
- Per-person colors.
- Confidence slider in UI (uses backend's `match_high`/`match_low` thresholds).
- Recognition on non-`person` classes.

## 2. Decisions (resolved in brainstorm)

| # | Decision | Choice |
|---|---|---|
| 1 | Run frequency | Throttled: ~1 second per camera (`live_recognition_interval`) |
| 2 | Cache window | Cached matches valid for 3 seconds; stale matches dropped |
| 3 | Unknown handling | Label as `"Unknown"` |
| 4 | Color coding | green (high ≥0.55), yellow (mid 0.40-0.55), gray (unknown) |
| 5 | Toggle | Sidebar checkbox "Recognize faces", default OFF |
| 6 | Transport | New `recognize: bool` field in WS packet header; result gains a `faces` array |
| 7 | Scope of recognition | Only `person` detections from YOLO; ignore other classes |

## 3. Architecture

```
ws frame arrives
   │
   ▼
yolo infer (existing) → result {boxes: [...]}
   │
   ▼  (only if header.recognize == true and result has person boxes)
LiveRecognizer.annotate(camera_id, frame_bgr, person_boxes):
   ├── if (now - cam.last_run_ts) >= interval:
   │      for each person box:
   │        crop frame
   │        FaceEngine.detect_and_embed(crop)
   │        Gallery.match(embedding) → name + score
   │        push to cam.cached
   │      cam.last_run_ts = now
   └── return cam.cached (filter stale > ttl)
   │
   ▼
result["faces"] = [{bbox, name, score, kind}]
   │
   ▼
ws.send_json(result)
   │  (also still triggers recorder_manager.feed, unchanged)
```

The throttling means: if user is recognizing at 15 fps, only ~1 of those 15 frames actually runs face engine. The other 14 reuse the cached result. Cache TTL is 3s so a 1s recognition cycle survives a brief miss (face went off-camera for a moment, then came back).

## 4. Components

### 4.1 Backend

**`backend/app/recognition/live.py`** — new file. `LiveRecognizer`.

```python
@dataclass(frozen=True)
class FaceMatch:
    bbox: tuple[int, int, int, int]
    name: str          # "Alice" | "Unknown"
    score: float
    kind: str          # "high" | "mid" | "unknown"


@dataclass
class _CamState:
    last_run_ts: float = 0.0
    cached: list[FaceMatch] = field(default_factory=list)
    cached_at: float = 0.0


class LiveRecognizer:
    def __init__(self, engine: FaceEngine, gallery: Gallery) -> None: ...
    def annotate(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        person_boxes: list[dict],   # YOLO 'detect' boxes (normalized x,y,w,h)
    ) -> list[FaceMatch]: ...
```

- One singleton instance lives in `app.recognition.__init__` next to `recognition_worker`.
- Per-camera state held in `dict[str, _CamState]`.
- `annotate` decides "run now or reuse cache" based on `now - cam.last_run_ts >= settings.live_recognition_interval`.
- When running: iterates `person_boxes`, crops the frame at each person bbox's pixel coords (normalized → multiply by frame W/H), passes the crop into `FaceEngine.detect_and_embed`. For every face the engine finds, matches against gallery and builds a `FaceMatch`.
- `kind` derived from score: `high` if ≥ `match_high`, `mid` if ≥ `match_low`, else `unknown`.
- After computing, stores result in `cam.cached`, updates `cam.cached_at`.
- Reads always return `cached` but filter out entries older than `live_recognition_cache_ttl` seconds.

**`backend/app/api/stream.py`** — minimal edit. After building `result` and *after* `recorder_manager.feed(...)`:

```python
if fp.header.get("recognize") and result.get("type") == "detect":
    boxes = result.get("boxes", [])
    person_boxes = [b for b in boxes if b.get("label") == "person"]
    if person_boxes:
        matches = live_recognizer.annotate(camera_id, img, person_boxes)
        result["faces"] = [asdict(m) for m in matches]
```

Same scope check for `pose` mode (people have boxes).

**Config additions in `backend/app/config.py`:**

```python
live_recognition_interval: float = 1.0     # seconds between runs per camera
live_recognition_cache_ttl: float = 3.0    # seconds a cached match stays valid
```

### 4.2 Frontend

**`frontend/src/state/appState.ts`** — add:
- `recognizeFaces: boolean` (default `false`)
- `setRecognizeFaces`

**`frontend/src/pages/LivePage.tsx`** — add a checkbox under the "Controls" section:

```tsx
<label style={{ display: 'block', margin: '4px 0' }}>
  <input type="checkbox" checked={recognizeFaces}
    onChange={(e) => setRecognizeFaces(e.target.checked)} />
  Recognize faces
</label>
```

**`frontend/src/stream/useFrameSender.ts`** — extend the header builder. For yolo modes:

```ts
return { camera_id: s.cameraId, conf: s.yoloConf, recognize: s.recognizeFaces };
```

For sam3 modes the field is harmless to include or omit; keep simple by always including.

**`frontend/src/types.ts`** — extend types:

```ts
export interface FaceMatch {
  bbox: [number, number, number, number];   // (x1, y1, x2, y2) source pixels
  name: string;                              // "Alice" | "Unknown"
  score: number;
  kind: 'high' | 'mid' | 'unknown';
}

// detect / pose result types gain an optional faces field:
export interface DetectResult {
  type: 'detect';
  frame_id: number;
  ms: number;
  boxes: Box[];
  faces?: FaceMatch[];
}
```

**`frontend/src/overlay/OverlayCanvas.tsx`** — when rendering a `detect` or `pose` result and `faces` is present, draw each match's name label above the corresponding bbox:

- Box label rendered just above the upper-left corner of the face bbox
- Background color by `kind`:
  - `high` → green (`#22c55e`)
  - `mid` → yellow (`#eab308`)
  - `unknown` → gray (`#6b7280`)
- White text, monospace font, 12px
- Coordinates: convert face bbox (source pixels) → canvas pixels using the same scale OverlayCanvas already uses for YOLO boxes

## 5. Data flow on one frame (recognize=true)

```
1. Frontend sends frame, header { camera_id, conf, recognize: true }
2. Backend YOLO infer → result {type: 'detect', boxes: [...]}
3. Recorder.feed(img, detections)                   (unchanged)
4. live_recognizer.annotate(camera_id, img, person_boxes) → faces
5. result["faces"] = [...]
6. ws.send_json(result)
7. OverlayCanvas draws bounding boxes + face name labels
```

Throttling: step 4 only invokes face engine once per second per camera; intermediate frames reuse the cached `faces` list.

## 6. Configuration summary

All in `backend/app/config.py`. New fields:

| Constant | Default | Purpose |
|---|---|---|
| `live_recognition_interval` | `1.0` | Seconds between face-engine runs per camera |
| `live_recognition_cache_ttl` | `3.0` | Seconds a cached match stays valid |

(`match_high` and `match_low` are reused from the recognition feature.)

## 7. Testing

Two backend unit tests (frontend has no test infra):

1. **`tests/recognition/test_live.py::test_annotate_runs_on_first_call`**
   - Feed `LiveRecognizer` synthetic frame + person box. Stub `FaceEngine.detect_and_embed` to return one known face. Assert: returns one `FaceMatch` with `kind="high"`, name matches gallery.

2. **`tests/recognition/test_live.py::test_annotate_throttles_subsequent_calls`**
   - Call twice in rapid succession with same `camera_id`. Assert: engine called exactly once; second call returns same cached match without re-invoking engine.

A third test for stale eviction is nice-to-have but optional.

## 8. Out of scope (deferred)

- ByteTrack tracking IDs — would let us cache identity across frames per `track_id` and skip recognizing the same person repeatedly. Worth doing as Phase B.
- Multi-face per person box — current design assumes one face per `person` detection. If a person box ever contains multiple faces, we'll pick the first.
- Recognition for sam3 / obb / cls modes (those modes don't have clean person boxes today).
- Per-person colors.
- Showing the recognition timestamp ("last seen X seconds ago").

## 9. Performance expectations

- Recognize OFF: zero change to live FPS.
- Recognize ON, 1 person on screen: ~50–150ms once per second on CPU. With 15fps target, expect ~12–14 fps observed (slight dip from the 1s spike).
- Recognize ON, 5 people on screen: ~250–750ms once per second. Heavy enough that user may want to throttle further. Acceptable for v1; can add a frame-skip slider later if needed.

---

*End of design.*

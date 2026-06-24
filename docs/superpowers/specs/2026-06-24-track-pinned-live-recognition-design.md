# Track-Pinned Live Face Recognition — Design

**Date:** 2026-06-24
**Status:** Approved (brainstorming) — ready for implementation plan
**Supersedes (live path only):** the throttled per-frame approach in
[2026-05-27-live-face-recognition-design.md](2026-05-27-live-face-recognition-design.md);
this is the "Phase B" that doc deferred.

## Problem

With the **Recognize faces** toggle on, names are **slow to first appear** and
**lag behind people** as they move. The video itself stays smooth — the pain is
recognition latency and how names stay attached to people.

Root causes in the current code:

1. **No real tracking.** [`YoloEngine.infer`](../../../backend/app/inference/yolo.py)
   always calls `.predict()`, ignores the `tracking` header, and `Box` has no
   track ID. The Tracking toggle is a **no-op on the backend**. With no stable
   identity, every cycle re-detects, re-embeds and re-matches from scratch, and
   names are stitched onto people by a **cached array index**
   ([`stream.py`](../../../backend/app/api/stream.py) `person_box_idx`) that
   drifts the moment two people move. This is the core of the lag.
2. **Throttle + cache** (`live_recognition_interval` 1s, `live_recognition_cache_ttl`
   3s): a name updates at most once a second and can linger 3s stale.
3. **Cold model load.** InsightFace `buffalo_l` lazy-loads on the *first*
   recognition, not at startup → the first name is slow.
4. **Per-person 640×640 detection.** [`live.py`](../../../backend/app/recognition/live.py)
   feeds each person crop to [`FaceEngine.detect_and_embed`](../../../backend/app/recognition/engine.py),
   which runs the RetinaFace detector at `det_size=(640, 640)` — N people = N full
   detector passes per cycle, each upscaling a small crop. This cost is *why* the
   throttle is set so high.

The gallery match itself (a small brute-force matmul) is **not** a bottleneck.

## Goal

A recognized person's name is **pinned to their tracking identity** and reused
every frame at zero cost (no lag); a brand-new person is recognized **immediately**
against a **pre-warmed** model (fast first appearance); names never bleed between
people or between cameras.

Non-goals: changing the matching thresholds, frontend overlay changes, tracking
for non-detect modes, live-path DB writes.

## Constraints / context

- **Multiple concurrent cameras** are a real scenario → tracker state must be
  **isolated per camera**. We cannot use `.track(persist=True)` on the shared
  registry engine (its tracker state would bleed across cameras).
- Single YOLO model must stay warm in VRAM (no per-camera model instances).
- The WebSocket inference loop must **never block** on face work and must never
  crash a stream on a recognition error.

## Approach (chosen)

**Track-pinned recognition.** Drive Ultralytics' standalone `BYTETracker`
**once per camera** off the shared, warm `.predict()` output; stamp a `track_id`
onto each person box; keep a per-`(camera_id, track_id)` name cache; recognize a
track only while it is new / unconfident / due for refresh, **off the event loop**;
warm InsightFace at startup.

Alternatives considered and rejected:
- **Make per-frame recognition faster, no tracking** — treats symptoms; without
  identity persistence names still float with throttle lag.
- **IoU pseudo-tracking** — reinvents tracking we get for free and is weaker
  through occlusions than ByteTrack.

### Why a standalone per-camera `BYTETracker`

Ultralytics ships `ultralytics.trackers.BYTETracker`. `update(boxes, frame)`
takes a `Boxes`-like object (`.xywh/.conf/.cls`) and returns rows of
`[x1, y1, x2, y2, track_id, score, cls, idx]`, where `idx` is the original
detection index — so we can map track IDs back onto our serialized boxes by order.
This lets detection stay a single shared `.predict()` (one model in VRAM) while
tracking is a cheap per-camera layer.

**Gotcha:** the `STrack` ID counter is **class-global** (shared across all
tracker instances, and `BYTETracker.__init__` calls `reset_id()`). Therefore:
- The name cache is keyed by **`(camera_id, track_id)`**, so cross-camera ID
  collisions are harmless.
- Each camera's tracker is instantiated **once** (lazily, on first frame) and
  never recreated mid-session. The implementation must avoid letting a later
  camera's construction renumber an existing camera's live IDs (e.g. construct
  trackers once and don't call `reset_id()` after the first); captured as a task
  in the plan.

## Architecture

### New: `backend/app/inference/tracking.py`

`CameraTrackers` singleton (wired like `recorder_manager` / `live_recognizer`):

- Holds `dict[camera_id -> BYTETracker]`, lazily created on first sight.
- ByteTrack args loaded **once** from Ultralytics' bundled `bytetrack.yaml`
  (`yaml_load(check_yaml("bytetrack.yaml"))` → `IterableSimpleNamespace`).
- `assign(camera_id, result_boxes, frame) -> list[int | None]` — returns a track
  ID per detection, aligned to detection order (`None` where ByteTrack didn't
  confirm/return a track for that detection).
- `drop(camera_id)` — evict a camera's tracker on disconnect.
- All Ultralytics result/tensor objects stay inside this module + `yolo.py`;
  only plain `int | None` IDs leave.

### Changed: `backend/app/domain.py`

`Box` gains `track_id: int | None = None` (kept last; default preserves existing
construction sites).

### Changed: `backend/app/inference/yolo.py`

Only the **detect** task supports tracking (for now). When
`header.get("tracking")` and a `camera_id` are present, `infer()` calls
`CameraTrackers.assign(...)` over the raw `result.boxes` and passes the resulting
IDs into `_extract_boxes`, which writes `track_id` onto each `Box`. Other tasks
and the no-tracking path are unchanged. `track_id` is serialized into the JSON
result (harmless to the frontend; enables future color-by-track).

### Reworked: `backend/app/recognition/live.py`

From a frame-snapshot cache to a **(camera_id, track_id) → name cache + async
worker**.

Cache value per track:
```
name: str           # resolved person name or "Unknown"
score: float
kind: str           # "high" | "mid" | "unknown"
resolved_at: float  # last successful recognition
last_seen: float    # last frame this track appeared
next_attempt_at: float   # earliest time we may (re)recognize
```

Per-track state machine:
```
NEW ──recognize immediately──▶ PENDING ──worker result──▶ CONFIRMED (high)
                                   │                    └─▶ TENTATIVE (mid/unknown/no-face)
CONFIRMED ──refresh interval──▶ re-recognize (keeps showing old name meanwhile)
TENTATIVE ──backoff retry──────▶ re-recognize
any state ──unseen for track_ttl──▶ evicted
```

Lifecycle rules (the cadence that keeps the GPU and the loop free):
- **New track → recognize once, immediately** (no initial throttle). Warm model
  → name in tens of ms.
- **Confirmed (`high`) → stop re-recognizing**; reuse the pinned name every frame
  at zero cost. Refresh only every `live_recognition_refresh` to catch ID swaps,
  showing the existing name while the refresh runs (no flicker).
- **Tentative (`mid` / `unknown` / no face) → retry with backoff**
  (`live_recognition_retry`), not every frame.
- **Per-track dedupe**: at most one queued job per track (latest crop wins).
- **Global queue cap** `recognition_queue_max`: excess jobs dropped, not buffered.
- **Eviction**: entries unseen for `track_ttl` are dropped.

Public API:
```python
def annotate(
    camera_id: str,
    frame_bgr: np.ndarray,
    person_boxes: list[dict],   # each carrying its track_id
) -> dict[int, FaceMatch]:      # track_id -> current known name
```
- Runs in **O(1)** per frame: read cache for known names, update `last_seen`,
  enqueue face jobs for tracks that are new / unconfident / due, evict stale
  tracks. Never calls the face engine inline.
- Returns names currently known; a brand-new track is absent here and its name
  lands a frame or two later once the worker finishes.

Background **single** worker thread (one face inference at a time → no
onnxruntime concurrency risk):
```
job(camera_id, track_id, crop) → face_engine.detect_and_embed(crop)
                               → gallery.match → resolve person name (one DB read)
                               → write cache[(camera_id, track_id)] under lock
```
- Highest-`det_score` face in the crop wins (a person box bounds one person).
- Every job wrapped in try/except: on failure, log and mark the track
  `TENTATIVE` (retries on backoff). Never propagates to the WS loop.
- Started in `main.py` lifespan, joined on shutdown (mirrors `recognition_worker`).

A **dedicated live `FaceEngine`** instance is prepared with
`det_size=(live_det_size, live_det_size)` so the live path can use a smaller
detector (320²) without affecting the offline clip worker (stays 640²).

### Changed: `backend/app/api/stream.py`

Replace the `person_box_idx` mapping. After inference:
- Person boxes already carry `track_id` (from the detect path).
- `names = live_recognizer.annotate(camera_id, img, person_boxes)`.
- For each result box whose `track_id` is in `names`, append the name suffix to
  its label (same overlay mechanism as today).
- On WS disconnect, also `CameraTrackers.drop(camera_id)` and clear that camera's
  live cache (alongside the existing `recorder_manager.close`).

### Changed: `backend/app/main.py`

In `lifespan`, after `gallery.reload()`: **warm InsightFace** (load `buffalo_l` +
one dummy embed) and **start the live recognition worker**; stop/join it on
shutdown.

### Changed: `backend/app/config.py`

```python
# --- recognition (live) ---
live_recognition_refresh: float = 10.0   # re-confirm a CONFIRMED track this often
live_recognition_retry:   float = 1.5    # backoff before retrying a TENTATIVE track
track_ttl:                float = 5.0     # evict a track's cached name after this unseen gap
recognition_queue_max:    int   = 16     # max queued face jobs before dropping
live_det_size:            int   = 320    # InsightFace det_size for live crops (offline stays 640)
```
Remove the obsolete `live_recognition_interval` and `live_recognition_cache_ttl`.
Matching thresholds `match_high` (0.55) / `match_low` (0.40) are unchanged — only
the cadence changes.

## Data flow (tracking + recognize on)

```
frame → YOLO .predict() (shared, warm)
      → CameraTrackers[camera_id].assign() → track_id per person box
      → boxes stamped with track_id (also sent to frontend)
stream handler:
  names = live_recognizer.annotate(camera_id, frame, person_boxes)   # O(1): read cache, enqueue misses
  attach cached name to each box whose track_id is known
  → send result            ← never blocked on face work
background worker (always on):
  job(camera_id, track_id, crop) → face detect+embed → match → resolve name → write cache
```

## Edge cases

- **No track ID** (tracking off, or detection ByteTrack didn't confirm) → no live
  name for that box. Tracking is required for live names by design (documented);
  the old index hack is not reintroduced.
- **Multiple faces in one crop** → highest `det_score` wins.
- **ID swap after occlusion** → new ID = new track, recognized immediately; old ID
  evicts after `track_ttl`. Brief and self-healing.
- **Facing away / tiny distant face** → no face found → `TENTATIVE`, cheap backoff;
  resolves when the face becomes visible. Faces below the resolution floor
  (frame ≤640px) stay Unknown — a resolution limit, not a bug.
- **Empty / un-enrolled gallery** → Unknown, no crash.
- **Camera disconnect** → drop tracker + cache entries.

## Error handling

- Worker job failure → log + `TENTATIVE`, stream unaffected.
- Tracker error on a frame → `None` IDs, detections still served, no client error.
- Bounded queue; overflow drops oldest pending per track, keeps newest crop.
- Worker started/joined in lifespan; a dead worker is logged.

## Testing

- **`tracking.py`** — two synthetic detections across consecutive frames: assert
  stable IDs across frames and per-camera isolation (two cameras' IDs don't
  collide in the cache). Pure, no GPU.
- **`LiveRecognizer`** — fake face engine + fake gallery (no CUDA), injected clock:
  (a) new track enqueues exactly one job; (b) confirmed track reuses cached name
  with no new job until `refresh`; (c) tentative retries only after `retry`
  backoff; (d) eviction after `track_ttl`; (e) queue cap drops excess.
- **`stream.py` attach** — boxes with track IDs + a cache dict: names land on the
  correct boxes (multi-person, no drift).
- Existing recognition/recording tests stay green; `ruff check app` clean.

## Out of scope (YAGNI)

- Frontend overlay changes (track_id is sent but unused for now).
- Tracking for pose / seg / obb modes (detect only).
- Live-path DB writes (live stays display-only; the clip worker still owns
  sightings / review-queue).

## Expected effect on the reported symptoms

- **First appearance:** (cold load + up to 1s throttle) → ~one warm inference.
- **Lag:** eliminated — a confirmed name is glued to the track instead of being
  recomputed and re-mapped each cycle.
- **Multi-person mis-association:** eliminated — names attach by `track_id`, not a
  drifting array index.

# Face Recognition — Design

**Date:** 2026-05-26
**Status:** Drafted from brainstorming session, awaiting user review
**Depends on:** [Motion-clip recording](./2026-05-26-motion-clip-recording-design.md) (Phase 1)

---

## 1. Goal

Recognize people in recorded clips and let the user search "show me clips containing Alice." Registration is hybrid:

- **Upload** a name + one or more photos to bootstrap a person, OR
- **Label** unknown faces that show up in clips via a Frigate-style review queue.

Recognition happens **after** a clip is closed — not on the live stream. This keeps live inference FPS untouched.

Non-goals:

- No live name overlays during inference (deferred — would need tracking to be efficient).
- No body re-identification (clothes/build matching).
- No retention or cleanup policy.
- No authentication — local app.
- No multi-person dedup beyond cosine similarity within a clip.

## 2. Decisions (resolved in brainstorm)

| # | Decision | Choice |
|---|---|---|
| 1 | Pipeline integration | Tag clips after-the-fact via background worker; no impact on live FPS |
| 2 | Face library | InsightFace (`buffalo_l` ONNX model), GPU-accelerated on RTX 4070 |
| 3 | Registration model | Hybrid: upload-based + Frigate-style review queue |
| 4 | Per-face artifacts | Face crop JPG + 512-dim embedding (original upload discarded) |
| 5 | Search UX | Person → list of clips containing them |
| 6 | Storage | SQLite (`backend/data/app.db`), embeddings as `BLOB` |
| 7 | Recognition sampling | 1 frame per second from each clip (~15 samples for a 15s clip) |
| 8 | Match thresholds | cosine ≥ 0.55 → high; < 0.40 → unknown (queue); between → queue with suggested match |
| 9 | UI placement | Separate routes: `/` (Live), `/people`, `/review` via `react-router-dom` |

## 3. Architecture

```
clip closed (MotionRecorder)
        │  on_clip_closed(clip_path)              ← new callback
        ▼
   RecognitionWorker.enqueue(clip_path)
        │  (background thread)
        ▼
  ┌──────────────────────────────────────┐
  │ For each sampled frame (~1/sec):     │
  │  1. InsightFace detect → face boxes  │
  │  2. Crop + align each face           │
  │  3. Compute 512-d embedding          │
  │  4. Gallery.match(emb) cosine        │
  │     ≥0.55         → ClipSighting     │
  │     0.40–0.55     → ReviewQueue +    │
  │                     suggested_id     │
  │     <0.40         → ReviewQueue      │
  └──────────────────────────────────────┘
```

The recognition path runs only in the worker thread. Inference and recording are unchanged.

Registration runs synchronously inside the FastAPI request handler for `POST /api/people` — for ≤10 images this is fast enough (~1s on GPU).

## 4. Data model (SQLite)

Single file: `backend/data/app.db`. Created on app startup if missing.

```sql
CREATE TABLE persons (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT UNIQUE NOT NULL,
  created_at  REAL NOT NULL
);

CREATE TABLE faces (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id    INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  crop_path    TEXT NOT NULL,    -- relative to backend/data/, e.g. "faces/3/abc.jpg"
  embedding    BLOB NOT NULL,    -- 512 float32 = 2048 bytes
  source       TEXT NOT NULL,    -- 'upload' | 'queue'
  created_at   REAL NOT NULL
);
CREATE INDEX faces_person_idx ON faces(person_id);

CREATE TABLE review_queue (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  crop_path       TEXT NOT NULL,
  embedding       BLOB NOT NULL,
  source_clip     TEXT,
  suggested_id    INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  suggested_score REAL,
  status          TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'labeled' | 'dismissed'
  created_at      REAL NOT NULL
);
CREATE INDEX review_status_idx ON review_queue(status, created_at);

CREATE TABLE clip_sightings (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_path   TEXT NOT NULL,
  person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  confidence  REAL NOT NULL,
  frame_ts    REAL NOT NULL,    -- seconds into the clip where best match occurred
  created_at  REAL NOT NULL,
  UNIQUE(clip_path, person_id)
);
CREATE INDEX sightings_clip_idx ON clip_sightings(clip_path);
CREATE INDEX sightings_person_idx ON clip_sightings(person_id);
```

**On-disk face layout:**

```
backend/data/
├── app.db
├── clips/                       (from Phase 1)
└── faces/
    ├── <person_id>/
    │   ├── upload_<uuid>.jpg
    │   └── queue_<uuid>.jpg
    └── ...
```

Crops are 112×112 JPEGs (InsightFace's standard alignment output). ~5–15 KB each.

## 5. Backend components

### 5.1 New files

**`backend/app/recognition/__init__.py`** — exports `face_engine`, `gallery`, `worker` singletons.

**`backend/app/recognition/db.py`** — SQLite wrapper.

- `get_conn() -> sqlite3.Connection` — one connection per thread via `threading.local`.
- `init_schema()` — idempotent CREATE TABLE statements; called from FastAPI lifespan.
- CRUD helpers: `add_person`, `list_persons`, `delete_person`, `add_face`, `list_faces_for`, `load_all_embeddings`, `enqueue_review`, `list_queue`, `update_queue_status`, `add_sighting`, `clips_for_person`.

**`backend/app/recognition/engine.py`** — `FaceEngine`.

```python
class FaceEngine:
    def __init__(self, device: str = "cuda") -> None: ...
    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[FaceResult]: ...

@dataclass(frozen=True)
class FaceResult:
    crop_bgr: np.ndarray     # 112×112×3 aligned crop
    embedding: np.ndarray    # 512-d float32, L2-normalized
    bbox: tuple[int, int, int, int]
    det_score: float
```

- Lazy-loads `insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])` on first call.
- Models auto-download to `~/.insightface/models/` on first use.

**`backend/app/recognition/gallery.py`** — `Gallery`.

```python
class Gallery:
    def __init__(self) -> None:
        self._matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self._person_ids: list[int] = []
        self.reload()

    def reload(self) -> None:
        # load all (person_id, embedding) pairs from DB into _matrix
        ...

    def match(self, query: np.ndarray) -> tuple[int | None, float]:
        # cosine via dot product (assumes L2-normalized embeddings)
        if self._matrix.shape[0] == 0:
            return None, 0.0
        scores = self._matrix @ query
        idx = int(np.argmax(scores))
        return self._person_ids[idx], float(scores[idx])

    def add(self, person_id: int, embedding: np.ndarray) -> None:
        # append to in-memory matrix; DB row is added by the caller
        ...
```

Thread-safe: a single `threading.Lock` around `reload`, `match`, `add`. Brute-force is O(N) at ~512 ops/face; trivial up to 100k faces.

**`backend/app/recognition/worker.py`** — `RecognitionWorker`.

```python
class RecognitionWorker:
    def __init__(self) -> None:
        self._q: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def enqueue(self, clip_path: Path) -> None: ...
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(path)
            except Exception:
                log.exception("recognition failed for %s", path)
            finally:
                self._q.task_done()

    def _process(self, path: Path) -> None:
        # cv2.VideoCapture sample 1 fps; detect+embed; per-face match → sighting or queue
        ...
```

Started by FastAPI `lifespan` startup, stopped on shutdown.

### 5.2 New API endpoints — `backend/app/api/recognition.py`

| Method | Path | Body / Query | Response |
|---|---|---|---|
| `GET` | `/api/people` | — | `[{id, name, face_count, latest_face_url}]` |
| `POST` | `/api/people` | multipart: `name` + `files[]` (1–10 images) | `{id, name, face_count}` |
| `DELETE` | `/api/people/{id}` | — | `{ok: true}` |
| `GET` | `/api/people/{id}` | — | `{id, name, faces: [{id, crop_url, source, created_at}]}` |
| `GET` | `/api/people/{id}/clips` | — | `[{clip_path, mp4_url, jpg_url, confidence, frame_ts}]` |
| `GET` | `/api/review` | `?limit=50` | `[{id, crop_url, suggested: {id, name, score} \| null, source_clip, created_at}]` |
| `POST` | `/api/review/{id}/label` | `{person_id}` OR `{new_name}` | `{person_id}` |
| `POST` | `/api/review/{id}/dismiss` | — | `{ok: true}` |

Also: `app.mount("/faces", StaticFiles(directory=settings.faces_dir))` — same pattern as `/clips/`.

### 5.3 Existing files — minimal edits

- **`backend/app/recording/recorder.py`** — `MotionRecorder.__init__` adds `on_clip_closed: Callable[[Path], None] | None = None`. Track the **first** clip path of the current event (`self._event_first_clip`); invoke the callback once in `_close_writer` at the end of an event (when transitioning to IDLE), not on rotation. Pass the first clip's path.

- **`backend/app/recording/manager.py`** — When creating a `MotionRecorder`, pass `on_clip_closed=recognition_worker.enqueue`.

- **`backend/app/main.py`** — In `lifespan`: `db.init_schema()`, `gallery.reload()`, `recognition_worker.start()` on startup; `recognition_worker.stop()` on shutdown. Mount `/faces` StaticFiles. Include `recognition` router.

- **`backend/app/config.py`** — add `faces_dir: Path = BACKEND_ROOT / "data" / "faces"`, `db_path: Path = BACKEND_ROOT / "data" / "app.db"`, thresholds `match_high: float = 0.55`, `match_low: float = 0.40`, `recognition_sample_fps: float = 1.0`.

### 5.4 Threading model

- FastAPI request handlers (synchronous endpoints): use per-thread sqlite3 connections via `threading.local`.
- `RecognitionWorker._run()`: its own thread; its own connection; calls `Gallery.match` (lock-protected) and `Gallery.add` only when a queue item is labeled.
- `Gallery` is a process-wide singleton with a lock around the embedding matrix.

## 6. Frontend changes

Install `react-router-dom`.

### 6.1 Routing shell

**`frontend/src/main.tsx`** wraps with `<BrowserRouter>`.

**New `frontend/src/App.tsx`** becomes a thin layout: a top nav with three links (`Live`, `People`, `Review`) + `<Outlet />`. The previous App body moves to `pages/LivePage.tsx` unchanged.

Routes:
- `/` → `<LivePage />`
- `/people` → `<PeoplePage />`
- `/people/:id` → `<PersonDetailPage />`
- `/review` → `<ReviewPage />`

### 6.2 New pages

**`pages/LivePage.tsx`** — the existing live view (current App body), no behavior change.

**`pages/PeoplePage.tsx`**

- Lists all people from `GET /api/people` as cards (thumbnail + name + face count)
- "Add person" button opens a modal: `<input type="text">` for name + `<input type="file" multiple accept="image/*">` → submits `multipart/form-data` to `POST /api/people` → on success, refreshes the list and closes
- Clicking a card navigates to `/people/:id`

**`pages/PersonDetailPage.tsx`**

- Top: person name, delete button (with confirm)
- Left pane: gallery of their face crops (thumbnails)
- Right pane: list of clips where they appeared (reuses the same card style as `ClipsPanel`), clicking opens the clip modal

**`pages/ReviewPage.tsx`**

- Polls `GET /api/review?limit=50` every 10s
- Grid of cards, each showing: crop thumbnail, suggested-match badge ("Maybe Alice 0.48") if any, four buttons: `Confirm`, `Label new...`, `Label existing...`, `Dismiss`
- `Confirm` (only if suggested) → `POST /api/review/{id}/label` with `{person_id: suggested.id}`
- `Label new` → name input prompt → POST with `{new_name}`
- `Label existing` → picker dropdown of all people → POST with `{person_id}`
- `Dismiss` → `POST /api/review/{id}/dismiss`

### 6.3 Shared components

**`frontend/src/clips/ClipsPanel.tsx`** — split into:
- `useClipsList(cameraId, peopleId?)` — fetching hook
- `<ClipsList clips={...} onOpen={...} />` — pure rendering

Both PeoplePage and the existing live view can render the same component with different data sources.

## 7. Data flow (one motion event end-to-end)

```
1. Live: WS → YOLO inference → result back to FE (unchanged)
2. recorder_manager.feed → MotionRecorder records clip(s)
3. Motion ends + post-roll → MotionRecorder._close_writer:
   a. release VideoWriter
   b. on_clip_closed(self._event_first_clip)
4. Manager-supplied callback = recognition_worker.enqueue(path)
   ─── background thread ───
5. Worker._process(path):
   a. cv2.VideoCapture; sample frames at 1 fps
   b. FaceEngine.detect_and_embed(frame) per sample
   c. For each face emb:
        person_id, score = Gallery.match(emb)
        if score >= 0.55:
            best[person_id] = max(best[person_id], (score, frame_ts))
        elif score >= 0.40:
            enqueue_review(crop, emb, source_clip=path, suggested_id=person_id, score)
        else:
            enqueue_review(crop, emb, source_clip=path, suggested=None)
   d. For each (person_id, (score, ts)) in best:
        add_sighting(path, person_id, score, ts)
```

User actions on the people / review pages mutate `persons`, `faces`, `review_queue` directly; after any face mutation, `gallery.reload()` is called.

## 8. Configuration summary

All in `backend/app/config.py`. New fields:

| Constant | Default | Purpose |
|---|---|---|
| `faces_dir` | `backend/data/faces` | Where face crops are stored |
| `db_path` | `backend/data/app.db` | SQLite database file |
| `match_high` | `0.55` | Cosine ≥ this → confirmed match |
| `match_low` | `0.40` | Cosine < this → unknown (queue without suggestion) |
| `recognition_sample_fps` | `1.0` | How often to sample frames from a clip |
| `recognition_device` | `"cuda"` | InsightFace device (`cuda` or `cpu`) |

## 9. Testing

Five offline test files (no GPU required; engine is stubbed in tests).

1. **`tests/recognition/test_db.py`** — schema init idempotent; `add_person` → `list_persons`; cascade delete removes faces + sightings; embedding BLOB round-trip preserves bytes.

2. **`tests/recognition/test_gallery.py`** — Gallery starts empty; `match()` on empty returns `(None, 0.0)`; after adding two synthetic L2-normalized vectors, query close to vector A returns A's `person_id` with high cosine; orthogonal query returns low cosine; `reload()` picks up new DB rows.

3. **`tests/recognition/test_worker.py`** — stub `FaceEngine.detect_and_embed` to return one known face + one unknown per frame; enqueue a fake clip path (with a stub `cv2.VideoCapture` returning N black frames); assert one `clip_sightings` row + one `review_queue` row appear after processing.

4. **`tests/recognition/test_api.py`** — FastAPI `TestClient`:
   - `POST /api/people` with a fixture image works; `GET /api/people` shows it with `face_count=1`
   - `POST /api/review/{id}/label` with `{person_id}` moves the row to faces and flips status
   - `DELETE /api/people/{id}` cascades and the gallery in-memory matrix shrinks

5. **`tests/recording/test_recorder_callback.py`** — extends Phase 1 tests: when motion lasts long enough to rotate the 15s clip twice, the `on_clip_closed` callback is invoked **once**, with the **first** clip's path (not the rotation clips).

Fixtures: one small face JPG (`tests/fixtures/face.jpg`, public-domain, ~10KB).

## 10. Out of scope

- Live name overlays during inference.
- Body re-identification.
- Retention / cleanup of old crops, queue, sightings.
- Authentication.
- Multi-stream face recognition pulled in real time from RTSP.
- A "merge two persons" operation (handled manually via DELETE + re-label for v1).

## 11. Future work

- **Phase B: live overlay** — add tracking IDs to the live pipeline, cache `track_id → name` lookups, draw the name above each person's bounding box in the overlay canvas.
- **Phase C: smart queue dedup** — when many similar unknown faces land in the queue, cluster them (cosine ≤ 0.2 apart) and present as a group: "12 faces, all look like the same person — label them all at once."
- **Phase D: face-bound clip search** — combine the existing `cam-front 2026-05-26` browse with person filters: "all clips of Alice on cam-kitchen this week."
- **Re-embed pipeline** — when InsightFace ships a better model, run a re-embed pass over all `crop_path`s without losing person labels.

---

*End of design.*

# Live Face Recognition Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a toggleable live overlay that draws recognized names above each person's bounding box on the camera feed.

**Architecture:** New `LiveRecognizer` singleton runs **inside the WS handler**, after YOLO inference and after the recorder hook. It's throttled to once per ~1 second per `camera_id` and caches results for 3 seconds. Detection results gain an optional `faces` array. Frontend has a sidebar checkbox; when enabled, the WS packet header includes `recognize: true` and the `OverlayCanvas` draws colored name labels on top of person boxes.

**Tech Stack:** Python 3.12 / FastAPI / InsightFace (already loaded). React + TypeScript + Canvas 2D.

**Spec:** [docs/superpowers/specs/2026-05-27-live-face-recognition-design.md](../specs/2026-05-27-live-face-recognition-design.md)

**Refinement vs spec:** Spec said `FaceMatch.bbox: tuple[int,int,int,int]` in source pixels. The frontend `OverlayCanvas` already expects normalized [0,1] coords (matches YOLO `Box.x/y/w/h` convention). Implementation emits normalized `(x, y, w, h)` so overlay rendering is symmetric with `drawBoxes`. Same data, different units.

---

## File structure

**New (backend):**
- `backend/app/recognition/live.py` — `LiveRecognizer` + `FaceMatch` + per-camera `_CamState`
- `backend/tests/recognition/test_live.py` — 2 tests

**Modified (backend):**
- `backend/app/config.py` — 2 new settings
- `backend/app/recognition/__init__.py` — export `live_recognizer` singleton
- `backend/app/api/stream.py` — call `live_recognizer.annotate` when `header.recognize` is truthy

**Modified (frontend):**
- `frontend/src/types.ts` — add `FaceMatch` interface, extend `DetectResult` / `Person` result with optional `faces`
- `frontend/src/state/appState.ts` — add `recognizeFaces` + setter
- `frontend/src/stream/useFrameSender.ts` — include `recognize` in header
- `frontend/src/pages/LivePage.tsx` — add "Recognize faces" checkbox
- `frontend/src/overlay/OverlayCanvas.tsx` — call new `drawFaceNames` helper after YOLO boxes/pose

**New (frontend):**
- `frontend/src/overlay/drawFaceNames.ts` — pure draw function

---

## Task 1: Config

**Files:** Modify `backend/app/config.py`.

- [ ] **Step 1:** Add two fields to `Settings` under the `# --- recognition ---` block (after `recognition_device`):

```python
    live_recognition_interval: float = 1.0     # seconds between runs per camera
    live_recognition_cache_ttl: float = 3.0    # seconds a cached match stays valid
```

- [ ] **Step 2:** Run `cd backend && .venv/Scripts/python.exe -m pytest -q` — expect all 38 tests still pass.

- [ ] **Step 3:** Commit.

```bash
git add backend/app/config.py
git commit -m "feat(recognition): live overlay config"
```

---

## Task 2: LiveRecognizer — failing tests

**Files:** Create `backend/tests/recognition/test_live.py`.

- [ ] **Step 1:** Write the test file:

```python
"""Tests for LiveRecognizer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.recognition import db as dbmod
from app.recognition.engine import FaceResult


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    import dataclasses
    patched = dataclasses.replace(
        cfg.settings,
        db_path=tmp_path / "app.db",
        faces_dir=tmp_path / "faces",
        live_recognition_interval=1.0,
        live_recognition_cache_ttl=3.0,
    )
    monkeypatch.setattr(cfg, "settings", patched)
    monkeypatch.setattr(dbmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()
    return tmp_path


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _person_box() -> dict:
    return {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_annotate_runs_on_first_call(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=alice_emb,
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    matches = lr.annotate("cam-A", _frame(), [_person_box()])
    assert engine.detect_and_embed.called
    assert len(matches) == 1
    assert matches[0].name == "Alice"
    assert matches[0].kind == "high"
    # Normalized bbox: ints 10/640..110/640, etc.
    x, y, w, h = matches[0].bbox
    assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0
    assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_annotate_throttles_subsequent_calls(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=alice_emb,
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    first = lr.annotate("cam-A", _frame(), [_person_box()])
    second = lr.annotate("cam-A", _frame(), [_person_box()])
    # Second call within throttle window: engine not re-invoked, cached returned.
    assert engine.detect_and_embed.call_count == 1
    assert len(second) == 1
    assert second[0].name == first[0].name


def test_unknown_face_gets_unknown_kind(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    # gallery has Alice
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    # Stub returns a very different embedding
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=_unit(999),
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    matches = lr.annotate("cam-A", _frame(), [_person_box()])
    assert len(matches) == 1
    assert matches[0].name == "Unknown"
    assert matches[0].kind == "unknown"
```

- [ ] **Step 2:** Run; expect ImportError.

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/recognition/test_live.py -v
```

---

## Task 3: LiveRecognizer — implementation

**Files:** Create `backend/app/recognition/live.py`. Modify `backend/app/recognition/__init__.py`.

- [ ] **Step 1:** Create `backend/app/recognition/live.py`:

```python
"""Throttled per-camera live face recognition.

Cropping happens here, not in the recognition worker — the live path needs to
operate on a single shared frame across many person detections, while the
worker reads back a saved video.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceMatch:
    bbox: tuple[float, float, float, float]   # normalized (x, y, w, h) of face crop
    name: str                                  # "Alice" | "Unknown"
    score: float
    kind: str                                  # "high" | "mid" | "unknown"


@dataclass
class _CamState:
    last_run_ts: float = 0.0
    cached: list[FaceMatch] = field(default_factory=list)
    cached_at: float = 0.0


class LiveRecognizer:
    def __init__(self, engine: FaceEngine, gallery: Gallery) -> None:
        self._engine = engine
        self._gallery = gallery
        self._state: dict[str, _CamState] = {}

    def annotate(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        person_boxes: list[dict],
    ) -> list[FaceMatch]:
        cam = self._state.setdefault(camera_id, _CamState())
        now = time.time()

        if now - cam.last_run_ts < settings.live_recognition_interval:
            # Throttled — return cache, dropping anything stale.
            return self._filter_stale(cam, now)

        cam.last_run_ts = now
        cam.cached = list(self._run(frame_bgr, person_boxes))
        cam.cached_at = now
        return cam.cached

    def _run(self, frame_bgr: np.ndarray, person_boxes: list[dict]):
        h, w = frame_bgr.shape[:2]
        for pb in person_boxes:
            x1 = int(round(float(pb["x"]) * w))
            y1 = int(round(float(pb["y"]) * h))
            x2 = int(round((float(pb["x"]) + float(pb["w"])) * w))
            y2 = int(round((float(pb["y"]) + float(pb["h"])) * h))
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            faces = self._engine.detect_and_embed(crop)
            for face in faces:
                pid, score = self._gallery.match(face.embedding)
                kind = self._classify(score, pid)
                if kind == "unknown":
                    name = "Unknown"
                else:
                    from app.recognition import db as dbmod
                    person = dbmod.get_person(pid) if pid is not None else None
                    name = person["name"] if person else "Unknown"

                # face.bbox is in CROP-local pixels. Convert to source-frame pixels,
                # then normalize.
                fx1, fy1, fx2, fy2 = face.bbox
                ax1 = x1 + fx1
                ay1 = y1 + fy1
                ax2 = x1 + fx2
                ay2 = y1 + fy2
                yield FaceMatch(
                    bbox=(ax1 / w, ay1 / h, (ax2 - ax1) / w, (ay2 - ay1) / h),
                    name=name,
                    score=float(score),
                    kind=kind,
                )

    @staticmethod
    def _classify(score: float, pid: int | None) -> str:
        if pid is None:
            return "unknown"
        if score >= settings.match_high:
            return "high"
        if score >= settings.match_low:
            return "mid"
        return "unknown"

    def _filter_stale(self, cam: _CamState, now: float) -> list[FaceMatch]:
        if now - cam.cached_at > settings.live_recognition_cache_ttl:
            return []
        return cam.cached
```

- [ ] **Step 2:** Update `backend/app/recognition/__init__.py` — add `live_recognizer` next to existing singletons. Final contents:

```python
"""Recognition subsystem."""
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.live import LiveRecognizer
from app.recognition.worker import RecognitionWorker

dbmod.init_schema()
face_engine = FaceEngine()
gallery = Gallery()
recognition_worker = RecognitionWorker(engine=face_engine, gallery=gallery)
live_recognizer = LiveRecognizer(engine=face_engine, gallery=gallery)

__all__ = ["face_engine", "gallery", "recognition_worker", "live_recognizer"]
```

- [ ] **Step 3:** Run tests:

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/recognition/test_live.py -v
```
Expected: 3 passed.

- [ ] **Step 4:** Run full backend suite to confirm no regressions:

```bash
cd backend && .venv/Scripts/python.exe -m pytest -q
```
Expected: 41 passed (38 prior + 3 live).

- [ ] **Step 5:** Commit.

```bash
git add backend/app/recognition/live.py backend/app/recognition/__init__.py backend/tests/recognition/test_live.py
git commit -m "feat(recognition): LiveRecognizer with per-camera throttle + cache"
```

---

## Task 4: Wire LiveRecognizer into `/ws`

**Files:** Modify `backend/app/api/stream.py`.

- [ ] **Step 1:** Add to imports (around the top with the other `from app.recognition...` lines — currently `from app.recognition import recorder_manager` won't exist, the existing import is via `app.recording`. Add the live one alongside the recorder import).

Find the import line:

```python
from app.recording import recorder_manager
```

(Or whatever line currently imports `recorder_manager`.) Just after it, add:

```python
from app.recognition import live_recognizer
from dataclasses import asdict
```

- [ ] **Step 2:** After the `if camera_id: recorder_manager.feed(...)` block — and before `frame_id += 1` — add the live recognition path:

```python
            if camera_id and isinstance(fp.header, dict) and fp.header.get("recognize"):
                person_boxes = [b for b in _detections_from_result(result) if b.get("label") == "person"]
                if person_boxes:
                    matches = live_recognizer.annotate(camera_id, img, person_boxes)
                    result["faces"] = [asdict(m) for m in matches]
```

Note: `_detections_from_result` is the existing helper that flattens both `detect.boxes` and `pose.people[*].box`.

- [ ] **Step 3:** Run all backend tests:

```bash
cd backend && .venv/Scripts/python.exe -m pytest -q
```
Expected: 41 passed.

- [ ] **Step 4:** Commit.

```bash
git add backend/app/api/stream.py
git commit -m "feat(recognition): wire live_recognizer into /ws"
```

---

## Task 5: Frontend — types + Zustand state

**Files:** Modify `frontend/src/types.ts`, `frontend/src/state/appState.ts`.

- [ ] **Step 1:** In `frontend/src/types.ts`, add the `FaceMatch` interface and extend the result types. Open the file and add (anywhere — keep types grouped logically):

```ts
export interface FaceMatch {
  bbox: [number, number, number, number];   // normalized x, y, w, h
  name: string;
  score: number;
  kind: 'high' | 'mid' | 'unknown';
}
```

Then find the type definitions for the `detect` and `pose` result members (likely in `ServerMessage` in `frontend/src/stream/protocol.ts`, or wherever `lastResult.type === 'detect'` is keyed). Each must gain an optional `faces?: FaceMatch[]`.

If the result types live in `protocol.ts`, edit them there. Otherwise edit `types.ts`. The implementer should locate the exact place by searching for `type: 'detect'`.

- [ ] **Step 2:** Modify `frontend/src/state/appState.ts`. Add a `recognizeFaces` field with default `false` and a setter `setRecognizeFaces`. Insert these alongside other booleans like `mirror` and `paused`:

In the interface:
```ts
  recognizeFaces: boolean;
  setRecognizeFaces: (b: boolean) => void;
```

In the `create<AppState>` initial object:
```ts
  recognizeFaces: false,
```

In the setter list:
```ts
  setRecognizeFaces: (recognizeFaces) => set({ recognizeFaces }),
```

- [ ] **Step 3:** Verify build:

```bash
cd frontend && npm run build
```
Expected: success.

- [ ] **Step 4:** Commit.

```bash
git add frontend/src/types.ts frontend/src/state/appState.ts frontend/src/stream/protocol.ts 2>/dev/null
git commit -m "feat(frontend): FaceMatch type + recognizeFaces state"
```

(The `protocol.ts` add is optional — if no changes there, the stage is a no-op.)

---

## Task 6: Frontend — send `recognize` in header + checkbox UI

**Files:** Modify `frontend/src/stream/useFrameSender.ts`, `frontend/src/pages/LivePage.tsx`.

- [ ] **Step 1:** In `frontend/src/stream/useFrameSender.ts`, subscribe to `recognizeFaces` and include it in the header.

Add to the Zustand subscriptions block (alongside `cameraId`):
```ts
  const recognizeFaces = useAppState((s) => s.recognizeFaces);
```

Update `stateRef`:
```ts
    const stateRef = useRef({ mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId, recognizeFaces });
    stateRef.current = { mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId, recognizeFaces };
```

In `buildHeader`, include `recognize` for yolo modes (and harmlessly for sam3):

```ts
    function buildHeader(s: typeof stateRef.current): Record<string, unknown> {
      const base: Record<string, unknown> = { camera_id: s.cameraId, recognize: s.recognizeFaces };
      if (s.mode.startsWith('yolo_')) {
        return { ...base, conf: s.yoloConf };
      }
      if (s.mode === 'sam3_image' || s.mode === 'sam3_video') {
        return {
          ...base,
          prompt: s.sam3Prompt,
          text: s.sam3Text,
          tracking: s.tracking,
        };
      }
      return base;
    }
```

- [ ] **Step 2:** In `frontend/src/pages/LivePage.tsx`, add a checkbox under the "Controls" section. Find:

```tsx
        <label style={{ display: 'block', margin: '4px 0' }}>
          <input type="checkbox" checked={paused} onChange={(e) => setPaused(e.target.checked)} /> Pause
        </label>
```

Just after it, insert:

```tsx
        <label style={{ display: 'block', margin: '4px 0' }}>
          <input type="checkbox" checked={recognizeFaces}
            onChange={(e) => setRecognizeFaces(e.target.checked)} /> Recognize faces
        </label>
```

And add the two Zustand reads at the top of the component, alongside `mirror`/`paused`:
```tsx
  const recognizeFaces = useAppState((s) => s.recognizeFaces);
  const setRecognizeFaces = useAppState((s) => s.setRecognizeFaces);
```

- [ ] **Step 3:** Build:

```bash
cd frontend && npm run build
```

- [ ] **Step 4:** Commit.

```bash
git add frontend/src/stream/useFrameSender.ts frontend/src/pages/LivePage.tsx
git commit -m "feat(frontend): Recognize faces toggle + send recognize header"
```

---

## Task 7: Frontend — draw face name labels

**Files:** Create `frontend/src/overlay/drawFaceNames.ts`. Modify `frontend/src/overlay/OverlayCanvas.tsx`.

- [ ] **Step 1:** Create `frontend/src/overlay/drawFaceNames.ts`:

```ts
import type { FaceMatch } from '../types';

const COLORS: Record<FaceMatch['kind'], string> = {
  high: '#22c55e',
  mid: '#eab308',
  unknown: '#6b7280',
};

export function drawFaceNames(ctx: CanvasRenderingContext2D, faces: FaceMatch[]): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.save();
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  for (const f of faces) {
    const [nx, ny, nw, nh] = f.bbox;
    const x = nx * w;
    const y = ny * h;
    const bw = nw * w;
    const bh = nh * h;

    // Face bbox outline
    ctx.strokeStyle = COLORS[f.kind];
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);

    // Label above the box
    const label = f.kind === 'unknown' ? 'Unknown' : `${f.name} ${f.score.toFixed(2)}`;
    const padding = 4;
    const textW = ctx.measureText(label).width + padding * 2;
    const textH = 18;
    const lx = x;
    const ly = Math.max(0, y - textH);
    ctx.fillStyle = COLORS[f.kind];
    ctx.fillRect(lx, ly, textW, textH);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, lx + padding, ly + 2);
  }
  ctx.restore();
}
```

- [ ] **Step 2:** Modify `frontend/src/overlay/OverlayCanvas.tsx`. Add the import:

```tsx
import { drawFaceNames } from './drawFaceNames';
```

Then, inside the `useEffect` body, after the `drawBoxes` / `drawPose` branches, render face labels when present. Replace the existing `if (lastResult.type === 'detect') {...}` and `else if (lastResult.type === 'pose') {...}` branches with:

```tsx
    if (lastResult.type === 'detect') {
      drawBoxes(ctx, lastResult.boxes, false);
      if (lastResult.faces && lastResult.faces.length > 0) {
        drawFaceNames(ctx, lastResult.faces);
      }
    } else if (lastResult.type === 'pose') {
      drawPose(ctx, lastResult.people);
      if (lastResult.faces && lastResult.faces.length > 0) {
        drawFaceNames(ctx, lastResult.faces);
      }
    }
```

- [ ] **Step 3:** Build:

```bash
cd frontend && npm run build
```
Expected: success. If TS complains that `lastResult.faces` doesn't exist on the inferred type, add the optional `faces?: FaceMatch[]` to the corresponding result type. The implementer should locate `type: 'detect'` / `type: 'pose'` in `frontend/src/stream/protocol.ts` (or `types.ts`) and extend it:

```ts
| { type: 'detect'; frame_id: number; ms: number; boxes: Box[]; faces?: FaceMatch[] }
| { type: 'pose'; frame_id: number; ms: number; people: Person[]; faces?: FaceMatch[] }
```

(Import `FaceMatch` from `../types` if needed.)

- [ ] **Step 4:** Commit.

```bash
git add frontend/src/overlay/drawFaceNames.ts frontend/src/overlay/OverlayCanvas.tsx frontend/src/stream/protocol.ts 2>/dev/null
git commit -m "feat(frontend): draw face name labels in OverlayCanvas"
```

---

## Task 8: Manual end-to-end verification

**Files:** none.

- [ ] **Step 1:** Restart backend (need a fresh import to pick up the new singleton; existing background uvicorn may auto-reload — confirm in logs):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

- [ ] **Step 2:** Start frontend:

```powershell
cd frontend; npm run dev
```

- [ ] **Step 3:** Open `http://localhost:5173`. Sidebar should show a new **"Recognize faces"** checkbox under Controls.

- [ ] **Step 4:** Default = off. Confirm camera + YOLO works exactly as before.

- [ ] **Step 5:** Enable "Recognize faces". Within ~1 second, a colored label should appear above your face:
  - **Green** with your name + score (e.g. `Alice 0.78`) if you've registered yourself
  - **Yellow** `Alice? 0.48` if your match is in the borderline range
  - **Gray** `Unknown` if no match

- [ ] **Step 6:** Move around. Label should follow your face within ~1s (some cached lag is expected — this is the throttle).

- [ ] **Step 7:** Cover your face / leave frame. Label disappears within 3s (cache TTL).

- [ ] **Step 8:** Toggle "Recognize faces" off. Labels disappear; live FPS should bounce back up.

- [ ] **Step 9:** Commit the final marker.

```bash
git commit --allow-empty -m "docs: live face recognition verified end-to-end"
```

---

## Self-review notes

- Spec section 2 decisions all map to tasks: throttle interval/TTL config (Task 1); engine call + cache logic (Tasks 2/3); WS header `recognize` field (Task 4 backend, Task 6 frontend); `faces` array in result (Task 4 backend, Task 5 frontend type); sidebar checkbox (Task 6); color-coded labels (Task 7).
- The bbox normalization decision is documented in the plan header (refinement vs spec).
- No "TBD" / "TODO" placeholders.
- Type/name consistency: `FaceMatch` shape (frontend `[x,y,w,h]` array, backend tuple) is wire-compatible since FastAPI emits dataclass tuples as JSON arrays. `LiveRecognizer.annotate(camera_id, frame_bgr, person_boxes)` signature used consistently in Tasks 2-4. `live_recognizer` singleton name used in Tasks 3/4 backend and never collides with `recognition_worker`.
- Frontend tests are not added — repo has no FE test infra. Manual verification (Task 8) is the standard.

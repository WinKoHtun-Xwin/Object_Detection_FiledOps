# Object Detection Playground

Live webcam inference playground. The browser captures webcam frames, streams JPEG-encoded frames over a binary WebSocket protocol to a FastAPI backend, which runs **YOLO26 (Ultralytics)** object detection and overlays normalized results on the live video. Layered on top: motion-triggered clip recording and InsightFace-based face recognition (live overlay + offline clip processing).

- **Backend**: Python 3.12 + FastAPI + WebSocket, CUDA-accelerated inference on RTX 4070.
- **Frontend**: Vite + React + TypeScript + Zustand. Captures the webcam, streams frames to the backend, and overlays results on the live video.

## Features

- **YOLO26 object detection** — selectable model size n/s/m/l/x (nano → extra), with an adjustable confidence threshold, switchable from the browser. Weights auto-download on first use.
- **Object tracking** — per-camera ByteTrack assigns a stable `track_id` to each detection when tracking (or recognition) is enabled.
- **Motion-triggered clip recording** — a motion detector watches allowlisted classes and records MP4 clips with pre/post-roll and a trigger snapshot, one recorder per camera.
- **Face recognition** (InsightFace `buffalo_l`):
  - **Live overlay** — track-pinned recognizer names people in real time off a background worker thread.
  - **Offline** — a background worker samples each closed motion clip, matches faces against the gallery, records confident sightings, and queues uncertain ones for human review.
  - **Management UI** — people list, per-person galleries + clips, and a review queue for labeling unknown faces.

## Quick start

```powershell
# Backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"                 # includes pytest + ruff; drop [dev] for runtime only
uvicorn app.main:app --reload           # serves on http://127.0.0.1:8000

# Frontend (in another shell)
cd frontend
npm install
npm run dev                             # Vite dev server on http://localhost:5173
```

Open http://localhost:5173 and allow camera access.

YOLO26 weights auto-download via Ultralytics on first use. InsightFace + onnxruntime models download on first recognition.

> The frontend has **no Vite proxy** — the REST base URL and WebSocket URL are hardcoded to `http://localhost:8000` / `ws://localhost:8000/ws`. If you change the backend port, update `frontend/src/api/recognition.ts`, `frontend/src/pages/LivePage.tsx`, and `cors_origins` in `backend/app/config.py`.

## Tests & lint

```powershell
cd backend
pytest                                  # all backend tests
ruff check app                          # lint (line-length 100, target py312)

cd ../frontend
npm run build                           # tsc -b && vite build (type-check is part of build)
npm run lint                            # eslint
```

## Structure

```
backend/   Python app — FastAPI, WebSocket /ws, inference / recording / recognition subsystems
frontend/  TS app — webcam capture, WS streaming, canvas overlays, people/review pages
docs/      Architecture notes (see docs/ooda-loop.md) + design specs and plans under docs/superpowers/
```

See [CLAUDE.md](CLAUDE.md) for the full architecture, the frame protocol contract, and subsystem details, and [docs/ooda-loop.md](docs/ooda-loop.md) for the pipeline as an OODA loop.

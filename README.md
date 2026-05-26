# Object Detection Playground

Live webcam inference with **YOLO26 (Ultralytics)** and **Meta SAM 3**, switchable from the browser.

- **Backend**: Python 3.12 + FastAPI + WebSocket, CUDA-accelerated inference on RTX 4070.
- **Frontend**: Vite + React + TypeScript + Zustand. Captures webcam, streams JPEG frames to backend, overlays results on live video.
- **Models**:
  - YOLO26: detect (`yolo26n.pt`) + pose (`yolo26n-pose.pt`) — **working**
  - SAM 3: image model + video predictor — code paths installed, **awaiting HuggingFace access approval for `facebook/sam3`**. Once approved, set `mode='sam3_image'` and the engine will lazy-load.

## Quick start

```powershell
# Backend
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
uvicorn app.main:app --reload

# Frontend (in another shell)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access.

## Structure

See [docs/design.md](docs/design.md) for the full architecture.

```
backend/   Python app — FastAPI + inference engines
frontend/  TS app — webcam, WS, canvas overlays
docs/      Protocol spec, design notes
scripts/   Dev scripts (download weights, etc.)
```

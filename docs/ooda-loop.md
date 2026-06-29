# OODA Loop — Object Detection Pipeline

The system is a continuous **OODA loop** (Observe → Orient → Decide → Act, then repeat).
Each camera frame is one turn of a fast inner loop running ~15×/sec. This doc shows two views:

1. **Current** — the pipeline as built today.
2. **Target** — the same loop with the planned **Vision LLM** layer added. New pieces are marked `★`.

---

## Diagram 1 — Current pipeline

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    OODA LOOP — CURRENT                                     ║
║                    one full turn per camera frame, ~15x/sec                ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────┐      binary WS packet        ┌────────────────────────┐
  │     1. OBSERVE      │  ── [mode|size|hdr|jpeg] ──▶  │      2. ORIENT         │
  │   "what is there?"  │                               │   "what does it mean?" │
  │                     │                               │                        │
  │ webcam → canvas     │                               │ JPEG → BGR decode      │
  │ 640px, JPEG q0.7    │                               │ YOLO26 detect infer    │
  │ buildPacket()       │                               │   → boxes              │
  │                     │                               │ face match vs gallery  │
  │ useFrameSender.ts   │                               │   → names (FaceMatch)  │
  │ stream.py  /ws      │                               │ registry.py inference/ │
  └─────────────────────┘                               │ recognition/gallery.py │
            ▲                                            └────────────────────────┘
            │                                                        │
            │ next frame: world changed,                             │ normalized
            │ re-observe                                             │ results [0,1]
            │                                                        ▼
  ┌─────────────────────┐       decision signals       ┌────────────────────────┐
  │      4. ACT         │ ◀──────────────────────────   │      3. DECIDE         │
  │   "do something"    │                               │  "is it worth it?"     │
  │                     │                               │                        │
  │ draw overlay        │                               │ motion? bbox drift     │
  │  boxes/names        │                               │   > motion_threshold   │
  │ record MP4 + snap   │                               │ face score tier:       │
  │ enqueue review item │                               │   ≥0.55  → sighting    │
  │                     │                               │   0.40-0.55 → review   │
  │ recorder.py         │                               │   <0.40  → unknown     │
  │ OverlayCanvas.tsx   │                               │ motion.py  config.py   │
  └─────────────────────┘                               └────────────────────────┘
            │
            │  clip closed ─▶ recognition_worker ─▶ gallery.reload()
            └────────────────────────────────────────────────────────────┐
                  FEEDBACK: an updated gallery changes the next ORIENT  ◀──┘
```

---

## Diagram 2 — Target pipeline (Vision LLM layer)

The fast loop is unchanged. On an **interesting event** (motion clip, unknown face) a
**Vision LLM** runs a slower, deeper Orient pass over a whole clip — powering scene
description, natural-language rule alerts, searchable Q&A, live chat, and user-defined
custom events. `★` marks planned pieces.

```
╔══════════════════════════════════════════════════════════════════════════╗
║              OODA LOOP — TARGET (Vision LLM layer)                         ║
║              fast inner loop ~15x/sec  ·  slow VLM lane on events          ║
║              ★ = planned / not yet built                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────┐                              ┌──────────────────────────────┐
  │     1. OBSERVE      │  ── frame packet ─────────▶   │      2. ORIENT                │
  │   "what is there?"  │                              │  "what does it mean?"          │
  │                     │                              │                                │
  │ webcam → canvas     │                              │ FAST lane (every frame):       │
  │ 640px JPEG packet   │                              │  YOLO26 detect → boxes         │
  │ useFrameSender.ts   │                              │  face match → names            │
  │ stream.py /ws       │                              │ ─────────────────────────────  │
  └─────────────────────┘                              │ ★ SLOW lane (on event):        │
            ▲                                           │  Vision LLM reasons over a     │
            │                                           │  whole clip / recent frames    │
            │ next frame                                │   → NL scene description       │
            │ (re-observe)                              │  vlm_worker (planned)          │
            │                                           └──────────────────────────────┘
            │                                                  │              │
            │                                       fast results│      ★ description
            │                                                  ▼              ▼
  ┌─────────────────────┐                              ┌──────────────────────────────┐
  │      4. ACT         │ ◀───────────────────────────  │      3. DECIDE               │
  │   "do something"    │     decisions / signals      │  "is it worth it? rule match?" │
  │                     │                              │                                │
  │ draw overlay        │                              │ motion drift > threshold       │
  │ record MP4 + snap   │                              │ face score tier (0.40 / 0.55)  │
  │ enqueue review      │                              │ ★ RULE ENGINE:                 │
  │ ★ send notification │                              │   NL rules + user-defined      │
  │ ★ write NL desc +   │                              │   custom events, evaluated     │
  │   embedding → index │                              │   against the VLM description  │
  │ ★ answer live chat  │                              │ motion.py  config.py           │
  │ recorder.py overlay │                              │ rule_engine (planned)          │
  └─────────────────────┘                              └──────────────────────────────┘
            │
            │  clip closed ─▶ recognition_worker ─▶ gallery.reload()
            │  ★ event ─▶ Vision LLM ─▶ description ─▶ event index (embeddings)
            └──────────────────────────────────────────────────────────────────┐
                FEEDBACK: updated gallery + searchable event history          ◀─┘
                inform the next ORIENT and answer future questions

  ┌───────────────────────────── ★ USER QUERY PATH (out of band) ──────────────────────┐
  │  Searchable Q&A  ── reads ─▶  event index  ◀── writes ──  the ACT stage             │
  │  Live chat over feed ── asks ─▶ Vision LLM over recent frames / event index         │
  └─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## OODA → system mapping

| OODA stage | What it does | Today | Planned (★) |
|------------|--------------|-------|-------------|
| **Observe** | capture raw input | webcam → JPEG → WS packet (`useFrameSender.ts`, `stream.py`) | — |
| **Orient** | turn input into meaning | YOLO detect inference + face match (`inference/`, `recognition/gallery.py`) | Vision LLM reasoning pass on events → NL description |
| **Decide** | choose a response | motion threshold + face score tiers (`motion.py`, `config.py`) | NL rule engine + user-defined custom events |
| **Act** | execute, change the world | overlay draw, record MP4, enqueue review (`recorder.py`, `OverlayCanvas.tsx`) | notifications, write description+embedding to event index, answer chat |

**Why event-triggered:** the fast loop handles per-frame perception cheaply; the Vision LLM
is expensive, so it runs only on events the fast loop already flagged — and gets rich context
(a whole clip, not one frame) to reason over. In OODA terms: a fast inner loop plus a slow,
deeper Orient lane.
```

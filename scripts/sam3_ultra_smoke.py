"""Smoke test SAM 3 via Ultralytics wrapper."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = str(Path(__file__).resolve().parent.parent / "backend" / "weights" / "sam3.pt")
print("weights:", WEIGHTS)

t0 = time.perf_counter()
from ultralytics.models.sam import SAM3SemanticPredictor

print(f"import ({time.perf_counter() - t0:.1f}s)")

t0 = time.perf_counter()
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=WEIGHTS,
    half=True,
    save=False,
    verbose=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)
print(f"predictor instantiated ({time.perf_counter() - t0:.1f}s)")
print(f"cuda mem: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
pil = Image.fromarray(img)

for label in ("first", "warm1", "warm2"):
    t0 = time.perf_counter()
    predictor.set_image(pil)
    results = predictor(text=["person"])
    dt = time.perf_counter() - t0
    r = results[0] if isinstance(results, list) else results
    masks = getattr(r, "masks", None)
    boxes = getattr(r, "boxes", None)
    n_masks = len(masks.data) if masks is not None and masks.data is not None else 0
    n_boxes = len(boxes) if boxes is not None else 0
    print(f"{label}: {dt * 1000:.0f} ms  masks={n_masks}  boxes={n_boxes}  cuda={torch.cuda.memory_allocated() / 1024 / 1024:.0f}MB")

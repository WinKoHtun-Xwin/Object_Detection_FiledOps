"""SAM 3 smoke test: download checkpoint, load model, run two inferences for timing."""
import time

import numpy as np
import torch
from PIL import Image

t0 = time.perf_counter()
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print(f"imports done ({time.perf_counter() - t0:.1f}s)")

t0 = time.perf_counter()
model = build_sam3_image_model()
print(f"model loaded ({time.perf_counter() - t0:.1f}s)")
print(f"cuda mem: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")

processor = Sam3Processor(model)

img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))

for label in ("first", "warm1", "warm2"):
    t0 = time.perf_counter()
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt="person", state=state)
    dt = time.perf_counter() - t0
    masks = state["masks"]
    print(f"{label} infer ({dt * 1000:.0f} ms) masks: {tuple(masks.shape)} dtype={masks.dtype}")

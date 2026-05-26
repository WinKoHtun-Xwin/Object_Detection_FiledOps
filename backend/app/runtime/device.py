"""Device probe — detects CUDA/CPU once at startup."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    kind: str       # "cuda" | "cpu"
    name: str       # e.g. "NVIDIA GeForce RTX 4070 Laptop GPU" or "CPU"
    vram_mb: int    # 0 for CPU


def probe() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(kind="cpu", name="CPU (torch not installed)", vram_mb=0)

    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            kind="cuda",
            name=torch.cuda.get_device_name(idx),
            vram_mb=props.total_memory // (1024 * 1024),
        )
    return DeviceInfo(kind="cpu", name="CPU", vram_mb=0)

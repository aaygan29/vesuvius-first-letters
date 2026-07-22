"""Device selection that works on CUDA, Apple-Silicon MPS, or CPU.

The official Vesuvius notebook hardcodes ``.cuda()``; this project runs on a
Mac, so we resolve the best available backend and expose an autocast helper
that is a no-op where mixed precision is not supported.
"""
from __future__ import annotations

import contextlib
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def autocast(device: torch.device):
    """Mixed precision on CUDA only. MPS autocast is flaky, so we skip it."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda"):
            yield
    else:
        yield


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"

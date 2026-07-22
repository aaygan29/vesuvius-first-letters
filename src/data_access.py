"""Streaming access to scroll segment surface-volumes + ink labels.

The `vesuvius` pip package streams CT data as remote Zarr, so we never
download whole terabyte scans -- only the small crop we ask for. As of
package version installed here, `vesuvius.Volume.load_data()` has a bug
reading OME-Zarr multiscale *groups* (it expects a plain Array and crashes
on `.shape` for a Group). We work around it by opening the Zarr store and
ink-label PNG directly at the same URLs the package itself would use, which
keeps the same lazy / range-request streaming behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import requests
import zarr
from PIL import Image

DATA_ROOT = "https://dl.ash2txt.org/other/dev/scrolls"


@dataclass
class SegmentCrop:
    volume: np.ndarray    # (Z, Y, X) float32 in [0, 1]
    inklabel: np.ndarray  # (Y, X) float32 in [0, 1]
    z0: int
    y_range: tuple[int, int]
    x_range: tuple[int, int]
    segment_id: str


def list_scroll_segments(scroll="1", energy="54", resolution="7.91"):
    """Return the segment IDs the vesuvius catalog knows about for a scan config."""
    import vesuvius
    files = vesuvius.list_files()
    return list(files[scroll][energy][resolution]["segments"])


def _segment_urls(segment_id, scroll="1", energy="54", resolution="7.91um"):
    base = f"{DATA_ROOT}/{scroll}/segments/{energy}keV_{resolution}/"
    return {
        "zarr": f"{base}{segment_id}.zarr/",
        "ink": f"{base}{segment_id}_inklabels.png",
    }


def open_segment_array(segment_id, scroll="1", energy="54", resolution="7.91um", level: int = 0):
    """Open the full-resolution Zarr array (lazy -- no data pulled yet)."""
    urls = _segment_urls(segment_id, scroll, energy, resolution)
    group = zarr.open(urls["zarr"], mode="r")
    return group[str(level)]  # (Z, Y, X)


def load_inklabel(segment_id, scroll="1", energy="54", resolution="7.91um") -> np.ndarray:
    urls = _segment_urls(segment_id, scroll, energy, resolution)
    img = Image.open(BytesIO(requests.get(urls["ink"], timeout=60).content)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def load_crop(
    segment_id,
    z_center: int = 32,
    z_depth: int = 16,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
    scroll="1", energy="54", resolution="7.91um",
) -> SegmentCrop:
    """Stream a surface-volume crop centered on the recto surface + its ink label.

    The recto surface of the 7.91um Scroll 1 segments sits near layer z=32; a
    depth of 16 layers straddling it captures the ink signal (matches the
    official notebook's z=26..42 window).
    """
    arr = open_segment_array(segment_id, scroll, energy, resolution)
    Z, Y, X = arr.shape

    if y_range is None:
        y_range = (0, Y)
    if x_range is None:
        x_range = (0, X)
    y0, y1 = min(y_range[0], Y), min(y_range[1], Y)
    x0, x1 = min(x_range[0], X), min(x_range[1], X)

    z0 = max(0, z_center - z_depth // 2)
    z1 = min(Z, z0 + z_depth)

    vol = np.asarray(arr[z0:z1, y0:y1, x0:x1], dtype=np.float32) / 255.0

    ink_full = load_inklabel(segment_id, scroll, energy, resolution)
    # Ink PNG can be padded slightly larger than the volume; both share the
    # same top-left origin, so a direct crop is safe.
    ink = ink_full[y0:y1, x0:x1]

    return SegmentCrop(
        volume=vol, inklabel=ink, z0=z0,
        y_range=(y0, y1), x_range=(x0, x1), segment_id=str(segment_id),
    )

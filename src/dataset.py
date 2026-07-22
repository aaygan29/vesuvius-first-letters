"""Tiling dataset: cut a surface-volume + 2D ink label into overlapping tiles.

Convention: a tile corner is stored as ``[y, x]`` (row, col). The validation
zone is ``[y0, y1, x0, x1]`` and is held out spatially so train/val never
overlap. Training tiles are kept only if they contain at least ``min_ink``
fraction of ink, which focuses the tiny local run on informative regions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumetricDataset(Dataset):
    def __init__(
        self,
        volume: np.ndarray,     # (Z, Y, X)
        label: np.ndarray,      # (Y, X) in [0, 1]
        tile_size: int,
        stride: int,
        validation_zone,        # [y0, y1, x0, x1]
        valid: bool = False,
        min_ink: float = 0.05,
    ):
        self.volume = volume
        self.label = label
        self.tile_size = tile_size
        self.stride = stride
        self.vz = validation_zone
        self.valid = valid
        self.min_ink = min_ink
        self.tiles, self.labels, self.corners = self._extract()

    def _in_val(self, y: int, x: int) -> bool:
        ts, (y0, y1, x0, x1) = self.tile_size, self.vz
        return (y >= y0) and (y + ts <= y1) and (x >= x0) and (x + ts <= x1)

    def _touches_val(self, y: int, x: int) -> bool:
        ts, (y0, y1, x0, x1) = self.tile_size, self.vz
        overlap_y = not (y + ts < y0 or y > y1)
        overlap_x = not (x + ts < x0 or x > x1)
        return overlap_y and overlap_x

    def _extract(self):
        Z, Y, X = self.volume.shape
        tiles, labels, corners = [], [], []
        ts, st = self.tile_size, self.stride
        for y in range(0, Y - ts + 1, st):
            for x in range(0, X - ts + 1, st):
                if self.valid:
                    if not self._in_val(y, x):
                        continue
                else:
                    if self._touches_val(y, x):
                        continue
                    lab = self.label[y:y + ts, x:x + ts]
                    if lab.mean() < self.min_ink:
                        continue
                tiles.append(self.volume[:, y:y + ts, x:x + ts])
                labels.append(self.label[y:y + ts, x:x + ts])
                corners.append([y, x])
        return tiles, labels, corners

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int):
        tile = torch.tensor(np.ascontiguousarray(self.tiles[idx]), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(np.ascontiguousarray(self.labels[idx]), dtype=torch.float32).unsqueeze(0)
        corner = torch.tensor(self.corners[idx], dtype=torch.int32)
        return tile, label, corner

"""End-to-end ink-detection pipeline on a single scroll segment crop.

Streams a surface-volume crop + ink label, tiles it, trains the 3D->2D U-Net
on the train zone, runs tiled inference over the crop, and writes a
ground-truth-vs-prediction PNG. Device-agnostic (CUDA / MPS / CPU).

This is a *local, laptop-scale* run meant to prove the machinery end-to-end on
real data. Full First-Letters-grade results need cloud GPUs (see configs/).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from device import get_device, autocast, describe
from model import UNet, initialize_weights
from dataset import VolumetricDataset
from data_access import load_crop


def _save_overlay(crop, pred: np.ndarray, out_path: str, val_rect):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mid = crop.volume.shape[0] // 2
    axes[0].imshow(crop.volume[mid], cmap="gray")
    axes[0].set_title("CT surface layer")
    axes[1].imshow(crop.inklabel, cmap="gray")
    axes[1].set_title("Ground-truth ink label")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Model ink prediction")
    if val_rect is not None:
        y0, y1, x0, x1 = val_rect
        for ax in axes:
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           linewidth=2, edgecolor="r", facecolor="none"))
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(
    segment_id,
    y_range,
    x_range,
    z_center=32,
    z_depth=16,
    tile_size=256,
    stride=128,
    batch_size=4,
    epochs=40,
    base=32,
    lr=1e-4,
    out_dir="outputs",
):
    device = get_device()
    print(f"[device] {describe(device)}")

    print(f"[data] streaming crop from segment {segment_id} ...")
    t0 = time.time()
    crop = load_crop(segment_id, z_center=z_center, z_depth=z_depth,
                     y_range=y_range, x_range=x_range)
    print(f"[data] volume {crop.volume.shape}, ink {crop.inklabel.shape}, "
          f"ink%={100*crop.inklabel.mean():.2f}  ({time.time()-t0:.1f}s)")

    Y = crop.inklabel.shape[0]
    X = crop.inklabel.shape[1]
    # Hold out a central square for validation.
    vh = min(tile_size * 2, Y // 3)
    vw = min(tile_size * 2, X // 3)
    vy0 = (Y - vh) // 2
    vx0 = (X - vw) // 2
    val_rect = [vy0, vy0 + vh, vx0, vx0 + vw]

    train_ds = VolumetricDataset(crop.volume, crop.inklabel, tile_size, stride, val_rect, valid=False)
    valid_ds = VolumetricDataset(crop.volume, crop.inklabel, tile_size, stride, val_rect, valid=True)
    print(f"[data] train tiles={len(train_ds)}  val tiles={len(valid_ds)}  val_rect={val_rect}")
    if len(train_ds) == 0:
        raise SystemExit("No training tiles with ink found — enlarge the crop or lower min_ink.")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = UNet(base=base).to(device)
    initialize_weights(model)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[train] {epochs} epochs, base={base}, tile={tile_size}, batch={batch_size}")
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for tiles, labels, _ in train_dl:
            tiles, labels = tiles.to(device), labels.to(device)
            opt.zero_grad()
            with autocast(device):
                out = model(tiles)
                loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            running += loss.item()
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[train] epoch {epoch+1}/{epochs}  loss={running/max(1,len(train_dl)):.4f}")

    # Tiled inference over the full crop.
    print("[infer] running tiled inference ...")
    pred = np.zeros_like(crop.inklabel, dtype=np.float32)
    count = np.zeros_like(crop.inklabel, dtype=np.float32)
    full_ds = VolumetricDataset(crop.volume, crop.inklabel, tile_size, stride,
                                validation_zone=[0, Y, 0, X], valid=True)
    full_dl = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for tiles, _, corners in full_dl:
            tiles = tiles.to(device)
            with autocast(device):
                out = torch.sigmoid(model(tiles))
            out = out.cpu().numpy()
            corners = corners.numpy()
            for i in range(corners.shape[0]):
                y, x = int(corners[i, 0]), int(corners[i, 1])
                pred[y:y + tile_size, x:x + tile_size] += out[i, 0]
                count[y:y + tile_size, x:x + tile_size] += 1
    count[count == 0] = 1
    pred /= count

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"ink_pred_{segment_id}.png")
    _save_overlay(crop, pred, png, val_rect)
    np.save(os.path.join(out_dir, f"ink_pred_{segment_id}.npy"), pred)
    print(f"[done] wrote {png}")
    return png


def main():
    ap = argparse.ArgumentParser(description="Local ink-detection demo on one segment crop.")
    ap.add_argument("--segment_id", default="20230827161847")
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=3200)
    ap.add_argument("--x0", type=int, default=2000)
    ap.add_argument("--x1", type=int, default=3200)
    ap.add_argument("--z_center", type=int, default=32)
    ap.add_argument("--z_depth", type=int, default=16)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    run(
        segment_id=args.segment_id,
        y_range=(args.y0, args.y1),
        x_range=(args.x0, args.x1),
        z_center=args.z_center,
        z_depth=args.z_depth,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base=args.base,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

# Vesuvius Challenge — First Letters attempt

Ink detection on carbonized Herculaneum scrolls, targeting the [Vesuvius Challenge
**First Letters** prize](https://scrollprize.org/prizes) ($50,000 per scroll:
find 10 legible letters within a single 4 cm² region of an unopened scroll).

This repo takes a virtually-unwrapped scroll **surface volume** (a thin stack of
CT layers straddling one papyrus sheet) and predicts a per-pixel **ink probability
map** with a 3D→2D U-Net, following the approach that first revealed the word
ΠΟΡΦΥΡΑϹ ("purple") inside a sealed scroll in 2023.

> Background: I work on neuro segmentation / volumetric medical imaging, so the
> core problem here (weak-signal segmentation in 3D CT volumes) is familiar
> territory ported to a new domain.

## Why the ink signal is hard

The eruption of Vesuvius carbonized papyrus *and* ink alike. Carbon ink on
carbonized papyrus is nearly invisible to X-ray attenuation — there is no simple
"bright = ink". The signal lives in **texture / morphology** (a "crackle"
pattern on the fiber surface) that a CNN can learn but the eye mostly cannot.
That is why we train on the surface volume rather than thresholding intensities.

## Pipeline

```
CT scan (Zarr, streamed)
   └─ Volume(segment_id)          # vesuvius pkg, remote S3/Zarr — no TB download
        └─ surface-volume crop     # (Z≈16 layers around recto surface, Y, X)
             └─ tile (256², stride 128)
                  └─ 3D→2D U-Net   # 3D conv over Z, max over depth, 2D U-Net
                       └─ BCE-with-logits training (held-out val zone)
                            └─ tiled inference → ink probability map (PNG/npy)
```

Code:

| file | role |
|------|------|
| `src/data_access.py` | stream a segment surface-volume crop + ink label |
| `src/dataset.py`     | tile the crop; spatial train/val split; ink-fraction filter |
| `src/model.py`       | 3D→2D U-Net (`base` width knob for laptop memory) |
| `src/device.py`      | CUDA / Apple-MPS / CPU selection + safe autocast |
| `src/pipeline.py`    | end-to-end: stream → train → infer → save overlay |

## Quickstart

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
vesuvius.accept_terms --yes            # one-time data license

# Local proof run on a real Scroll 1 segment crop:
python src/pipeline.py --segment_id 20230827161847 \
    --y0 2000 --y1 3200 --x0 2000 --x1 3200 --epochs 40
# -> outputs/ink_pred_20230827161847.png  (CT | ground truth | prediction)
```

## Current status

`outputs/ink_pred_20230827161847.png` is a real run: streamed crop, 52
training tiles, 30 epochs, on MPS. The pipeline works end-to-end, but at this
scale the prediction has **not** converged to legible letters (loss 0.65→0.50,
still mostly texture) — that needs far more data/epochs/model capacity, i.e.
cloud GPU time (see below), not a change to the approach itself.

## Local vs cloud

This machine is an Apple-Silicon Mac (MPS, **no CUDA**) with limited disk. The
`vesuvius` package **streams** CT data as remote Zarr, so data access works
fine locally, and the pipeline runs end-to-end on a small crop on MPS to prove
the machinery. **Competitive** First-Letters results need cloud GPUs and the
full-scale models (TimeSformer / 3D-ResNet ensembles). See
[`docs/cloud_gpu.md`](docs/cloud_gpu.md) and [`configs/`](configs/).

## First Letters submission checklist

See [`docs/submission_checklist.md`](docs/submission_checklist.md) for the exact
prize requirements (tifxyz mesh, programmatic static image, scale bar, held-out
validation, false-positive mitigation, Docker reproducibility) and how each maps
to this repo.

## Credit / prior art

- Official `vesuvius` package + ink-detection notebook (ScrollPrize/villa).
- 2023 Grand Prize solution (Nader, Farritor, Schilliger) — TimeSformer + 3D-ResNet.
- Data © EduceLab / University of Kentucky & Vesuvius Challenge; used under the
  challenge data license.

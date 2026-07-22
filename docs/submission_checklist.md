# First Letters submission checklist

Source: [scrollprize.org/prizes](https://scrollprize.org/prizes) (accessed 2026-07-22).
$50,000 per scroll, up to $500,000 total (max 10 of 13 eligible 2027 Grand Prize
scrolls: PHerc. 125, 191, 211, 257, 268, 358, 800, 813, 826, 1203, 1218, 1447, 1545).
Deadline **June 25, 2027, 11:59pm Pacific**. First qualifying submission wins.

## Bar to clear
10 legible letters within a single **4 cm²** region, on one of the 13 eligible
scrolls (**not** Scroll 1, where this repo's proof-of-concept segment lives —
Scroll 1 was already read for the 2023 prize).

## Required artifacts

- [ ] **Mesh** in `tifxyz` format, low-distortion 2D flattening of the region
      → produced by segmentation (VC3D / ThaumatoAnakalyptor), not this repo yet.
- [ ] **Single static image**, generated *programmatically* from the CT scan —
      no hand-drawn letters. `src/pipeline.py`'s `_save_overlay` produces exactly
      this class of artifact (needs to be run against a real held-out eligible-scroll
      region, not the demo Scroll 1 crop).
- [ ] **Scale bar**: 1cm reference + letter-size annotations on the image.
      *(not yet in `_save_overlay` — TODO before any real submission.)*
- [ ] Letters oriented so they run **parallel to horizontal papyrus fibers**.
- [ ] **No overlap** between the predicted/shown region and any training data.
- [ ] **Methodology docs**: Docker image (if automated) or walkthrough video
      (if semi-automated). `docs/cloud_gpu.md` + a `Dockerfile` cover this.
- [ ] **False-positive mitigation** write-up. Notably: avoid ink models that key
      off tiny (< 0.5mm × 0.5mm) windows — treated as a noise-fitting red flag.
- [ ] **Held-out validation** results (`src/pipeline.py` already spatially
      splits train/val — extend the same discipline to the real submission run).

## Process

- [ ] Join Vesuvius Challenge Discord (prerequisite for submitting).
- [ ] Keep any real discovery **confidential** until official announcement.
- [ ] Submit via https://forms.gle/TM5ao8GwC2mDrdLk9

## Gap between this repo today and a real submission

This repo currently proves the *ink-detection* half of the pipeline on a
Scroll 1 demo segment with existing labels. A real submission additionally needs:

1. A **segmented, flattened surface** on one of the 13 eligible *unread* scrolls
   (this repo does not do segmentation/unwrapping — see `docs/cloud_gpu.md` for
   pointers to VC3D / ThaumatoAnakalyptor, the tools that do).
2. **No ground-truth ink labels exist** on those unread regions by definition —
   training has to follow the historical bootstrap loop (Casey's crackle →
   Luke's/Youssef's iterative pseudo-labeling), not supervised fine-tuning alone.
3. Scale-bar annotation + papyrological-plausibility self-review before submitting.

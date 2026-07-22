# Scaling beyond a laptop

`src/pipeline.py` proves the pipeline end-to-end on Apple-Silicon MPS with a
small crop and a lightweight U-Net (`base=24`, ~30 epochs, one segment). Real
First Letters-grade work needs more compute and more of the actual stack:

## Compute

- **Colab (free tier T4 / A100 if available)**: fastest path to a real CUDA GPU
  for iterating on `src/pipeline.py` with larger `base`, bigger crops, more epochs.
- The official ink-detection notebooks are already Colab-ready:
  - [Data access](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example1_data_access.ipynb)
  - [Ink detection](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example2_ink_detection.ipynb)
- For a serious run: rent an A100/H100 (Lambda, RunPod, Paperspace) — the
  Grand-Prize-winning TimeSformer/3D-ResNet ensemble in
  [`ink-detection/`](https://github.com/ScrollPrize/villa/tree/main/ink-detection)
  wants `--gpus all --shm-size=150g`.

## Bigger models to graduate to

Once the laptop U-Net proof-of-concept is validated, the next steps (roughly
increasing cost/complexity) are:

1. **Pretrained checkpoints** — `vesuvius.predict` runs pretrained nnUNet-v2
   models from Hugging Face (`huggingface.co/scrollprize`) directly on remote
   Zarr, no local training needed. Fastest way to get a strong baseline.
2. **`vesuvius` package trainer** (`vesuvius.train`) — the maintained
   nnUNetv2-based trainer with multi-task/multi-class support, proper
   augmentation, and distributed inference (`--num_parts`/`--part_id`).
3. **Grand-Prize architecture** (`ink-detection/` in villa monorepo) —
   TimeSformer-small (divided space-time attention) + 3D-ResNet101 + I3D
   ensemble, trained via 15 rounds of label cleaning/expansion. This is what
   actually won the $700k Grand Prize; treat it as the reference ceiling.

## Segmentation / unwrapping (not yet in this repo)

Ink detection only matters on a properly flattened surface. For the 13 First
Letters-eligible scrolls (unlike the demo Scroll 1 segment used here, which is
already segmented), you'd first need:

- **VC3D** (Volume Cartographer 3D) — manual/assisted fiber & winding tracing.
- **ThaumatoAnakalyptor** — automatic 3D segmentation
  ([github.com/schillij95/ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor)),
  handles "mushy"/twisted regions better than manual tracing.
- Output: a `tifxyz` mesh with low-distortion 2D flattening — the format the
  First Letters submission requires.

## Data at scale

- `s3://vesuvius-challenge-open-data/` — full raw CT (terabytes; stream, don't download).
- Curated ink/spiral datasets: [huggingface.co/buckets/scrollprize/datasets](https://huggingface.co/buckets/scrollprize/datasets)
- Data Browser: https://scrollprize.org/data_browser

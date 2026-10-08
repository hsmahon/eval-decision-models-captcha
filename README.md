# Eval - Decision Models - CAPTCHA

How do decision models perform as visual decision complexity increases?

> [!WARNING]
>
> This README was written by a human, but all code changes, and
> additional documentation were authored entirely by Muse Spark 1.3 in OpenCode.

## Model (pinned)

- `StrandsAgents/strands-decider-2B-hobson-v21` — typed decision model (`noul` yes/no + calibrated confidence), LoRA on Qwen3.5-2B.
- Vision via `pip install "strands-decider[vision]"` (needs `transformers>=5.18`), loaded as `VisionDeciderModel`. No vision training; frozen Qwen3.5 tower.
- Every eval call is one `noul` per cell crop with the fixed prompt above. See `eval.ipynb` §2.

## Dataset — COCO Photo Crops

The grids in `data/images/` are real photographs, rebuilt from
**COCO 2017** (via the Hugging Face mirror `detection-datasets/coco`,
research use — see `data/ATTRIBUTION.md`)

Why COCO specifically:

- **Real pixels, free ground truth.** COCO ships bounding boxes for every
  target class we need (traffic light, car, bus, bicycle, fire hydrant, stop
  sign). Each grid cell is a square crop around a real annotated instance, so
  cell-level labels stay deterministic — no hand-labeling, no guessing.
- **Difficulty becomes visual, not decorative.** Easy cells get the largest,
  most prominent instances; hard cells get the smallest/most cluttered ones
  (ranked by box-area ratio). The easy/med/hard split now measures something.

Grid spec: 150 images (30 each of 3×3, 3×4, 4×4, 4×5, 5×5), 320 px/cell,
2,460 cells total. Per-cell label mix: car 388, bicycle 417, bus 327,
stoplight 277, hydrant 347, sign 314, empty 390 (background crops from
street-scene images only).

## Reproducing the Eval

Run on a Colab GPU or any Linux/Mac with a GPU:

Colab (GPU runtime → L4/T4):
1. Upload this repo (or `git clone <url>`).
2. Open `eval.ipynb` → set `LIMIT = 5`, run all (smoke test, no crash).
3. Set `LIMIT = None`, run all (~8 min on GPU at ~200 ms/call) → overwrites `data/results.json`.
4. Open `index.html` via any static server or GH Pages.

Local:
```bash
pip install "strands-decider[vision]" pillow
# smoke, then full — same LIMIT pattern inside the notebook
```

`DRY_RUN_NO_MODEL` in §3 checks cropping/timing without weights.

## What `results.json` records

`manifest` (model, prompt verbatim, backend, timestamp, package versions, n_images_run/150) + `decisions[]` (GT, pred, confidence, latency_ms per cell) + aggregates (accuracy + Wilson 95% CI, positive-class F1, mean/median/p95 latency, confidence buckets, per-image leaderboard). Recompute metrics without re-running the model.

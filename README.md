# Vision decision eval — `StrandsAgents/strands-decider-2B-hobson-v21`

Question: how does a small vision decision model perform as visual decision complexity increases?

Synthetic benchmark only (not for bypassing real CAPTCHAs). Per-cell task: “Does this square contain a stoplight?”

## Model (pinned)

- `StrandsAgents/strands-decider-2B-hobson-v21` — typed decision model (`noul` yes/no + calibrated confidence), LoRA on Qwen3.5-2B.
- Vision via `pip install "strands-decider[vision]"` (needs `transformers>=5.18`), loaded as `VisionDeciderModel`. No vision training; frozen Qwen3.5 tower.
- Every eval call is one `noul` per cell crop with the fixed prompt above. See `eval.ipynb` §2.

## Layout (3 things)

- `eval.ipynb` — the experiment. Runs the 150-image / 2,460-cell eval, writes `data/results.json`.
- `index.html` — single-file static page (inline CSS+JS). Reads `./data/results.json`. Never calls the model. Served from repo root on GH Pages.
- `data/` — `images/*.png` + `labels.jsonl` (fixed artifact) + `results.json` (generated).

## Reproduce (GPU required)

This box (4 CPU, 7 GB RAM, no GPU, no torch) cannot run the 2B vision model — CPU would OOM/crawl on 2,460 calls. Run on a Colab GPU or any Linux/Mac with a GPU:

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

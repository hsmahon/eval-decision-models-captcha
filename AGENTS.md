# AGENTS.md — eval-decision-models-captcha

## Project overview

Benchmark: how does a small vision decision model perform as visual decision
complexity increases, using CAPTCHA-style grids?

- Question per cell: `Does this square contain a stoplight?` (one `noul` yes/no + confidence call per cell crop).
- Dataset: 150 fixed synthetic grids in `data/images/` (30 each of 3×3, 3×4, 4×4, 4×5, 5×5; 10 easy / 10 medium / 10 hard per size), 320 px/cell, 2,460 cells total. Real COCO 2017 photo crops (see `data/ATTRIBUTION.md`), not scraped CAPTCHAs.
- Artifacts: `eval.ipynb` (Artifact 1, runs model → writes `data/results.json`) → `index.html` (Artifact 2, static dashboard, never calls the model).
- Ethics: synthetic benchmark only, not for bypassing real CAPTCHAs.

## Tech stack

- Model (pinned): `StrandsAgents/strands-decider-2B-hobson-v21` — typed decision model, LoRA on Qwen3.5-2B. Vision via frozen Qwen3.5 tower, no vision training.
- Python: `strands-decider[vision]` (requires `transformers>=5.18`), `pillow`. Metrics in notebook are stdlib-only (`math`, `statistics`, `json`).
- Model API: `VisionDeciderModel.load(MODEL_ID)` then `ask_noul / noul / ask(image, question)`, with CLI fallback `strands-decider ask --image crop.png --noul "..."`. See `eval.ipynb` §2.
- Frontend: vanilla HTML/CSS/JS in `index.html`, no build step, fetches `./data/results.json`.
- Data files: `data/labels.jsonl` (cell-level ground truth), `data/results.json` (manifest + decisions + aggregates).

## Repo layout

- `eval.ipynb` — §0 setup, §1 dataset check, §2 model load + crop/ask helpers, §3 run eval (`LIMIT`, `DRY_RUN_NO_MODEL`), §4 metrics → `results.json`.
- `data/images/` — 150 `.png` grids named by `image_id`.
- `data/labels.jsonl`, `data/results.json`, `data/ATTRIBUTION.md`, `data/README.dataset.md`.
- `index.html` — accuracy vs grid size, difficulty split, latency, confidence buckets, leaderboard with cell overlay.
- `README.md` — human-written intro + Colab/local repro instructions.

## Workflows

- Colab (preferred, T4 GPU): see `README.md` for copy-paste setup. Smoke test `LIMIT = 5` before full `LIMIT = None` (~8 min, ~200 ms/call).
- Local: `pip install "strands-decider[vision]" pillow`, same `LIMIT` pattern. Needs a GPU for full run.
- Dry run without weights: `DRY_RUN_NO_MODEL = True` in §3 checks cropping/timing only.
- Viewing results: serve repo root statically (e.g. `python3 -m http.server`) or GH Pages, then open `index.html`.

## Conventions for agents

- Keep model ID, prompt (`Does this square contain a stoplight?`), and 150-image set pinned for comparability. Don't regenerate images or hand-edit labels.
- `results.json` schema: `manifest` + `overall` + `by_grid` + `by_grid_difficulty` + `per_image` + `decisions[]`; keep `schema_version` bump if changed.
- Keep `index.html` model-free (reads `results.json` only).
- All code/docs changes here are agent-authored (Muse Spark 1.3 in OpenCode); README intro stays human-written.

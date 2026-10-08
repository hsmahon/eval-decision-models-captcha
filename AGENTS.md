# AGENTS.md — eval-decision-models-captcha

## Project overview

Benchmark: how do vision decision models perform as visual decision
complexity increases, using CAPTCHA-style grids?

- Question per cell (prompt-v2-neutral): `Is a stoplight visible in this image?`
  with explicit true/false `criteria` (whole-or-partial signals count;
  lookalikes excluded). One typed yes/no call per cell crop.
- Dataset: 150 fixed synthetic grids in `data/images/` (30 each of 3×3, 3×4, 4×4, 4×5, 5×5; 10 easy / 10 medium / 10 hard per size), 320 px/cell, 2,460 cells total. Real COCO 2017 photo crops (see `data/ATTRIBUTION.md`), not scraped CAPTCHAs.
- Artifacts: eval runners (notebook for local, `scripts/` for hosted) → one
  `data/results_<backend>.json` per backend → `index.html` (static dashboard,
  never calls any model).
- Ethics: synthetic benchmark only, not for bypassing real CAPTCHAs.

## Backends (Jev excluded — text-only, no image input)

| key | model | runner | cost basis |
|---|---|---|---|
| `strands` | `StrandsAgents/strands-decider-2B-hobson-v21` (LoRA on Qwen3.5-2B, frozen Qwen3.5 vision tower) | `eval.ipynb` §§2–4, Colab T4 | $0.00 local |
| `d1` | `LiquidAI/d1-3B` (native `noul` head via `system_one`, `transformers>=5.14`, bf16 ~7–8 GB) | `eval_d1.ipynb` (Colab T4), same prompt-v2 + criteria | $0.00 local |
| `clef-flash` | `@cf/cloudflare/clef-flash` 9B (`noul` + native `criteria`, base64 `images[]`) | hosted script (removed; results kept) | $0.09/1M input tok (from `usage`) |
| `openai` | `gpt-6-luna` via `POST /v1/decisions` (`predicate`, base64 data-URL) | hosted script (removed; results kept) | $0.10/1M input tok (measured ~345 tok/cell) |

Results so far (2,460 cells each): clef-flash 0.9553 (p50 622 ms, $0.27),
openai 0.9435 (p50 226 ms, $0.10). Local backends pending (Colab).

## Tech stack

- Local Python: `strands-decider[vision]` (`transformers>=5.18`), `LiquidAI/d1-3B`
  (`transformers>=5.14`, `trust_remote_code=True`), `pillow`. Metrics are
  stdlib-only (`math`, `statistics`, `json`).
- Hosted runners: stdlib-only (`urllib`) + `pillow` for cropping. No SDK needed.
- Model APIs: strands `VisionDeciderModel.load(...)` → `ask_noul / noul / ask`
  (CLI fallback `strands-decider ask --image crop.png --noul "..."`);
  d1 `AutoModel.from_pretrained("LiquidAI/d1-3B")` → `system_one(None, {noul...}, images=[crop])`.
- Frontend: vanilla HTML/CSS/JS in `index.html`, no build step. Loads
  `./data/results.json` (strands) + `./data/results_<key>.json`, with embedded
  dry-run fallback for `file://`. Three sections: accuracy-vs-size chart (mean
  default, single-model dropdown), model comparison (accuracy · p50/p95 ms · cost),
  per-image leaderboard with cell overlay.
- Data files: `data/labels.jsonl` (cell-level ground truth),
  `data/results.json` + `data/results_<backend>.json` (manifest + decisions + aggregates).

## Repo layout

- `eval.ipynb` — strands runner: §0 setup, §1 dataset check, §2 model load + crop/ask helpers, §3 run eval (`LIMIT`, `DRY_RUN_NO_MODEL`), §4 metrics → `results.json`.
- `eval_d1.ipynb` — d1-3B runner (same structure, writes `data/results_d1.json`).
- `data/images/` — 150 `.png` grids named by `image_id`.
- `data/labels.jsonl`, `data/results*.json`, `data/ATTRIBUTION.md`, `data/README.dataset.md`.
- `index.html` — 3-section leaderboard (see Tech stack).
- `README.md` — human-written intro + Colab/local repro + backend table.

## Workflows

- Colab (local backends, T4 GPU): see `README.md`. Smoke `LIMIT = 5` before full
  `LIMIT = None`. `eval.ipynb` writes `data/results.json` (the dashboard reads
  strands from that filename — don't rename it).
- Viewing results: GH Pages (auto-build from `main`), hard-refresh after deploy.
  Dashboard needs no server (embedded fallback covers `file://`).

## Conventions for agents

- Keep model IDs, prompt-v2 + criteria, and 150-image set pinned for comparability. Don't regenerate images or hand-edit labels.
- Results schema: `manifest` (model, prompt, prompt_version, criteria, est_cost_usd, backend, n_images_run/150, full_run) + `overall` + `by_grid` + `by_grid_difficulty` + `per_image` + `decisions[]`; bump `schema_version` if changed.
- Latency is per-decision-call milliseconds: `lat_median` = p50, `lat_p95` nearest-rank. Dashboard header: `p50 / p95`.
- Local backends record `est_cost_usd: 0.0`; hosted compute it from metered input tokens.
- Keep `index.html` model-free (reads results files only). Mock values only for unrun backends, untagged.
- All code/docs changes here are agent-authored (Muse Spark 1.3 in OpenCode); README intro stays human-written.

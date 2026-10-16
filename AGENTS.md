# AGENTS.md — eval-decision-models-captcha

## Project overview

Benchmark: how do vision decision models perform as visual decision
complexity increases, using CAPTCHA-style grids?

- Question per cell (prompt-v2-neutral): `Is a stoplight visible in this image?`
  with explicit true/false `criteria` (whole-or-partial signals count;
  lookalikes excluded). One typed yes/no call per cell crop.
- Dataset: 150 fixed synthetic grids in `data/images/` (30 each of 3×3, 3×4, 4×4, 4×5, 5×5; 10 easy / 10 medium / 10 hard per size), 320 px/cell, 2,460 cells total. Real COCO 2017 photo crops (see `data/ATTRIBUTION.md`), not scraped CAPTCHAs.
- Artifacts: eval runners (hosted scripts removed after runs) → one
  `data/results_<backend>.json` per backend → `index.html` (static dashboard,
  never calls any model).
- Ethics: synthetic benchmark only, not for bypassing real CAPTCHAs.

## Backends (hosted only — local runners removed at wrap-up)

| key | model | cost basis |
|---|---|---|
| `clef-flash` | `@cf/cloudflare/clef-flash` 9B (`noul` + native `criteria`, base64 `images[]`) | $0.09/1M input tok (from `usage`) |
| `openai` | `gpt-6-luna` via `POST /v1/decisions` (`predicate`, base64 data-URL) | $0.10/1M input tok (measured ~345 tok/cell) |
| `clef-omni` | `@cf/cloudflare/clef-omni` 30B-A3B (`noul` + native `criteria`, base64 `images[]`) | $0.15/1M input tok (from `usage`) |
| `jev` | `typesafe/jev-1.14` via OpenRouter `POST /api/alpha/decisions` (`noul`, base64 PNG data URL in `state`) | $0.042/1M input tok (metered usage) |
| `liquid-d1` | Liquid AI `d1` via `POST /decisions/v1/systemone` (`noul`, base64 PNG data URL in `images[]`) | $0.02/1M input tok (Liquid pricing) |
| `liquid-d1-ultra` | Liquid AI `d1-ultra` via `POST /decisions/v1/systemone` (`noul`, base64 PNG data URL in `images[]`) | $0.042/1M input tok |

Results (2,460 cells each): clef-flash 0.9553 (p50 622 ms, $0.27),
openai 0.9435 (p50 226 ms, $0.10), clef-omni 0.9585 (p50 391 ms, $0.11).
Jev 1.14 through OpenRouter: 0.9443 accuracy (p50 301 ms, $0.068),
served as `typesafe/jev-1.14-20261014`.
Liquid d1: 0.9488 accuracy (p50 480 ms, $0.011906); d1-ultra: 0.9606 accuracy
(p50 267 ms, $0.025003). Both ran all 2,460 cells via Liquid API. Results are
stored separately in `data/results_liquid-d1.json` and
`data/results_liquid-d1-ultra.json`.

## Tech stack

- Model APIs: plain HTTPS calls — Clef-flash via Workers AI REST (`images[]` + `noul`
  with native `criteria`), OpenAI via `POST /v1/decisions` (`predicate`, base64 data-URL).
  Jev through OpenRouter uses `/api/alpha/decisions`; Liquid d1 and d1-ultra use
  `POST /decisions/v1/systemone` with base64 PNG data URLs in `images[]`. Cropping uses Pillow;
  metrics are stdlib-only (`math`, `statistics`, `json`).
- Frontend: vanilla HTML/CSS/JS in `index.html`, no build step. Loads
  `./data/results_<key>.json` (one file per backend). Three sections:
  accuracy-vs-size chart (mean default, single-model dropdown), model comparison
  (accuracy · p50/p95 ms · cost), per-image leaderboard with cell overlay.
- Data files: `data/labels.jsonl` (cell-level ground truth),
  `data/results*.json`, `data/ATTRIBUTION.md`, `data/README.dataset.md`.

## Repo layout

- `data/results_<backend>.json` — one per backend (manifest + decisions + aggregates).
- `data/images/` — 150 `.png` grids named by `image_id`.
- `data/labels.jsonl`, `data/results*.json`, `data/ATTRIBUTION.md`, `data/README.dataset.md`.
- `index.html` — 3-section leaderboard (see Tech stack).
- `README.md` — human-written intro + repro + backend table.

## Workflows

- Viewing results: GH Pages (auto-build from `main`), hard-refresh after deploy.
  Dashboard needs no server (embedded fallback covers `file://`).

## Conventions for agents

- Keep model IDs, prompt-v2 + criteria, and 150-image set pinned for comparability. Don't regenerate images or hand-edit labels.
- Results schema: `manifest` (model, prompt, prompt_version, criteria, est_cost_usd, backend, n_images_run/150, full_run) + `overall` + `by_grid` + `by_grid_difficulty` + `per_image` + `decisions[]`; bump `schema_version` if changed.
- Latency is per-decision-call milliseconds: `lat_median` = p50, `lat_p95` / `lat_p99` nearest-rank. Dashboard header: `p50 / p95 / p99`.
- Hosted backends compute `est_cost_usd` from metered input tokens.
- Keep `index.html` model-free (reads results files only).
- All code/docs changes here are agent-authored (Muse Spark 1.3 in OpenCode); README intro stays human-written.

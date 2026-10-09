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
| `clef-omni` | `@cf/cloudflare/clef-omni` 30B-A3B (`noul` + native `criteria`, base64 `images[]`; smoke 2/150 — full run split across free-tier quota resets) | $0.15/1M input tok (from `usage`) |

Results (2,460 cells each): clef-flash 0.9553 (p50 622 ms, $0.27),
openai 0.9435 (p50 226 ms, $0.10).

## Tech stack

- Model APIs: plain HTTPS calls — Clef-flash via Workers AI REST (`images[]` + `noul`
  with native `criteria`), OpenAI via `POST /v1/decisions` (`predicate`, base64 data-URL).
  Cell crops are inline base64 PNGs; metrics are stdlib-only (`math`, `statistics`, `json`).
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

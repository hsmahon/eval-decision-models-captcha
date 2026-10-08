# Eval - Decision Models - CAPTCHA

How do decision models perform as visual decision complexity increases? I wanted to evaluate decision models on images to see if they could solve an infamous annoyance - CAPTCHAs.

> [!WARNING]
>
> This README was written by a human (me!), but all code changes, and
> additional documentation were authored entirely by Muse Spark 1.3 in OpenCode.

## Models (leaderboard)

| backend | model | runs where | cost |
|---|---|---|---|
| `strands` | `StrandsAgents/strands-decider-2B-hobson-v21` — `noul` yes/no + calibrated confidence, LoRA on Qwen3.5-2B, frozen Qwen3.5 vision tower (`transformers>=5.18`) | local GPU | $0.00 |
| `d1` | Liquid `d1-3B` (open weights, text+vision) | local GPU | $0.00 |
| `clef-flash` | Cloudflare Clef-flash 9B (`@cf/cloudflare/clef-flash`) | Workers AI | $0.09/1M input tok |
| `openai` | OpenAI Decisions API (GPT-6 Luna, public beta) | hosted API | $0.10/1M input tok |

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

Grid spec: 150 images (30 each of 3×3, 3×4, 4×4, 4×5, 5×5).

## Reproducing the Eval

Run on a Colab GPU or any Linux/Mac with a GPU. Colab's most available free
GPU is the **T4** — select it via `Runtime → Change runtime type → T4 GPU`.

Colab (T4 GPU runtime):
```bash
git clone https://github.com/hsmahon/strands-decider-vision-eval
%cd strands-decider-vision-eval
!pip install "strands-decider[vision]" pillow
```
1. Open `eval.ipynb` in Colab (File → Upload notebook). Cell 1 auto-clones the repo so `data/` is present — just run cells top to bottom.
2. Run §0–§1 — confirm `labels.jsonl` loads (150 images) and package versions print.
3. Set `LIMIT = 5`, Run All (smoke test, no crash).
4. Set `LIMIT = None`, Run All (~8 min on T4 at ~200 ms/call) → overwrites `data/results.json`.
5. Download `data/results.json` when done (Files pane → download, or `from google.colab import files; files.download('data/results.json')`).
6. To view `index.html`: download the repo back locally and serve the root statically (`python3 -m http.server`), or push to GH Pages — Colab preview alone won't resolve `./data/results.json`.

Local:
```bash
pip install "strands-decider[vision]" pillow
# smoke, then full — same LIMIT pattern inside the notebook
```

`DRY_RUN_NO_MODEL` in §3 checks cropping/timing without weights.

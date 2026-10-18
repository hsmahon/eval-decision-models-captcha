# Eval - Decision Models - CAPTCHA

<p>
  <img src="./assets/openai.svg" height="40" alt="OpenAI" />
  &nbsp;&nbsp;
  <img src="./assets/cloudflare.svg" height="40" alt="Cloudflare" />
  &nbsp;&nbsp;
  <a href="https://openrouter.ai"><img src="./assets/openrouter.svg" height="40" alt="OpenRouter" /></a>
</p>

How do decision models perform as visual decision complexity increases? I wanted to evaluate decision models on images to see if they could solve an infamous annoyance - CAPTCHAs.

> [!WARNING]
>
> This README was written by a human (me!), but all code changes, and
> additional documentation were authored entirely by Muse Spark 1.3 in OpenCode.

**[Live leaderboard →](https://hsmahon.github.io/eval-decision-models-captcha/)**

## Models (leaderboard)

Eight evaluated models, each tested on all 150 images (2,460 cell decisions).

| Model | Accuracy | Cost | p50 | p99 |
|---|---|---|---|---|
| Cloudflare Clef-flash 9B (`@cf/cloudflare/clef-flash`) | 0.9553 | $0.27 | 622 ms | 1,674 ms |
| OpenAI Decisions API (GPT-6 Luna, public beta) | 0.9435 | $0.10 | 226 ms | 1,802 ms |
| Cloudflare Clef-omni 30B-A3B (`@cf/cloudflare/clef-omni`) | 0.9585 | $0.11 | 391 ms | 1,372 ms |
| Jev 1.14 via OpenRouter (`typesafe/jev-1.14`) | 0.9443 | $0.068 | 301 ms | 1,571 ms |
| Liquid AI d1 | 0.9488 | $0.011906 | 480 ms | 3,256 ms |
| Liquid AI d1-ultra | 0.9606 | $0.025003 | 267 ms | 1,066 ms |
| Perplexity Decider Flash V1.2 via OpenRouter (`perplexity/pplx-decider-flash-v1.2`) | 0.9524 | $0.005633 | 312 ms | 1,193 ms |
| Perplexity Decider V1.2 via OpenRouter (`perplexity/pplx-decider-v1.2`) | 0.9524 | $0.011267 | 471 ms | 9,787 ms |

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

# Eval - Decision Models - CAPTCHA

<p>
  <img src="./assets/openai.svg" height="40" alt="OpenAI" />
  &nbsp;&nbsp;
  <img src="./assets/cloudflare.svg" height="40" alt="Cloudflare" />
</p>

How do decision models perform as visual decision complexity increases? I wanted to evaluate decision models on images to see if they could solve an infamous annoyance - CAPTCHAs.

> [!WARNING]
>
> This README was written by a human (me!), but all code changes, and
> additional documentation were authored entirely by Muse Spark 1.3 in OpenCode.

**[Live leaderboard →](https://hsmahon.github.io/eval-decision-models-captcha/)**

## Models (leaderboard)

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

## Reproducing the Eval
- **OpenAI**: create a key at `platform.openai.com` (needs billing credit), export `OPENAI_API_KEY`. Calls go to `POST https://api.openai.com/v1/decisions` with `model: gpt-6-luna` and one `predicate` question per cell crop (see the [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)).
- **Cloudflare**: from your Cloudflare dashboard, note your account ID and create an API token with the Workers AI permission; export `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Calls go to `POST https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/@cf/cloudflare/clef-flash` with a `noul` question plus native `criteria` (see [clef-flash docs](https://developers.cloudflare.com/workers-ai/models/clef-flash/)).
- **Jev via OpenRouter**: create an OpenRouter API key and export `OPENROUTER_API_KEY`. Install Pillow if needed (`python -m pip install Pillow`), then run `python eval_jev.py`; it pins `typesafe/jev-1.14`, sends one crop per `POST /api/alpha/decisions` request, and writes `data/results_jev.json`.
- **Perplexity via OpenRouter**: export `OPENROUTER_API_KEY`, install Pillow, then run `python eval_perplexity.py --model pplx-decider-flash-v1.2` and `python eval_perplexity.py --model pplx-decider-v1.2`. Each uses the pinned model ID through `/api/alpha/decisions` and saves a separate results file. Interrupted runs resume from per-cell checkpoints. The recorded Perplexity runs use `--workers 4` per model; latency is measured per request. Published input pricing is [$0.01/M for Flash](https://openrouter.ai/perplexity/pplx-decider-flash-v1.2) and [$0.02/M for Decider](https://openrouter.ai/perplexity/pplx-decider-v1.2); API-reported cost takes precedence.
- **Liquid AI**: create a key in the Liquid Console and export `LIQUID_API_KEY`. Run `python eval_liquid.py` to evaluate both pinned models (`d1` and `d1-ultra`) through Liquid's decisions API; results are written separately to `data/results_liquid-d1.json` and `data/results_liquid-d1-ultra.json`. Use `--model d1` or `--model d1-ultra` to run one model.

Each cell crop is sent as an inline base64 PNG for the OpenAI, Cloudflare, Jev, Perplexity, and Liquid backends. The fixed question is:
> Is a stoplight visible in this image?

## Build it Yourself

To set up your own agent to build on Cloudflare, follow the [Agent setup guide](https://developers.cloudflare.com/agent-setup/) — pick your agent (there's an [OpenCode guide](https://developers.cloudflare.com/agent-setup/opencode/); also Claude Code, Codex, Cursor, Copilot, and more), then:

```bash
curl -fsSL https://opencode.ai/install | bash   # install OpenCode
npx skills add https://github.com/cloudflare/skills  # teach it Cloudflare
```

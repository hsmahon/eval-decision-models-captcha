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

Accuracy over 2,460 cell decisions each; p50/p99 are per-decision-call latency in ms. Cost is the full-run total at $0.09/1M input tok (Clef) and $0.10/1M (OpenAI).

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

Both backends are plain HTTPS calls — no GPU needed. Get an API key for each:

- **OpenAI**: create a key at `platform.openai.com` (needs billing credit), export `OPENAI_API_KEY`. Calls go to `POST https://api.openai.com/v1/decisions` with `model: gpt-6-luna` and one `predicate` question per cell crop (see the [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)).
- **Cloudflare**: from your Cloudflare dashboard, note your account ID and create an API token with the Workers AI permission; export `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Calls go to `POST https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/@cf/cloudflare/clef-flash` with a `noul` question plus native `criteria` (see [clef-flash docs](https://developers.cloudflare.com/workers-ai/models/clef-flash/)).

Each cell crop is sent as an inline base64 PNG with the fixed question `Is a stoplight visible in this image?` and the true/false criteria in `eval.ipynb` §2. Results land in `data/results_<backend>.json` (same schema for both); push to `main` and the dashboard redeploys.

## Build it yourself with Cloudflare + your agent

This whole project was built by an agent. To set up your own agent to build on Cloudflare, follow the [Agent setup guide](https://developers.cloudflare.com/agent-setup/) — pick your agent (there's an [OpenCode guide](https://developers.cloudflare.com/agent-setup/opencode/); also Claude Code, Codex, Cursor, Copilot, and more), then:

```bash
curl -fsSL https://opencode.ai/install | bash   # install OpenCode
npx skills add https://github.com/cloudflare/skills  # teach it Cloudflare
```

```jsonc
// .opencode.jsonc — live Cloudflare API access via MCP
{ "mcp": { "cloudflare": { "type": "remote", "url": "https://mcp.cloudflare.com/mcp", "enabled": true } } }
```

The bundled skills cover Workers, Workers AI, Wrangler, and the `cf` CLI (`npm install -g cf`), so the agent can call any of 2,500+ Cloudflare API endpoints itself.

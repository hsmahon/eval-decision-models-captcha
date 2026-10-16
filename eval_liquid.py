#!/usr/bin/env python3
"""Run the fixed cell-level benchmark against Liquid AI d1 decision models."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval_jev import (
    BENCHMARK, CRITERIA, GRID_PADDING, CELL_PITCH, CELL_SIZE, IMAGES_DIR,
    LABELS_PATH, PROMPT, aggregate, load_labels, write_json,
)


ROOT = Path(__file__).resolve().parent
API_URL = "https://api.liquid.ai/decisions/v1/systemone"
MODELS = ("d1", "d1-ultra")
STATE = "Image crop for binary stoplight detection."


def decision_request(label: dict, cell_index: int, model: str) -> bytes:
    from io import BytesIO
    from PIL import Image

    image_path = IMAGES_DIR / f'{label["id"]}.png'
    with Image.open(image_path) as image:
        left = GRID_PADDING + (cell_index % label["cols"]) * CELL_PITCH
        right = left + CELL_SIZE
        top = GRID_PADDING + (cell_index // label["cols"]) * CELL_PITCH
        bottom = top + CELL_SIZE
        crop = image.crop((left, top, right, bottom))
        if crop.size != (CELL_SIZE, CELL_SIZE):
            raise RuntimeError(f'{label["id"]} cell {cell_index} is {crop.size}, expected {CELL_SIZE}x{CELL_SIZE}')
        encoded = BytesIO()
        crop.save(encoded, format="PNG")
        image_data = base64.b64encode(encoded.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "images": [f"data:image/png;base64,{image_data}"],
        "state": STATE,
        "questions": {"stoplight": {
            "type": "noul",
            "instructions": PROMPT,
            "criteria": CRITERIA,
        }},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def request_decision(label: dict, cell_index: int, model: str, token: str) -> tuple[dict, int, float]:
    body = decision_request(label, cell_index, model)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    started = time.perf_counter()
    raw = ""
    for attempt in range(6):
        req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in (408, 429, 499, 500, 502, 503, 504, 520, 522, 529) or attempt == 5:
                raise RuntimeError(f"Liquid API HTTP {error.code}: {detail[:1000]}") from error
            retry_after = error.headers.get("Retry-After", "")
            try:
                delay = min(60.0, max(1.0, float(retry_after)))
            except ValueError:
                delay = min(30.0, 2.0 ** attempt)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 5:
                raise RuntimeError(f"Liquid API request failed after retries: {error}") from error
            time.sleep(min(30.0, 2.0 ** attempt))
    elapsed_ms = (time.perf_counter() - started) * 1000
    response = json.loads(raw)
    actual_model = response.get("model") or model
    answer = response.get("answers", {}).get("stoplight", {})
    probability = answer.get("noul")
    if answer.get("type") != "noul" or not isinstance(probability, (int, float)):
        raise RuntimeError(f"Missing numeric Noul answer: {raw[:1000]}")
    probability = min(1.0, max(0.0, float(probability)))
    usage = response.get("usage", {})
    used_tokens = int(usage.get("input_tokens", usage.get("total_input_tokens", 0)) or 0)
    # Liquid bills d1 at $0.02/MTok and d1-ultra at $0.042/MTok.
    price_per_million = 0.02 if model == "d1" else 0.042
    cost = usage.get("cost", usage.get("cost_usd"))
    if cost is None:
        cost = used_tokens * price_per_million / 1_000_000
    return ({
        "image_id": label["id"], "grid": label["grid"], "rows": label["rows"],
        "cols": label["cols"], "difficulty": label["difficulty"], "cell_index": cell_index,
        "ground_truth": cell_index in label["positive_cells"],
        "prediction": probability >= 0.5,
        "confidence": round(probability if probability >= 0.5 else 1 - probability, 4),
        "latency_ms": round(elapsed_ms, 1), "model": model, "served_model": actual_model,
        "prompt": PROMPT, "raw": json.dumps(answer, separators=(",", ":")),
        "input_tokens": used_tokens, "cost_usd": float(cost),
    }, used_tokens, float(cost))


def run(model: str, args: argparse.Namespace, token: str) -> int:
    result_path = ROOT / f"data/results_liquid-{model}.json"
    labels = load_labels()
    if args.limit_images is not None:
        if args.limit_images < 1 or args.limit_images > len(labels):
            raise ValueError("--limit-images must be between 1 and 150")
        labels = labels[:args.limit_images]
    image_ids = [label["id"] for label in labels]
    expected_cells = sum(label["rows"] * label["cols"] for label in labels)
    run_suffix = "full" if len(labels) == 150 else f"smoke_{len(labels)}"
    progress_path = ROOT / f"data/.liquid_{model.replace('-', '_')}_{run_suffix}_progress.json"
    if args.fresh:
        progress_path.unlink(missing_ok=True)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("model") != model or progress.get("image_ids") != image_ids:
            raise RuntimeError(f"Progress file {progress_path.name} belongs to another run; pass --fresh")
        decisions = progress["decisions"]
        input_tokens = progress["input_tokens"]
        cost_usd = progress["cost_usd"]
        elapsed_before_resume = progress.get("elapsed_s", 0.0)
    else:
        decisions, input_tokens, cost_usd, elapsed_before_resume = [], 0, 0.0, 0.0

    completed = {(d["image_id"], d["cell_index"]) for d in decisions}
    started = time.monotonic()
    total = expected_cells
    for label in labels:
        for cell_index in range(label["rows"] * label["cols"]):
            if (label["id"], cell_index) in completed:
                continue
            decision, used_tokens, cost = request_decision(label, cell_index, model, token)
            decisions.append(decision)
            input_tokens += used_tokens
            cost_usd += cost
            write_json(progress_path, {
                "model": model, "image_ids": image_ids, "decisions": decisions,
                "input_tokens": input_tokens, "cost_usd": cost_usd,
                "elapsed_s": elapsed_before_resume + time.monotonic() - started,
            })
            print(f"{model}: {len(decisions)}/{total}: {label['id']} cell {cell_index} · {decision['latency_ms']:.0f} ms", flush=True)

    decisions.sort(key=lambda d: (d["image_id"], d["cell_index"]))
    overall, by_grid, by_grid_difficulty, per_image = aggregate(decisions, labels)
    full_run = len(decisions) == 2460 and len(labels) == 150
    result = {
        "schema_version": 1,
        "manifest": {
            "model": model,
            "served_models": sorted({d.get("served_model", model) for d in decisions}),
            "prompt": PROMPT, "prompt_version": "prompt-v2-neutral", "criteria": CRITERIA,
            "est_cost_usd": round(cost_usd, 6), "input_tokens_total": input_tokens,
            "temperature": "n/a (decision head)", "benchmark": BENCHMARK,
            "n_images_run": len({d["image_id"] for d in decisions}), "n_images_total": 150,
            "full_run": full_run, "backend": f"liquid-ai-{model}", "refusals": 0,
            "wall_s": round(elapsed_before_resume + time.monotonic() - started, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
        "overall": overall, "by_grid": by_grid,
        "by_grid_difficulty": by_grid_difficulty, "per_image": per_image, "decisions": decisions,
    }
    write_json(result_path, result)
    if full_run:
        progress_path.unlink(missing_ok=True)
    print(f"Wrote {result_path.relative_to(ROOT)}: {len(decisions)} cells, accuracy={overall['accuracy']}, cost=${cost_usd:.6f}")
    return 0 if len(decisions) == expected_cells else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS + ("both",), default="both")
    parser.add_argument("--limit-images", type=int, help="Run only the first N images as an API smoke run")
    parser.add_argument("--fresh", action="store_true", help="Discard previous partial progress")
    args = parser.parse_args()
    token = os.environ.get("LIQUID_API_KEY")
    if not token:
        print("Set LIQUID_API_KEY before running this evaluation.", file=sys.stderr)
        return 2
    models = MODELS if args.model == "both" else (args.model,)
    for model in models:
        status = run(model, args, token)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the fixed cell-level benchmark against Jev 1.14 through OpenRouter."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "data/labels.jsonl"
IMAGES_DIR = ROOT / "data/images"
RESULT_PATH = ROOT / "data/results_jev.json"
PROGRESS_PATH = ROOT / "data/.jev_progress.json"
MODEL = "typesafe/jev-1.14"
API_URL = "https://openrouter.ai/api/alpha/decisions"
PROMPT = "Is a stoplight visible in this image?"
CRITERIA = {
    "true": "A traffic stoplight is visible, whole or partial — including cropped edges, night shots, or small/distant signals",
    "false": "No stoplight visible — including lookalikes such as street lamps, car tail lights, or traffic signs without lights",
}
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
BENCHMARK = "captcha_vision_benchmark_150 v1"
CELL_SIZE = 320
CELL_PITCH = 328
GRID_PADDING = 8


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def wilson_interval(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    phat = correct / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return [round(max(0, center - margin), 4), round(min(1, center + margin), 4)]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def summarize(decisions: list[dict]) -> dict:
    n = len(decisions)
    correct = sum(d["prediction"] == d["ground_truth"] for d in decisions)
    positives = [d for d in decisions if d["ground_truth"]]
    predicted_positive = [d for d in decisions if d["prediction"]]
    tp = sum(d["prediction"] and d["ground_truth"] for d in decisions)
    precision = tp / len(predicted_positive) if predicted_positive else 0.0
    recall = tp / len(positives) if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    conf = [d["confidence"] for d in decisions]
    lats = [d["latency_ms"] for d in decisions]
    buckets = {}
    for name, lo, hi in [("0-20", 0, .2), ("20-40", .2, .4), ("40-60", .4, .6),
                         ("60-80", .6, .8), ("80-100", .8, 1.000001)]:
        members = [d for d in decisions if lo <= d["confidence"] < hi]
        buckets[name] = {
            "n": len(members),
            "acc": rounded(sum(d["prediction"] == d["ground_truth"] for d in members) / len(members)) if members else None,
        }
    return {
        "n_images": len({d["image_id"] for d in decisions}),
        "n_cells": n,
        "accuracy": rounded(correct / n) if n else None,
        "ci95": wilson_interval(correct, n),
        "precision_pos": rounded(precision),
        "recall_pos": rounded(recall),
        "f1_pos": rounded(f1),
        "avg_confidence": rounded(statistics.fmean(conf)) if conf else None,
        "lat_mean": round(statistics.fmean(lats), 1) if lats else None,
        "lat_median": round(statistics.median(lats), 1) if lats else None,
        "lat_p95": round(percentile(lats, .95), 1) if lats else None,
        "lat_p99": round(percentile(lats, .99), 1) if lats else None,
        "conf_buckets": buckets,
    }


def aggregate(decisions: list[dict], labels: list[dict]) -> tuple[dict, dict, dict, list[dict]]:
    by_grid: dict[str, list[dict]] = defaultdict(list)
    by_grid_difficulty: dict[str, list[dict]] = defaultdict(list)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for decision in decisions:
        by_grid[decision["grid"]].append(decision)
        by_grid_difficulty[f'{decision["grid"]}/{decision["difficulty"]}'].append(decision)
        by_image[decision["image_id"]].append(decision)

    image_labels = {label["id"]: label for label in labels}
    per_image = []
    for image_id, image_decisions in by_image.items():
        label = image_labels[image_id]
        n = len(image_decisions)
        per_image.append({
            "image_id": image_id,
            "grid": label["grid"],
            "difficulty": label["difficulty"],
            "rows": label["rows"],
            "cols": label["cols"],
            "cell_acc": rounded(sum(d["prediction"] == d["ground_truth"] for d in image_decisions) / n) if n else None,
            "full_success": all(d["prediction"] == d["ground_truth"] for d in image_decisions),
            "avg_conf": rounded(statistics.fmean(d["confidence"] for d in image_decisions)) if n else None,
            "total_ms": round(sum(d["latency_ms"] for d in image_decisions), 1),
            "n_cells": n,
            "positive_cells": label["positive_cells"],
        })
    per_image.sort(key=lambda row: row["image_id"])
    return (
        summarize(decisions),
        {grid: summarize(rows) for grid, rows in sorted(by_grid.items())},
        {group: summarize(rows) for group, rows in sorted(by_grid_difficulty.items())},
        per_image,
    )


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_labels() -> list[dict]:
    with LABELS_PATH.open() as stream:
        labels = [json.loads(line) for line in stream if line.strip()]
    if len(labels) != 150:
        raise RuntimeError(f"Expected 150 labels, found {len(labels)}")
    return labels


def decision_request(label: dict, cell_index: int) -> bytes:
    image_path = IMAGES_DIR / f'{label["id"]}.png'
    with Image.open(image_path) as image:
        width, height = image.size
        left = GRID_PADDING + (cell_index % label["cols"]) * CELL_PITCH
        right = left + CELL_SIZE
        top = GRID_PADDING + (cell_index // label["cols"]) * CELL_PITCH
        bottom = top + CELL_SIZE
        crop = image.crop((left, top, right, bottom))
        if crop.size != (CELL_SIZE, CELL_SIZE):
            raise RuntimeError(f'{label["id"]} cell {cell_index} is {crop.size}, expected {CELL_SIZE}x{CELL_SIZE}')
        from io import BytesIO
        encoded = BytesIO()
        crop.save(encoded, format="PNG")
        image_data = base64.b64encode(encoded.getvalue()).decode("ascii")
    payload = {
        "model": MODEL,
        "state": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}],
        "questions": {
            "stoplight": {
                "type": "noul",
                "instructions": PROMPT,
                "criteria": CRITERIA,
            }
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def request_decision(label: dict, cell_index: int, token: str) -> tuple[dict, int]:
    body = decision_request(label, cell_index)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hsmahon.github.io/eval-decision-models-captcha/",
        "X-OpenRouter-Title": "Eval Decision Models CAPTCHA",
    }
    started = time.perf_counter()
    for attempt in range(6):
        req = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in (408, 429, 499, 500, 502, 503, 504, 520, 522, 529) or attempt == 5:
                raise RuntimeError(f"OpenRouter API HTTP {error.code}: {detail[:1000]}") from error
            retry_after = error.headers.get("Retry-After", "")
            try:
                delay = min(60.0, max(1.0, float(retry_after)))
            except ValueError:
                delay = min(30.0, 2.0 ** attempt)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 5:
                raise RuntimeError(f"OpenRouter API request failed after retries: {error}") from error
            time.sleep(min(30.0, 2.0 ** attempt))
    elapsed_ms = (time.perf_counter() - started) * 1000
    response = json.loads(raw)
    actual_model = response.get("model")
    if actual_model and actual_model != MODEL and not actual_model.startswith(f"{MODEL}-"):
        raise RuntimeError(f"Requested {MODEL}, but API answered with {actual_model!r}")
    answer = response.get("answers", {}).get("stoplight", {})
    if answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
        raise RuntimeError(f"Missing numeric Noul answer: {raw[:1000]}")
    probability = min(1.0, max(0.0, float(answer["noul"])))
    prediction = probability >= 0.5
    usage = response.get("usage", {})
    used_tokens = int(usage.get("input_tokens", 0))
    cost = usage.get("cost", usage.get("cost_usd"))
    if cost is None:
        cost = used_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000
    return {
        "image_id": label["id"],
        "grid": label["grid"],
        "rows": label["rows"],
        "cols": label["cols"],
        "difficulty": label["difficulty"],
        "cell_index": cell_index,
        "ground_truth": cell_index in label["positive_cells"],
        "prediction": prediction,
        "confidence": round(probability if prediction else 1 - probability, 4),
        "latency_ms": round(elapsed_ms, 1),
        "model": MODEL,
        "served_model": actual_model or MODEL,
        "prompt": PROMPT,
        "raw": json.dumps(answer, separators=(",", ":")),
        "input_tokens": used_tokens,
        "cost_usd": float(cost),
    }, used_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-images", type=int, help="Run only the first N labeled images (for an API smoke run)")
    parser.add_argument("--fresh", action="store_true", help="Discard a previous partial run and start over")
    args = parser.parse_args()
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        print("Set OPENROUTER_API_KEY before running this evaluation.", file=sys.stderr)
        return 2
    labels = load_labels()
    if args.limit_images is not None:
        if args.limit_images < 1 or args.limit_images > len(labels):
            parser.error("--limit-images must be between 1 and 150")
        labels = labels[:args.limit_images]
    expected_cells = sum(label["rows"] * label["cols"] for label in labels)

    if args.fresh:
        PROGRESS_PATH.unlink(missing_ok=True)
    if PROGRESS_PATH.exists():
        progress = json.loads(PROGRESS_PATH.read_text())
        if progress.get("model") != MODEL or progress.get("image_ids") != [x["id"] for x in labels]:
            raise RuntimeError("Progress file belongs to a different model or image selection; remove it or pass --fresh")
        decisions = progress["decisions"]
        input_tokens = progress["input_tokens"]
        cost_usd = progress.get("cost_usd", input_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000)
        elapsed_before_resume = progress.get("elapsed_s", 0.0)
    else:
        decisions = []
        input_tokens = 0
        cost_usd = 0.0
        elapsed_before_resume = 0.0

    completed = {(d["image_id"], d["cell_index"]) for d in decisions}
    started = time.monotonic()
    total = sum(x["rows"] * x["cols"] for x in labels)
    for label in labels:
        for cell_index in range(label["rows"] * label["cols"]):
            if (label["id"], cell_index) in completed:
                continue
            decision, used_tokens = request_decision(label, cell_index, token)
            decisions.append(decision)
            input_tokens += used_tokens
            cost_usd += decision["cost_usd"]
            write_json(PROGRESS_PATH, {
                "model": MODEL,
                "image_ids": [x["id"] for x in labels],
                "decisions": decisions,
                "input_tokens": input_tokens,
                "cost_usd": cost_usd,
                "elapsed_s": elapsed_before_resume + time.monotonic() - started,
            })
            print(f"{len(decisions)}/{total}: {label['id']} cell {cell_index} · {decision['latency_ms']:.0f} ms")

    decisions.sort(key=lambda d: (d["image_id"], d["cell_index"]))
    overall, by_grid, by_grid_difficulty, per_image = aggregate(decisions, labels)
    full_run = len(decisions) == 2460 and len(labels) == 150
    result = {
        "schema_version": 1,
        "manifest": {
            "model": MODEL,
            "served_models": sorted({d.get("served_model", MODEL) for d in decisions}),
            "prompt": PROMPT,
            "prompt_version": "prompt-v2-neutral",
            "criteria": CRITERIA,
            "est_cost_usd": round(cost_usd, 6),
            "input_tokens_total": input_tokens,
            "temperature": "n/a (decision head)",
            "benchmark": BENCHMARK,
            "n_images_run": len({d["image_id"] for d in decisions}),
            "n_images_total": 150,
            "full_run": full_run,
            "backend": "openrouter-typesafe-jev",
            "refusals": 0,
            "wall_s": round(elapsed_before_resume + time.monotonic() - started, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
        "overall": overall,
        "by_grid": by_grid,
        "by_grid_difficulty": by_grid_difficulty,
        "per_image": per_image,
        "decisions": decisions,
    }
    write_json(RESULT_PATH, result)
    if full_run:
        PROGRESS_PATH.unlink(missing_ok=True)
    print(f"Wrote {RESULT_PATH.relative_to(ROOT)}: {len(decisions)} cells, accuracy={overall['accuracy']}, cost=${result['manifest']['est_cost_usd']:.6f}")
    return 0 if len(decisions) == expected_cells else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared eval harness: crop cells from grid images, ask one backend per cell,
write data/results_<backend>.json in the same schema as eval.ipynb §4.

Usage:
    LIMIT=5 python3 scripts/run_openai.py        # smoke: 5 images
    LIMIT=None python3 scripts/run_openai.py      # full: 150 images
    PROBE=1 python3 scripts/run_openai.py         # single cell, prints raw response

Env: OPENAI_API_KEY (or pass --api-key). Never commit keys.
"""

import argparse
import base64
import datetime
import io
import json
import math
import os
import statistics
import sys
import time
import urllib.request

PROMPT = "Is a stoplight visible in this image?"
PROMPT_VERSION = "prompt-v2-neutral"
CRITERIA = {
    "true": "A traffic stoplight is visible, whole or partial — including cropped edges, night shots, or small/distant signals",
    "false": "No stoplight visible — including lookalikes such as street lamps, car tail lights, or traffic signs without lights",
}
MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/decisions"
COST_PER_MTOK = 0.10  # $/1M input tokens, decisions endpoint
TOKEN_PER_CELL_EST = 1200  # refined from usage totals at runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load_items(limit):
    items = []
    with open(os.path.join(DATA, "labels.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit is not None:
        items = items[:limit]
    return items


def crop_cells(img_path, rows_n, cols_n):
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    cw, ch = W // cols_n, H // rows_n
    out = []
    for r in range(rows_n):
        for c in range(cols_n):
            x1, y1 = c * cw, r * ch
            x2 = (c + 1) * cw if c < cols_n - 1 else W
            y2 = (r + 1) * ch if r < rows_n - 1 else H
            buf = io.BytesIO()
            im.crop((x1, y1, x2, y2)).save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def ask_cell(b64, api_key, retries=4):
    body = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect this grid square."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                ],
            }
        ],
        "questions": [
            {
                "type": "predicate",
                "name": "stoplight",
                "instructions": f"{PROMPT} Answer based only on visible pixels. "
                f"YES when: {CRITERIA['true']}. NO when: {CRITERIA['false']}.",
            }
        ],
    }
    last_err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                ENDPOINT,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            ms = (time.perf_counter() - t0) * 1000.0
            answers = payload.get("answers", payload.get("data", {}).get("answers", []))
            ans = answers[0] if isinstance(answers, list) else answers.get("stoplight", {})
            if isinstance(ans, dict) and ans.get("type") == "refusal":
                return None, None, "refusal", ms, payload
            p = float(ans.get("probability", ans.get("value")))
            return (p >= 0.5), (p if p >= 0.5 else 1 - p), json.dumps(ans)[:120], ms, payload
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ask_cell failed after {retries} retries: {last_err}")


def wilson(p, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0, (c - m) / d), 4), round(min(1, (c + m) / d), 4)]


def summarize(ds):
    n = len(ds)
    acc = sum(d["prediction"] == d["ground_truth"] for d in ds) / max(1, n)
    tp = sum(d["prediction"] and d["ground_truth"] for d in ds)
    fp = sum(d["prediction"] and not d["ground_truth"] for d in ds)
    fn = sum((not d["prediction"]) and d["ground_truth"] for d in ds)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    lat = sorted(d["latency_ms"] for d in ds)
    confs = [d["confidence"] for d in ds if d["confidence"] is not None]
    buckets = {}
    for lo in [0, 20, 40, 60, 80]:
        b = [
            d
            for d in ds
            if d["confidence"] is not None
            and (lo / 100 <= d["confidence"] < (lo + 20) / 100 or (lo == 80 and d["confidence"] == 1.0))
        ]
        buckets[f"{lo}-{lo+20}"] = {
            "n": len(b),
            "acc": (round(sum(d["prediction"] == d["ground_truth"] for d in b) / len(b), 4) if b else None),
        }
    return {
        "n_images": len(set(d["image_id"] for d in ds)),
        "n_cells": n,
        "accuracy": round(acc, 4),
        "ci95": wilson(acc, n),
        "precision_pos": round(prec, 4),
        "recall_pos": round(rec, 4),
        "f1_pos": round(f1, 4),
        "avg_confidence": (round(sum(confs) / len(confs), 4) if confs else None),
        "lat_mean": (round(statistics.mean(lat), 1) if lat else None),
        "lat_median": (round(statistics.median(lat), 1) if lat else None),
        "lat_p95": (round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None),
        "conf_buckets": buckets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--out", default=os.path.join(DATA, "results_openai.json"))
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("Set OPENAI_API_KEY env or pass --api-key")

    limit_env = os.environ.get("LIMIT", "5")
    limit = None if limit_env in ("None", "none", "") else int(limit_env)
    items = load_items(limit)
    print(f"images: {len(items)} (LIMIT={limit_env})")

    if os.environ.get("PROBE") == "1":
        r = items[0]
        b64 = crop_cells(os.path.join(DATA, r["image"]), r["rows"], r["cols"])[0]
        print(json.dumps(ask_cell(b64, args.api_key)[4], indent=1)[:2000])
        return

    decisions, refusals = [], 0
    t_start = time.perf_counter()
    for i, r in enumerate(items):
        positives = set(r["positive_cells"])
        for idx, b64 in enumerate(crop_cells(os.path.join(DATA, r["image"]), r["rows"], r["cols"])):
            pred, conf, raw, ms, _payload = ask_cell(b64, args.api_key)
            if pred is None:
                refusals += 1
                continue
            decisions.append(
                {
                    "image_id": r["id"],
                    "grid": r["grid"],
                    "rows": r["rows"],
                    "cols": r["cols"],
                    "difficulty": r["difficulty"],
                    "cell_index": idx,
                    "ground_truth": idx in positives,
                    "prediction": pred,
                    "confidence": round(conf, 4) if conf is not None else None,
                    "latency_ms": round(ms, 1),
                    "model": MODEL,
                    "prompt": PROMPT,
                    "raw": raw,
                }
            )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} images, {len(decisions)} cells")
    wall_s = time.perf_counter() - t_start

    total_in_tok = len(decisions) * TOKEN_PER_CELL_EST
    est_cost = round(total_in_tok / 1e6 * COST_PER_MTOK, 4)
    by_grid = {g: summarize([d for d in decisions if d["grid"] == g]) for g in ["3x3", "3x4", "4x4", "4x5", "5x5"]}
    by_gd = {
        f"{g}/{df}": summarize([d for d in decisions if d["grid"] == g and d["difficulty"] == df])
        for g in ["3x3", "3x4", "4x4", "4x5", "5x5"]
        for df in ["easy", "medium", "hard"]
    }
    per_image = []
    for r in items:
        ds = [d for d in decisions if d["image_id"] == r["id"]]
        s = summarize(ds) if ds else {"accuracy": None, "avg_confidence": None}
        per_image.append(
            {
                "image_id": r["id"],
                "grid": r["grid"],
                "difficulty": r["difficulty"],
                "rows": r["rows"],
                "cols": r["cols"],
                "cell_acc": s.get("accuracy"),
                "full_success": (s.get("accuracy") == 1.0) if s.get("accuracy") is not None else None,
                "avg_conf": s.get("avg_confidence"),
                "total_ms": round(sum(d["latency_ms"] for d in ds), 1) if ds else None,
                "n_cells": len(ds),
                "positive_cells": r["positive_cells"],
            }
        )
    results = {
        "schema_version": 1,
        "manifest": {
            "model": MODEL,
            "prompt": PROMPT,
            "prompt_version": PROMPT_VERSION,
            "criteria": CRITERIA,
            "est_cost_usd": est_cost,
            "temperature": "n/a (decision head)",
            "benchmark": "captcha_vision_benchmark_150 v1",
            "n_images_run": len(items),
            "n_images_total": 150,
            "full_run": limit is None,
            "backend": "openai-decisions",
            "refusals": refusals,
            "wall_s": round(wall_s, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "overall": summarize(decisions),
        "by_grid": by_grid,
        "by_grid_difficulty": by_gd,
        "per_image": sorted(per_image, key=lambda x: (x["cell_acc"] is None, x["cell_acc"] or 0)),
        "decisions": decisions,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}: {len(decisions)} decisions, refusals={refusals}")
    print(json.dumps({"overall": results["overall"], "by_grid": {k: v["accuracy"] for k, v in by_grid.items()}}, indent=1))
    print(f"est cost: ${est_cost} | wall: {wall_s:.0f}s")


if __name__ == "__main__":
    main()

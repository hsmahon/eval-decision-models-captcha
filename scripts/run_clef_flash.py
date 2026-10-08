"""Clef-flash 9B runner via Cloudflare Workers AI. Same schema as run_openai.py.

Usage:
    LIMIT=5 PROBE=1 python3 scripts/run_clef_flash.py   # single cell probe
    LIMIT=5 python3 scripts/run_clef_flash.py            # smoke
    LIMIT=None python3 scripts/run_clef_flash.py         # full

Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN. Never commit keys.
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_openai as H  # noqa: E402  (shared harness: crops, metrics, schema)

MODEL_ID = "clef-flash"
COST_PER_MTOK = 0.09  # $/1M input tokens (decisions score in one prefill pass)
TOKEN_PER_CELL_EST = 1200


def endpoint(account_id):
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/{MODEL_ID}"


def ask_cell(b64, account_id, token, retries=4):
    body = {
        "model": MODEL_ID,
        "state": "single CAPTCHA-grid cell crop",
        "questions": {
            "stoplight": {
                "type": "noul",
                "instructions": f"{H.PROMPT} Answer based only on visible pixels. "
                f"YES when: {H.CRITERIA['true']}. NO when: {H.CRITERIA['false']}.",
            }
        },
        "images": [{"content_type": "image/png", "base64": b64}],
    }
    last_err = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                endpoint(account_id),
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            ms = (time.perf_counter() - t0) * 1000.0
            if not payload.get("success", True):
                raise RuntimeError(str(payload.get("errors", payload))[:200])
            ans = payload["result"]["answers"]["stoplight"]
            p = float(ans.get("value", ans.get("noul")))
            conf = ans.get("confidence")
            conf = float(conf) if conf is not None else (p if p >= 0.5 else 1 - p)
            return (p >= 0.5), conf, json.dumps(ans)[:120], ms, payload
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ask_cell failed after {retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-id", default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    ap.add_argument("--api-token", default=os.environ.get("CLOUDFLARE_API_TOKEN"))
    ap.add_argument("--out", default=os.path.join(H.DATA, "results_clef-flash.json"))
    args = ap.parse_args()
    if not args.account_id or not args.api_token:
        sys.exit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN env (or flags)")

    limit_env = os.environ.get("LIMIT", "5")
    limit = None if limit_env in ("None", "none", "") else int(limit_env)
    items = H.load_items(limit)
    print(f"images: {len(items)} (LIMIT={limit_env})")

    if os.environ.get("PROBE") == "1":
        r = items[0]
        b64 = H.crop_cells(os.path.join(H.DATA, r["image"]), r["rows"], r["cols"])[0]
        print(json.dumps(ask_cell(b64, args.account_id, args.api_token)[4], indent=1)[:2000])
        return

    decisions = []
    prompt_tok_total = 0
    t_start = time.perf_counter()
    for i, r in enumerate(items):
        positives = set(r["positive_cells"])
        for idx, b64 in enumerate(H.crop_cells(os.path.join(H.DATA, r["image"]), r["rows"], r["cols"])):
            pred, conf, raw, ms, payload = ask_cell(b64, args.account_id, args.api_token)
            try:
                prompt_tok_total += payload["result"].get("usage", {}).get("prompt_tokens", 0) or 0
            except (KeyError, TypeError):
                pass
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
                    "model": MODEL_ID,
                    "prompt": H.PROMPT,
                    "raw": raw,
                }
            )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} images, {len(decisions)} cells")
    wall_s = time.perf_counter() - t_start

    if prompt_tok_total:
        est_cost = round(prompt_tok_total / 1e6 * COST_PER_MTOK, 4)
    else:
        est_cost = round(len(decisions) * TOKEN_PER_CELL_EST / 1e6 * COST_PER_MTOK, 4)
    by_grid = {g: H.summarize([d for d in decisions if d["grid"] == g]) for g in ["3x3", "3x4", "4x4", "4x5", "5x5"]}
    by_gd = {
        f"{g}/{df}": H.summarize([d for d in decisions if d["grid"] == g and d["difficulty"] == df])
        for g in ["3x3", "3x4", "4x4", "4x5", "5x5"]
        for df in ["easy", "medium", "hard"]
    }
    per_image = []
    for r in items:
        ds = [d for d in decisions if d["image_id"] == r["id"]]
        s = H.summarize(ds) if ds else {"accuracy": None, "avg_confidence": None}
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
            "model": MODEL_ID,
            "prompt": H.PROMPT,
            "prompt_version": H.PROMPT_VERSION,
            "criteria": H.CRITERIA,
            "est_cost_usd": est_cost,
            "prompt_tokens_total": prompt_tok_total or None,
            "temperature": "n/a (decision head)",
            "benchmark": "captcha_vision_benchmark_150 v1",
            "n_images_run": len(items),
            "n_images_total": 150,
            "full_run": limit is None,
            "backend": "clef-flash",
            "wall_s": round(wall_s, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "overall": H.summarize(decisions),
        "by_grid": by_grid,
        "by_grid_difficulty": by_gd,
        "per_image": sorted(per_image, key=lambda x: (x["cell_acc"] is None, x["cell_acc"] or 0)),
        "decisions": decisions,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}: {len(decisions)} decisions")
    print(json.dumps({"overall": results["overall"], "by_grid": {k: v["accuracy"] for k, v in by_grid.items()}}, indent=1))
    print(f"est cost: ${est_cost} | wall: {wall_s:.0f}s")


if __name__ == "__main__":
    main()

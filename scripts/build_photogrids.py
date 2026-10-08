"""Rebuild the 150 benchmark grids from real photo crops (COCO via HF mirror).

Keeps data/labels.jsonl identical (IDs, grids, difficulty, positive_cells,
per-cell labels) — only pixels change. Difficulty is made visual: easy cells
get large prominent instances, hard cells get small/cluttered ones.

Source: HuggingFace `detection-datasets/coco` (COCO 2017 train images +
instances; research use, see data/ATTRIBUTION.md).
Category map: car=3, bicycle=2, bus=6, stoplight=10 (traffic light),
hydrant=11 (fire hydrant), sign=13 (stop sign), empty=background crop.

Usage: python3 scripts/build_photogrids.py
Env: needs pillow + pyarrow + huggingface_hub (for the initial shard fetch).
"""
import glob
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = DATA / "labels.jsonl"
OUT = DATA / "images"

CELL = 320          # px per cell (was ~160 in cartoon set)
GUTTER = 8
BG = (245, 245, 245)
JPEG_Q = 88

CAT = {"car": 2, "bicycle": 1, "bus": 5, "stoplight": 9,
       "hydrant": 10, "sign": 11}
STREET_CATS = {2, 3, 5, 7, 9, 11}  # car, motorcycle, bus, truck, traffic light, stop sign
random.seed(20261008)


def square_crop(im, x, y, w, h, context=0.45, min_frac=0.06):
    """Square crop around a box with context margin, clamped to image."""
    W, H = im.size
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + context)
    side = max(side, min(W, H) * min_frac)
    side = min(side, W, H)
    x1 = min(max(0, cx - side / 2), W - side)
    y1 = min(max(0, cy - side / 2), H - side)
    crop = im.crop((int(x1), int(y1), int(x1 + side), int(y1 + side)))
    return crop


def collect_pool(shard_paths):
    """Pass 1: gather (area_ratio, image_bytes, bbox) per category."""
    import pyarrow.parquet as pq
    pool = defaultdict(list)   # label -> [(area_ratio, img_bytes, W, H, bbox)]
    bg_candidates = []         # (img_bytes, W, H, boxes)
    for sp in shard_paths:
        t = pq.read_table(sp, columns=["image_id", "image", "width",
                                       "height", "objects"])
        for i in range(t.num_rows):
            img_bytes = t.column("image")[i].as_py()["bytes"]
            W = t.column("width")[i].as_py()
            H = t.column("height")[i].as_py()
            obj = t.column("objects")[i].as_py()
            cats, boxes, areas = obj["category"], obj["bbox"], obj["area"]
            seen_bg = False
            for c, b, a in zip(cats, boxes, areas):
                # NOTE: this mirror stores boxes as CORNERS [x1,y1,x2,y2]
                # (verified: (x2-x1)*(y2-y1) ~= area; w*h-as-xywh mismatches
                # 99.8% of boxes). Convert to [x,y,w,h] here.
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                if w < 16 or h < 16:
                    continue
                for label, cid in CAT.items():
                    if c == cid:
                        pool[label].append((a / (W * H), img_bytes, W, H,
                                            [x1, y1, w, h]))
            if len(cats) <= 6 and not seen_bg:
                if any(c in STREET_CATS for c in cats):
                    bg_candidates.append((img_bytes, W, H, boxes))
        del t
    return pool, bg_candidates


def overlaps(b, boxes):
    x, y, w, h = b
    for q in boxes:
        ix = max(0, min(x + w, q[0] + q[2]) - max(x, q[0]))
        iy = max(0, min(y + h, q[1] + q[3]) - max(y, q[1]))
        if ix * iy > 0.05 * w * h:
            return True
    return False


def bg_crop(img_bytes, W, H, boxes):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    side = min(W, H) * random.uniform(0.25, 0.45)
    for _ in range(12):
        x = random.uniform(0, W - side)
        y = random.uniform(0, H - side)
        if not overlaps((x, y, side, side), boxes):
            return im.crop((int(x), int(y), int(x + side), int(y + side)))
    return None


def main():
    rows = [json.loads(l) for l in open(LABELS)]
    shards = sorted(glob.glob(
        str(Path.home() / ".cache/huggingface/hub/datasets--detection-datasets--coco"
             "/snapshots/*/data/train-0000[0-3]-*.parquet")))
    assert len(shards) >= 4, f"need 4 COCO shards, have {shards}"
    print("collecting pool from", len(shards), "shards...")
    pool, bg_candidates = collect_pool(shards)
    for label, need in [("car", 388), ("bicycle", 417), ("bus", 327),
                        ("stoplight", 277), ("hydrant", 347), ("sign", 314)]:
        print(f"  {label}: have {len(pool[label])}, need {need}")

    # Demand per label per difficulty (difficulty-aware assignment).
    demand = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for cell in r["cells"]:
            demand[cell["label"]][r["difficulty"]] += 1

    # Sort pool: easy <- largest instances, hard <- smallest.
    cell_img = {}   # (label, difficulty, k) -> PIL cell
    img_cache = {}
    for label in list(CAT) + ["empty"]:
        if label == "empty":
            continue
        insts = sorted(pool[label], key=lambda t: -t[0])
        n = sum(demand[label].values())
        # Top-up short pools (hydrant, stop sign) by cycling instances —
        # each reuse gets a fresh random context margin + alternating flip.
        repeats = []
        i = 0
        while len(insts) + len(repeats) < n:
            repeats.append(insts[i % len(insts)])
            i += 1
        insts = insts + repeats
        thirds = {"easy": insts[: len(insts) // 3] or insts,
                  "medium": insts[len(insts) // 3: 2 * len(insts) // 3] or insts,
                  "hard": insts[2 * len(insts) // 3:] or insts}
        for diff in ("easy", "medium", "hard"):
            lst = thirds[diff][:]
            random.shuffle(lst)
            for k in range(demand[label][diff]):
                area, img_bytes, W, H, b = lst[k % len(lst)]
                flip = (k // len(lst)) % 2 == 1  # alternate flip on reuses
                key = id(img_bytes)
                im = img_cache.get(key)
                if im is None:
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_cache[key] = im
                crop = square_crop(im, *b,
                                   context=random.uniform(0.3, 0.6))
                if flip:
                    crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                cell_img[(label, diff, k)] = crop.resize((CELL, CELL),
                                                        Image.LANCZOS)
            print(f"  assigned {label}/{diff}: {demand[label][diff]}")

    # Empty cells from background regions.
    n_empty = sum(demand["empty"].values())
    empties = []
    random.shuffle(bg_candidates)
    for img_bytes, W, H, boxes in bg_candidates:
        if len(empties) >= n_empty:
            break
        for _ in range(2):
            c = bg_crop(img_bytes, W, H, boxes)
            if c is not None:
                empties.append(c.resize((CELL, CELL), Image.LANCZOS))
                if len(empties) >= n_empty:
                    break
    assert len(empties) >= n_empty, f"empty crops {len(empties)} < {n_empty}"
    print(f"  assigned empty: {n_empty}")

    # Compose grids — same filenames/IDs as before.
    counters = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cells = []
        for cell in r["cells"]:
            label = cell["label"]
            if label == "empty":
                cells.append(empties.pop())
            else:
                k = counters[label][r["difficulty"]]
                cells.append(cell_img[(label, r["difficulty"], k)])
                counters[label][r["difficulty"]] += 1
        gw, gh = r["cols"] * CELL + (r["cols"] + 1) * GUTTER, \
            r["rows"] * CELL + (r["rows"] + 1) * GUTTER
        grid = Image.new("RGB", (gw, gh), BG)
        it = iter(cells)
        for rr in range(r["rows"]):
            for cc in range(r["cols"]):
                grid.paste(next(it),
                           (GUTTER + cc * (CELL + GUTTER),
                            GUTTER + rr * (CELL + GUTTER)))
        grid.save(OUT / f'{r["id"]}.png')
    print("wrote 150 grids to", OUT)


if __name__ == "__main__":
    main()

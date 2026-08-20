#!/usr/bin/env python
"""Stub generator -- exercises the whole pipeline without a diffusion model.

Every part of a real run except the model call is code we wrote and can get
wrong: manifest reading, shard selection, resume, filename convention, the
write-then-rename, error capture, and the entire scoring and aggregation path.
Debugging any of that on a queue with a 3.5 hour wait and a 24 hour cap is the
expensive way to find out. This produces real PNGs cheaply so the end-to-end
path can be proven on a laptop first.

It deliberately mirrors gen_infu.py's CLI so the same SLURM plumbing and the
same manifests drive both.

Modes, chosen per cell so one run covers every downstream branch:
  preserve  -- reference face composited onto a plain canvas. ID Loss should be
               near zero; that is a positive control for the ID metric itself.
  masked    -- photoreal face on a heavily stylised background. This is the FMI
               positive control configs/metrics.yaml demands: FMI must come out
               strongly positive.
  blended   -- face stylised along with the background. FMI should be near zero.
  noface    -- no face at all. Exercises the undefined-metric path, which is a
               real production branch under heavy stylisation.

Usage:
  python src/gen_stub.py --manifest tests/fixtures/manifest_test.parquet \\
      --out tests/fixtures/out/stub --mode auto
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import RunLog, cell_to_file, select_work  # noqa: E402

MODES = ["preserve", "masked", "blended", "noface"]


def stylise(img: Image.Image, strength: int = 5) -> Image.Image:
    """Painterly non-photoreal look.

    Posterise-and-blur was too weak: CLIP scored it barely differently from the
    original, so the FMI control could not separate a masked face from a blended
    one. OpenCV's edge-aware NPR stylisation produces a genuinely painterly
    result that CLIP reads as such.
    """
    import cv2
    import numpy as np
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = cv2.stylization(arr, sigma_s=60, sigma_r=0.45)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def canvas(w: int, h: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    base = Image.new("RGB", (w, h), (rng.randint(60, 200),
                                     rng.randint(60, 200),
                                     rng.randint(60, 200)))
    d = ImageDraw.Draw(base)
    for _ in range(24):                      # some structure so crops differ
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        d.ellipse([x0, y0, x0 + rng.randint(40, 260), y0 + rng.randint(40, 260)],
                  fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    return base.filter(ImageFilter.GaussianBlur(radius=6))


def compose(ref: Image.Image, w: int, h: int, seed: int, mode: str) -> Image.Image:
    bg = canvas(w, h, seed)
    if mode == "noface":
        return bg
    if mode in ("masked", "blended"):
        bg = stylise(bg, strength=6)

    # a face large enough for the detector, placed in the upper-middle third
    side = int(min(w, h) * 0.55)
    face = ref.convert("RGB").resize((side, side), Image.LANCZOS)
    if mode == "blended":
        face = stylise(face, strength=6)
    bg.paste(face, ((w - side) // 2, int(h * 0.12)))
    return bg


def pick_mode(cell: str, requested: str) -> str:
    if requested != "auto":
        return requested
    # deterministic spread so a single run covers all four branches
    return MODES[sum(ord(c) for c in cell) % len(MODES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto"] + MODES)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fail-rate", type=float, default=0.0,
                    help="Randomly raise, to prove one bad cell cannot end a run.")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = Path(a.manifest)
    man = pd.read_parquet(mpath if mpath.is_absolute() else ROOT / mpath)
    todo = select_work(man, out, a.shard, a.num_shards, a.limit)

    print(f"manifest {len(man)} cells | shard {a.shard}/{a.num_shards} "
          f"| {len(todo)} to generate", flush=True)
    if todo.empty:
        print("nothing to do")
        return 0

    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))
    modes_used, log = {}, RunLog(out / f"log_shard{a.shard}.csv")

    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        mode = pick_mode(r.cell, a.mode)
        try:
            if a.fail_rate and random.Random(r.seed).random() < a.fail_rate:
                raise RuntimeError("synthetic failure")
            ref = Image.open(ROOT / r.id_path)
            img = compose(ref, a.width, a.height, int(r.seed), mode)
            tmp = out / (cell_to_file(r.cell) + ".part")
            img.save(tmp, format="PNG")
            tmp.rename(out / cell_to_file(r.cell))
            status = "ok"
            modes_used[mode] = modes_used.get(mode, 0) + 1
        except Exception as e:                 # noqa: BLE001
            status = f"error:{type(e).__name__}:{str(e)[:60]}"
            (out / (cell_to_file(r.cell) + ".part")).unlink(missing_ok=True)
            print(f"  [{r.cell}] {status}", flush=True)
        log.add(r.cell, status, time.time() - t)

    log.flush()
    (out / "modes.json").write_text(json.dumps(modes_used, indent=1))
    print(f"\n{log.summary()}")
    print("modes:", modes_used)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Generate the Part 2 calibration set: stylised reference photographs.

Plain FLUX.1-dev img2img -- no InfuseNet, no PuLID, no identity conditioning at
all. That absence is the point. Each output is the reference photograph itself
pushed through a stylising img2img pass, so the identity is held exactly
constant and any ID Loss that appears is ArcFace failing to read a stylised
face rather than a generator failing to preserve a face. See
`tools/build_calib_manifest.py` for why Part 2 cannot be interpreted without it.

Same contract as the other runners: one manifest in, one PNG per cell out,
sharded, resumable, one bad cell never ends the run.

Runs in the `infu` env, which already has diffusers 0.31.0 and the FLUX
snapshot; no new environment is needed.

Usage:
  python src/gen_calib.py --manifest benchmark/manifest_calib.parquet \\
      --out $WORK/prl26/results/images/calib --cpu-offload
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import (RunLog, cell_to_file, resolve_model_dir,  # noqa: E402
                               select_work)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="Model CPU offload; ample on a 40 GB A100 and avoids "
                         "a second quantisation path in the project.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def prepare_source(path: Path, width: int, height: int) -> Image.Image:
    """Centre-crop the reference to the target aspect, then resize.

    A plain resize to 864x1152 would stretch a square portrait by a third.
    ArcFace aligns by landmarks, so a geometric distortion of that size changes
    the embedding -- and it would change it in the CALIBRATION arm only, which
    is precisely the arm whose job is to isolate a single effect. Cropping keeps
    the face's proportions and costs only some background.
    """
    img = Image.open(path).convert("RGB")
    target = width / height
    w, h = img.size
    if w / h > target:                      # too wide: trim the sides
        new_w = int(round(h * target))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:                                   # too tall: trim top and bottom
        new_h = int(round(w / target))
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img.resize((width, height), Image.LANCZOS)


def main():
    a = build_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mpath = Path(a.manifest)
    man = pd.read_parquet(mpath if mpath.is_absolute() else ROOT / mpath)
    todo = select_work(man, out, a.shard, a.num_shards, a.limit)

    print(f"manifest {len(man)} cells | shard {a.shard}/{a.num_shards} "
          f"| {len(todo)} to generate", flush=True)
    if a.dry_run:
        print(todo.head(5)[["cell", "strength", "seed"]].to_string(index=False))
        return 0
    if todo.empty:
        print("nothing to do")
        return 0

    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))

    from diffusers import FluxImg2ImgPipeline

    # Compute nodes are offline and diffusers cannot resolve a repo id under
    # HF_HUB_OFFLINE even when the weights are cached; this returns the concrete
    # snapshot directory instead. Same fix as gen_infu.py.
    base_path = resolve_model_dir(a.base_model)
    print(f"base model: {base_path}", flush=True)

    t0 = time.time()
    pipe = FluxImg2ImgPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    if a.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    print(f"pipeline ready in {time.time()-t0:.0f} s", flush=True)

    log = RunLog(out / f"log_shard{a.shard}.csv")
    failures = 0
    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        try:
            src = prepare_source(ROOT / r.id_path, a.width, a.height)
            # Seeded per cell from the manifest, so the calibration set is as
            # reproducible as everything else in the project.
            gen = torch.Generator(device="cpu").manual_seed(int(r.seed))
            img = pipe(
                prompt=r.prompt,
                image=src,
                strength=float(r.strength),
                width=a.width, height=a.height,
                num_inference_steps=a.num_steps,
                guidance_scale=a.guidance_scale,
                generator=gen,
            ).images[0]
            tmp = out / (cell_to_file(r.cell) + ".part")
            img.save(tmp, format="PNG")          # write-then-rename, so a killed
            tmp.rename(out / cell_to_file(r.cell))   # job leaves no truncated PNG
            status = "ok"
        except Exception as e:                 # noqa: BLE001
            status = f"error:{type(e).__name__}:{str(e)[:80]}"
            (out / (cell_to_file(r.cell) + ".part")).unlink(missing_ok=True)
            print(f"  [{r.cell}] {status}", flush=True)
            # Full traceback for the first few: an 80-character status is enough
            # to spot a pattern but useless for diagnosis, and re-running a GPU
            # job just to learn which line raised costs a queue wait.
            if failures < 3:
                traceback.print_exc()
                sys.stdout.flush()
            failures += 1
        log.add(r.cell, status, time.time() - t)
        if n % 25 == 0:
            print("  " + log.progress(n, len(todo)), flush=True)

    log.flush()
    print("\n" + log.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

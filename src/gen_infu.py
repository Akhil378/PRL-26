#!/usr/bin/env python
"""Batch generation with InfiniteYou (InfU).

The upstream `test.py` constructs the whole InfUFluxPipeline, produces one
image and exits, and exposes no width/height. Calling it 1,500 times would
spend most of the compute budget loading FLUX from disk. This builds the
pipeline once and iterates a frozen manifest, skipping work that already exists.

Usage:
  python src/gen_infu.py --manifest benchmark/manifest_repro.parquet \\
      --out $WORK/prl26/results/images/infu_aes2 \\
      --infu-repo $WORK/prl26/InfiniteYou --quantize-8bit \\
      --shard $SLURM_ARRAY_TASK_ID --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import RunLog, cell_to_file, resolve_model_dir, select_work  # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--infu-repo", required=True,
                    help="Checkout of github.com/bytedance/InfiniteYou (for `pipelines`).")
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--model-dir", default="ByteDance/InfiniteYou")
    ap.add_argument("--model-version", default="aes_stage2",
                    choices=["aes_stage2", "sim_stage1"])
    ap.add_argument("--infu-flux-version", default="v1.0")
    ap.add_argument("--insightface-root", default=None)
    # memory profile: fixed once in Phase 1 and never mixed across runs
    ap.add_argument("--quantize-8bit", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true")
    # generation settings, at the paper's defaults
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--cond-scale", type=float, default=1.0,
                    help="infusenet_conditioning_scale; swept in Part 2.")
    ap.add_argument("--guidance-start", type=float, default=0.0)
    ap.add_argument("--guidance-end", type=float, default=1.0)
    # sharding
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the work plan and exit without loading models.")
    return ap.parse_args()


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
        print(todo.head(5)[["cell", "seed"]].to_string(index=False))
        return 0
    if todo.empty:
        print("nothing to do")
        return 0

    # settings are written beside the images so a run can always be traced back
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))

    sys.path.insert(0, str(Path(a.infu_repo)))
    from pipelines.pipeline_infu_flux import InfUFluxPipeline

    infu_path = resolve_model_dir(
        a.model_dir, f"infu_flux_{a.infu_flux_version}/{a.model_version}")
    # Resolve the BASE model to a local path too. Compute nodes are offline, and
    # InfUFluxPipeline hands base_model_path straight to diffusers, which tries
    # the hub API for a repo id and dies under HF_HUB_OFFLINE even though the
    # weights are cached. snapshot_download() reads the cache offline, so this
    # turns a repo id into the concrete snapshot directory.
    base_path = resolve_model_dir(a.base_model)
    print(f"base model:  {base_path}", flush=True)
    print(f"infu model:  {infu_path}", flush=True)
    t0 = time.time()
    pipe = InfUFluxPipeline(
        base_model_path=base_path,
        infu_model_path=infu_path,
        insightface_root_path=a.insightface_root or ".",
        infu_flux_version=a.infu_flux_version,
        model_version=a.model_version,
        quantize_8bit=a.quantize_8bit,
        cpu_offload=a.cpu_offload,
    )
    print(f"pipeline ready in {time.time()-t0:.0f} s", flush=True)

    log = RunLog(out / f"log_shard{a.shard}.csv")
    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        try:
            img = pipe(
                id_image=Image.open(ROOT / r.id_path).convert("RGB"),
                prompt=r.prompt,
                control_image=None,
                width=a.width, height=a.height,
                seed=int(r.seed),
                guidance_scale=a.guidance_scale,
                num_steps=a.num_steps,
                infusenet_conditioning_scale=a.cond_scale,
                infusenet_guidance_start=a.guidance_start,
                infusenet_guidance_end=a.guidance_end,
                cpu_offload=a.cpu_offload,
            )
            tmp = out / (cell_to_file(r.cell) + ".part")
            img.save(tmp, format="PNG")                      # write-then-rename so a killed job
            tmp.rename(out / cell_to_file(r.cell))   # never leaves a truncated PNG
            status = "ok"
        except Exception as e:                 # noqa: BLE001 - one bad cell must not end the run
            status = f"error:{type(e).__name__}:{str(e)[:80]}"
            (out / (cell_to_file(r.cell) + ".part")).unlink(missing_ok=True)
            print(f"  [{r.cell}] {status}", flush=True)
        log.add(r.cell, status, time.time() - t)
        if n % 25 == 0:
            print("  " + log.progress(n, len(todo)), flush=True)

    log.flush()
    print("\n" + log.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

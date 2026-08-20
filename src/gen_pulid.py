#!/usr/bin/env python
"""Batch generation with PuLID-FLUX, the primary baseline.

Same contract as gen_infu.py: one manifest in, one PNG per cell out, resumable.
PuLID runs on the Black Forest Labs `flux` codebase rather than Diffusers, so it
needs its own conda environment -- do not try to merge the two.

On baseline fairness: PuLID's own documentation recommends different settings
for photorealistic and stylised scenes (`timestep_to_start_cfg` around 4 for
photoreal, 0-1 for stylised). Running Part 2 at the photoreal defaults would
hand InfU an unearned advantage on precisely the question being asked, so
--id-start is exposed and the value used is recorded in run_config.json.

Usage:
  python src/gen_pulid.py --manifest benchmark/manifest_repro.parquet \\
      --out $WORK/prl26/results/images/pulid --pulid-repo $WORK/prl26/PuLID \\
      --offload --shard $SLURM_ARRAY_TASK_ID --num-shards 4
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
from src.runner_common import RunLog, cell_to_file, select_work  # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pulid-repo", required=True,
                    help="Checkout of github.com/ToTheBeginning/PuLID.")
    ap.add_argument("--name", default="flux-dev")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--fp8", action="store_true",
                    help="Degrades facial detail; prefer bf16+offload on A100.")
    ap.add_argument("--onnx-provider", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--id-weight", type=float, default=1.0,
                    help="Identity strength; the Part 2 sweep knob.")
    ap.add_argument("--id-start", type=int, default=4,
                    help="timestep_to_start_cfg; PuLID suggests 0-1 for stylised prompts.")
    ap.add_argument("--true-cfg", type=float, default=1.0)
    ap.add_argument("--max-sequence-length", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
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

    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))

    sys.path.insert(0, str(Path(a.pulid_repo)))
    from flux.util import SamplingOptions  # noqa: F401  (kept for parity with upstream)
    from app_flux import FluxGenerator

    print("loading PuLID-FLUX...", flush=True)
    t0 = time.time()
    gen = FluxGenerator(a.name, "cuda", a.offload, aggressive_offload=False,
                        args=argparse.Namespace(fp8=a.fp8,
                                                onnx_provider=a.onnx_provider))
    print(f"pipeline ready in {time.time()-t0:.0f} s", flush=True)

    log = RunLog(out / f"log_shard{a.shard}.csv")
    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        try:
            result = gen.generate_image(
                width=a.width, height=a.height, num_steps=a.num_steps,
                start_step=a.id_start, guidance=a.guidance, seed=int(r.seed),
                prompt=r.prompt,
                id_image=Image.open(ROOT / r.id_path).convert("RGB"),
                id_weight=a.id_weight,
                neg_prompt="", true_cfg=a.true_cfg, timestep_to_start_cfg=a.id_start,
                max_sequence_length=a.max_sequence_length,
            )
            img = result[0] if isinstance(result, (tuple, list)) else result
            tmp = out / (cell_to_file(r.cell) + ".part")
            img.save(tmp, format="PNG")
            tmp.rename(out / cell_to_file(r.cell))
            status = "ok"
        except Exception as e:                 # noqa: BLE001
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

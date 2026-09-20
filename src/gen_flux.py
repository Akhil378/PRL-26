#!/usr/bin/env python
"""Text-only FLUX.1-dev generation: the CLIPScore ceiling.

Why this run exists
-------------------
The paper's central architectural claim is about text alignment, and it is
stated relative to a ceiling: InfU reaches CLIPScore 0.318 against PuLID's
0.286, "reducing the gap to the upper-bound performance of FLUX.1-dev on our
test set (0.334) by 66.7%". The argument is that injecting identity through
residual connections costs less text alignment than modifying attention does.

That claim cannot be evaluated without the ceiling measured on the same
benchmark. This run generates each prompt with no identity conditioning at all,
on the same base model, resolution, step count, guidance and precision as Part 1,
so the resulting CLIPScore is the upper bound for this prompt set specifically.

One image per prompt, not per cell: with no identity there is nothing for a cell
to vary. The prompt set is therefore identical to Part 1's but the weighting
differs, because Part 1 pairs each prompt with 7 or 8 identities. Both the plain
and the manifest-weighted means are reported by the analysis so the comparison
can be made on either basis.

Usage:
  python src/gen_flux.py --prompts benchmark/prompts_200.json \\
      --out $WORK/prl26/results/images/ceiling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.manifest import derive_seed                                  # noqa: E402
from src.runner_common import RunLog, cell_to_file, resolve_model_dir, select_work  # noqa: E402


def build_work(prompts_path: Path) -> pd.DataFrame:
    """One row per prompt, in the same shape the shard/resume helpers expect.

    The seed derivation matches src/manifest.py so that this run is reproducible
    from the prompt file alone, exactly as the cell manifests are.
    """
    prompts = json.load(open(prompts_path, encoding="utf-8"))
    rows = []
    for p in prompts:
        cell = f'textonly|{p["pid"]}'
        rows.append({
            "cell": cell,
            "pid": p["pid"],
            "prompt": p["text"],
            "seed": derive_seed(cell),
            "gender": p["gender"],
            "face_size": p["face_size"],
            "complexity": p["complexity"],
            "length": p["length"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="benchmark/prompts_200.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ppath = Path(a.prompts)
    work = build_work(ppath if ppath.is_absolute() else ROOT / ppath)
    todo = select_work(work, out, a.shard, a.num_shards, a.limit)

    print(f"{len(work)} prompts | shard {a.shard}/{a.num_shards} "
          f"| {len(todo)} to generate", flush=True)
    if a.dry_run:
        print(todo.head(5)[["cell", "seed"]].to_string(index=False))
        return 0
    if todo.empty:
        print("nothing to do")
        return 0

    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))

    import torch
    from diffusers import FluxPipeline

    base_path = resolve_model_dir(a.base_model)
    print(f"base model: {base_path}", flush=True)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    # Same precision and memory strategy as the Part 1 arms, so the ceiling is
    # not measured under a different numerical regime than the thing it bounds.
    pipe.enable_model_cpu_offload()
    print(f"pipeline ready in {time.time()-t0:.0f} s", flush=True)

    log = RunLog(out / f"log_shard{a.shard}.csv")
    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        try:
            gen = torch.Generator("cpu").manual_seed(int(r.seed))
            img = pipe(
                prompt=r.prompt,
                width=a.width, height=a.height,
                num_inference_steps=a.num_steps,
                guidance_scale=a.guidance_scale,
                generator=gen,
            ).images[0]
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

#!/usr/bin/env python
"""Batch generation with PuLID-FLUX, the primary baseline.

Same contract as gen_infu.py: one manifest in, one PNG per cell out, resumable.
PuLID runs on the Black Forest Labs `flux` codebase rather than Diffusers, so it
needs its own conda environment -- do not try to merge the two.

On baseline fairness: PuLID's own documentation recommends different settings
for photorealistic and stylised scenes. The knob is `start_step`, the gradio
"timestep to start inserting ID" slider -- docs/pulid_for_flux.md: "For
generating realistic images, we suggest setting this to 4 ... For generating
stylized images, we suggest setting it to 0-1." Running Part 2 at the photoreal
value would hand InfU an unearned advantage on precisely the question being
asked, so --id-start is exposed and the value used is recorded in run_config.json.

`timestep_to_start_cfg` is a DIFFERENT parameter (when true CFG kicks in) and is
inert while --true-cfg is 1.0, which is the fake-CFG setting upstream recommends
for photorealistic scenes. It has its own flag, --cfg-start; an earlier version
of this file drove both from --id-start, which silently mislabelled the
experiment's main knob.

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
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    # MEASURED: the pulid env has onnxruntime and onnxruntime-gpu both at 1.29.0,
    # and the CPU-only wheel wins -- get_available_providers() returns only
    # Azure and CPU. Requesting CUDA does NOT raise; onnxruntime warns and
    # falls back, so "gpu" here would record a provider we are not using.
    # Default to the truth. InfU also runs insightface on CPU, so this keeps the
    # two methods consistent rather than introducing an asymmetry.
    ap.add_argument("--onnx-provider", default="cpu", choices=["gpu", "cpu"])
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--id-weight", type=float, default=1.0,
                    help="Identity strength; the Part 2 sweep knob.")
    ap.add_argument("--id-start", type=int, default=4,
                    help="start_step: timestep at which ID insertion begins. "
                         "PuLID suggests 4 for photoreal, 0-1 for stylised.")
    ap.add_argument("--cfg-start", type=int, default=1,
                    help="timestep_to_start_cfg; inert unless --true-cfg > 1.")
    ap.add_argument("--pulid-version", default="v0.9.1",
                    choices=["v0.9.0", "v0.9.1"],
                    help="PuLID release; v0.9.1 is upstream's default and is "
                         "~5 points better on facial similarity.")
    ap.add_argument("--pulid-weights", default=None,
                    help="Explicit .safetensors path; defaults to the staged "
                         "models/pulid_flux_<version>.safetensors.")
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

    # PuLID hardcodes its weight paths relative to the CWD -- flux/util.py wants
    # models/flux1-dev.safetensors, pipeline_flux.py wants models/antelopev2 and
    # models/pulid_flux_<version>.safetensors. On an offline compute node a
    # missing file surfaces as a confusing hub error minutes into the job, so
    # check here and fail immediately with something actionable.
    weights = a.pulid_weights or f"models/pulid_flux_{a.pulid_version}.safetensors"
    required = ["models/flux1-dev.safetensors", "models/ae.safetensors",
                "models/antelopev2/glintr100.onnx", weights]
    absent = [p for p in required if not Path(p).exists()]
    if absent:
        print(f"cwd is {Path.cwd()}", flush=True)
        for p in absent:
            print(f"  MISSING {p}", flush=True)
        print("run: conda run -p $WORK/envs/pulid --no-capture-output "
              "python tools/stage_pulid.py", flush=True)
        return 2

    sys.path.insert(0, str(Path(a.pulid_repo)))
    from flux.util import SamplingOptions  # noqa: F401  (kept for parity with upstream)
    from app_flux import FluxGenerator

    print("loading PuLID-FLUX...", flush=True)
    t0 = time.time()
    # FluxGenerator reads pretrained_model and version off this Namespace as well
    # as fp8/onnx_provider; omitting either raises AttributeError after the model
    # has already loaded, which wastes the whole pipeline-load time.
    gen = FluxGenerator(a.name, "cuda", a.offload, aggressive_offload=False,
                        args=argparse.Namespace(fp8=a.fp8,
                                                onnx_provider=a.onnx_provider,
                                                pretrained_model=a.pulid_weights,
                                                version=a.pulid_version))
    print(f"pipeline ready in {time.time()-t0:.0f} s", flush=True)

    # MEASURED FIX -- do not remove without re-reading this.
    #
    # app_flux.py:154 wraps the VAE decode in torch.autocast(cuda, bfloat16),
    # which casts the decoder's activations to bf16. torch 2.0.1+cu117 has no
    # CUDA kernel for nearest-neighbour upsample in bf16, so the decoder's
    # first Upsample raises
    #     "upsample_nearest2d_out_frame" not implemented for 'BFloat16'
    # at autoencoder.py:104 -- AFTER all 30 denoising steps have completed.
    # Every failed cell was a finished image thrown away on the last step,
    # which is why the failures cost a full 41.8 s each (job 1789499).
    #
    # It is the GPU kernel that is missing, not a stray CPU tensor: bf16
    # nearest upsample on the CPU works fine in this build, verified directly.
    # So "move the module to cuda" is exactly the wrong fix.
    #
    # The AE's own parameters are fp32 (verified: load_ae builds under
    # torch.device(cpu) at default dtype and load_state_dict copies rather
    # than assigns, so the bf16 checkpoint never changes the param dtype).
    # bf16 here was only ever an autocast artefact, so decoding in fp32 matches
    # the weights and is if anything more accurate. The cost is one decode per
    # image, not one per timestep, so it is immaterial next to 30 steps.
    _decode = gen.ae.decode

    def decode_fp32(z):
        with torch.autocast(device_type="cuda", enabled=False):
            return _decode(z.float())

    gen.ae.decode = decode_fp32
    print("VAE decode pinned to fp32 (bf16 nearest-upsample has no CUDA kernel "
          "in torch 2.0.1)", flush=True)

    log = RunLog(out / f"log_shard{a.shard}.csv")
    failures = 0
    for n, r in enumerate(todo.itertuples(), 1):
        t = time.time()
        try:
            # get_id_embedding documents "numpy rgb image, range [0, 255]" and
            # does its own cv2 RGB->BGR conversion, so hand it an RGB array.
            # A PIL image reaches resize_numpy_image_long, which calls
            # image.shape and dies; feeding BGR instead would be worse still --
            # it would silently corrupt every identity embedding.
            id_rgb = np.asarray(Image.open(ROOT / r.id_path).convert("RGB"))
            result = gen.generate_image(
                width=a.width, height=a.height, num_steps=a.num_steps,
                start_step=a.id_start, guidance=a.guidance, seed=int(r.seed),
                prompt=r.prompt,
                id_image=id_rgb,
                id_weight=a.id_weight,
                neg_prompt="", true_cfg=a.true_cfg, timestep_to_start_cfg=a.cfg_start,
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
            # Print the full traceback for the first few failures. The 80-char
            # status is enough to spot a pattern across 1500 cells but useless
            # for diagnosis, and re-running a GPU job just to learn which line
            # raised costs a queue wait. Capped so a systematically failing
            # shard cannot bury the log.
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

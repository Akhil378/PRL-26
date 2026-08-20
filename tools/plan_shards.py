#!/usr/bin/env python
"""Work out how to split a run across SLURM array tasks.

The naive choice -- one long job with --time=24:00:00 -- is the worst option on
a fully subscribed cluster. SLURM's backfill scheduler slots a job into a gap
only when the gap is at least as long as the requested walltime, so a 24 hour
request waits for a 24 hour gap while a 2 hour request drops into the far more
common short gaps. Asking for less time makes a job start sooner, and splitting
a run into more, shorter shards multiplies that effect.

The counterweight: model load costs a fixed few minutes per task, so tiny shards
waste GPU time reloading FLUX. This finds the balance.

Usage:
  python tools/plan_shards.py --manifest benchmark/manifest_repro.parquet \\
      --sec-per-image 30 --target-hours 2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sec-per-image", type=float, required=True,
                    help="MEASURED, from slurm/smoke.sbatch. Do not guess.")
    ap.add_argument("--target-hours", type=float, default=2.0,
                    help="Preferred shard length; shorter backfills sooner.")
    ap.add_argument("--load-minutes", type=float, default=5.0,
                    help="Pipeline construction cost paid once per array task.")
    ap.add_argument("--safety", type=float, default=1.4,
                    help="Walltime margin over the estimate.")
    ap.add_argument("--max-gpus", type=int, default=50,
                    help="QOS ceiling: normal allows gres/gpu=50 per user.")
    a = ap.parse_args()

    mp = Path(a.manifest)
    n = len(pd.read_parquet(mp if mp.is_absolute() else ROOT / mp))
    total_h = n * a.sec_per_image / 3600

    print(f"manifest        {n} cells")
    print(f"measured        {a.sec_per_image:.1f} s/image")
    print(f"pure GPU time   {total_h:.1f} h per method\n")

    print(f"  {'shards':>7}{'cells':>8}{'run':>9}{'+load':>8}{'--time':>10}"
          f"{'overhead':>10}")
    best = None
    for k in [1, 2, 4, 6, 8, 12, 16, 24, 32]:
        if k > a.max_gpus:
            continue
        cells = math.ceil(n / k)
        run_h = cells * a.sec_per_image / 3600
        wall_h = run_h + a.load_minutes / 60
        req = wall_h * a.safety
        if req > 24:
            note = "  exceeds 24 h cap"
        else:
            note = ""
        overhead = (a.load_minutes / 60 * k) / total_h
        hh, mm = int(req), int(round((req - int(req)) * 60))
        mark = ""
        if req <= 24 and best is None and wall_h <= a.target_hours:
            best, mark = k, "  <-- recommended"
        print(f"  {k:>7}{cells:>8}{run_h:>8.1f}h{wall_h:>7.1f}h"
              f"{f'{hh:02d}:{mm:02d}:00':>10}{overhead:>9.0%}{note}{mark}")

    if best is None:
        best = 8
        print("\n  no split reaches the target; falling back to 8")

    cells = math.ceil(n / best)
    req = (cells * a.sec_per_image / 3600 + a.load_minutes / 60) * a.safety
    hh, mm = int(req), int(round((req - int(req)) * 60))
    print(f"\nrecommended sbatch settings:")
    print(f"  #SBATCH --array=0-{best-1}")
    print(f"  #SBATCH --time={hh:02d}:{mm:02d}:00")
    print(f"  --num-shards {best}")
    print(f"\nwhy: {best} shards of ~{cells} cells, ~{req/a.safety:.1f} h each, "
          f"requested with a {a.safety:.1f}x margin.")
    print("Shorter requests backfill into gaps that a 24 h request cannot use.")
    print("If a shard is killed anyway, resubmitting the same array resumes it:")
    print("finished cells are skipped, so nothing is recomputed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the Part 2 sweep manifest: the slice the Pareto frontier is traced on.

The frontier is the submitted proposal's headline deliverable -- CLIPScore
(style adoption) plotted against ID Loss (identity retention) as the identity
strength is varied, so the two methods are compared as trade-off CURVES rather
than as single points. `configs/style.yaml` pre-registers the knob values:
InfU `infusenet_conditioning_scale` and PuLID `id_weight`, both at
[0.6, 0.8, 1.0], over a slice of 12 base prompts.

THE SLICE IS A STRICT SUBSET OF THE MAIN GRID, WHICH IS THE WHOLE POINT
----------------------------------------------------------------------
This filters `manifest_style.parquet` rather than regenerating cells. That
guarantees the sweep's cell keys, and therefore its seeds, are identical to the
main grid's. Two consequences, both load-bearing:

  1. The 1.0 point of every curve is ALREADY GENERATED. It is the main grid.
     Only 0.6 and 0.8 need new images, so a three-point frontier costs two
     points, not three.
  2. Each frontier point differs from its neighbours in the scale knob and
     nothing else -- same prompts, same identities, same latents. A curve built
     from independently sampled slices would confound the knob with sampling
     noise, and at these sample sizes that noise is larger than the effect.

Regenerating the slice with a fresh seed rule would silently destroy both.

SELECTION
---------
The 36 base prompts are exactly balanced: 2 genders x 3 lengths x 6. The slice
takes 2 from each of those 6 strata, so 12 prompts preserve the balance rather
than sampling it away. Within a stratum the pick is by a stable hash of the
prompt id, not alphabetical: ids run p001, p002, ... in authoring order, and
taking the lowest few would correlate the slice with whatever order the prompts
happened to be written in.

Each base prompt already pairs with 4 same-gender identities per style, so:
12 prompts x 4 identities x 4 styles = 192 cells per method per scale point.
With two new scale points and two methods that is 768 images, roughly 6.3 GPU-h
at the measured rates.

Usage:
  python tools/build_sweep_manifest.py
  python tools/build_sweep_manifest.py --base-prompts 8      # a narrower frontier
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def stable_rank(key: str) -> int:
    """Deterministic, order-independent rank for tie-breaking within a stratum."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-manifest", default="benchmark/manifest_style.parquet")
    ap.add_argument("--out", default="benchmark/manifest_sweep.parquet")
    ap.add_argument("--base-prompts", type=int, default=12,
                    help="Total base prompts in the slice; split evenly across "
                         "the gender x length strata.")
    a = ap.parse_args()

    style = pd.read_parquet(ROOT / a.style_manifest)
    base = style.drop_duplicates("base_pid")[
        ["base_pid", "gender", "length"]].reset_index(drop=True)

    strata = base.groupby(["gender", "length"], sort=True)
    n_strata = strata.ngroups
    if a.base_prompts % n_strata:
        raise SystemExit(
            f"--base-prompts {a.base_prompts} does not divide evenly across the "
            f"{n_strata} gender x length strata; use a multiple of {n_strata} so "
            "the slice stays balanced")
    per = a.base_prompts // n_strata

    picked = []
    for (g, ln), grp in strata:
        ordered = grp.assign(_r=grp["base_pid"].map(stable_rank)).sort_values("_r")
        take = ordered.head(per)
        if len(take) < per:
            raise SystemExit(f"stratum {g}/{ln} has only {len(grp)} prompts, "
                             f"need {per}")
        picked.extend(take["base_pid"].tolist())

    sweep = style[style["base_pid"].isin(picked)].reset_index(drop=True)

    # The subset property is what makes the 1.0 point free and the curve paired.
    # Assert it rather than trusting the filter.
    assert set(sweep["cell"]) <= set(style["cell"]), "sweep is not a subset"
    merged = sweep.merge(style[["cell", "seed"]], on="cell", suffixes=("", "_ref"))
    assert (merged["seed"] == merged["seed_ref"]).all(), \
        "seeds diverge from the main grid -- the 1.0 point would not be reusable"

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(out, index=False)

    bsel = base[base["base_pid"].isin(picked)]
    print(f"wrote {out}  ({len(sweep)} cells)")
    print(f"  base prompts {len(picked)} of {len(base)}")
    print("  balance (gender x length):")
    print(bsel.groupby(["gender", "length"]).size().to_string())
    print(f"  identities {sweep['iid'].nunique()}  "
          f"styles {sorted(sweep['style'].unique())}")
    print(f"  cells per style: "
          f"{sweep.groupby('style').size().unique().tolist()}")
    print()
    print("  The 1.0 point of each curve is the MAIN GRID -- do not regenerate it.")
    print("  Only 0.6 and 0.8 need new images:")
    print(f"    {len(sweep)} cells x 2 scale points x 2 methods = "
          f"{len(sweep) * 4} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())

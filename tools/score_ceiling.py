#!/usr/bin/env python
"""Score the text-only ceiling run: CLIPScore alone.

The ceiling images have no identity conditioning, so there is no reference face
and ID Loss is undefined by construction rather than by failure. src/score_all.py
expects a cell manifest with an `id_path` per row, which this run does not have,
so the CLIP half is done here instead of bending that script into a shape it was
not written for.

Two means are reported. The plain mean treats every prompt equally. The
manifest-weighted mean weights each prompt by how often it appears in the Part 1
manifest, which is 8 for female prompts and 7 for male ones under gender-matched
pairing; that is the weighting the Part 1 CLIPScore already carries, so it is the
one that makes the ceiling and the methods directly comparable.

Usage:
  python tools/score_ceiling.py --images $WORK/prl26/results/images/ceiling \\
      --out $WORK/prl26/results/scores/ceiling_prompts.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.clip_backend import CLIPBackend        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--prompts", default="benchmark/prompts_200.json")
    ap.add_argument("--manifest", default="benchmark/manifest_repro.parquet",
                    help="Only used to recover each prompt's Part 1 weight.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    img_dir = Path(a.images)
    prompts = json.load(open(ROOT / a.prompts, encoding="utf-8"))

    weights = {}
    mpath = ROOT / a.manifest
    if mpath.exists():
        weights = pd.read_parquet(mpath).pid.value_counts().to_dict()

    print("loading CLIP backbones...")
    clips = {b: CLIPBackend(b) for b in ("B32", "L14")}

    rows, missing = [], 0
    for p in prompts:
        f = img_dir / f'textonly__{p["pid"]}.png'
        if not f.exists():
            missing += 1
            continue
        img = Image.open(f).convert("RGB")
        rec = {"pid": p["pid"], "prompt": p["text"], "gender": p["gender"],
               "face_size": p["face_size"], "complexity": p["complexity"],
               "length": p["length"], "weight": weights.get(p["pid"], 1)}
        for b, c in clips.items():
            rec[f"clipscore_{b.lower()}"] = c.clipscore(img, p["text"])
        rows.append(rec)

    if not rows:
        print("no ceiling images found")
        return 1

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nwrote {out}  ({len(df)} prompts, {missing} missing)")
    for b in ("b32", "l14"):
        col = f"clipscore_{b}"
        plain = df[col].mean()
        wtd = (df[col] * df.weight).sum() / df.weight.sum()
        print(f"  CLIPScore {b.upper():<4} plain={plain:.4f}  "
              f"manifest-weighted={wtd:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

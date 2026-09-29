#!/usr/bin/env python
"""Every Part 2 image against every style phrase, so style is compared on one phrase.

The pre-registered "style adoption" delta is
    CLIP(styled image, its style phrase) - CLIP(photo image, "a natural colour photograph"),
two different phrases, so its sign mixes how well the style took with how well
two sentences happen to match two images (CONTEXT, claim audit of 24 Sep). The
fix needs the photo image scored against each style phrase too:
    style gain = CLIP(styled image, phrase) - CLIP(photo image, same phrase),
same method, same identity, same base prompt and seed, same phrase. With all four
phrases per image it also gives a zero-shot style label: the phrase an image
matches best, which a reader can take as "CLIP calls this a watercolour".

Both backbones are scored; ViT-L/14 is the one the style score and FMI use.

Usage:
  python tools/score_style_phrases.py --images $WORK/prl26/results/images
      --out $WORK/prl26/results/scores/style_phrases.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.clip_backend import CLIPBackend  # noqa: E402
from src.runner_common import cell_to_file  # noqa: E402

SETS = [("style_infu", "manifest_style"), ("style_pulid", "manifest_style"),
        ("sweep_infu_cs06", "manifest_sweep"), ("sweep_infu_cs08", "manifest_sweep"),
        ("sweep_pulid_iw06", "manifest_sweep"), ("sweep_pulid_iw08", "manifest_sweep")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    phrases = (pd.read_parquet(ROOT / "benchmark/manifest_style.parquet")
               .groupby("style")["style_phrase"].first().to_dict())
    clips = {b: CLIPBackend(b) for b in ("L14", "B32")}
    text = {b: torch.cat([c.encode_text(phrases[s]) for s in phrases]) for b, c in clips.items()}
    rows = []
    for method, manifest in SETS:
        m = pd.read_parquet(ROOT / "benchmark" / f"{manifest}.parquet")
        for r in m.itertuples():
            p = Path(a.images) / method / cell_to_file(r.cell)
            rec = {"cell": r.cell, "method": method, "iid": r.iid, "base_pid": r.base_pid,
                   "style": r.style, "image_found": p.exists()}
            if p.exists():
                img = Image.open(p).convert("RGB")
                for b, c in clips.items():
                    sims = (c.encode_image(img) @ text[b].T).squeeze(0).tolist()
                    rec.update({f"sim_{b.lower()}_{s}": v for s, v in zip(phrases, sims)})
            rows.append(rec)
        print(f"{method}: {len(m)} images", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the Part 2 calibration manifest: the ArcFace domain-shift floor.

WHAT THIS CONTROLS FOR
----------------------
Part 2 measures how far ID Loss rises when a prompt is stylised. That rise has
two possible causes and they are not separable without this control:

  1. the generator actually lost the identity under stylistic conditioning
     -- the effect Part 2 exists to measure;
  2. ArcFace simply reads a watercolour worse than it reads a photograph,
     regardless of whether the identity survived.

Cause 2 is a property of the recogniser, not of InfU or PuLID, and it would
inflate every stylised ID Loss in the experiment by an unknown amount. Reporting
a raw rise as evidence of identity loss without excluding it is the central
threat to Part 2's validity, and `configs/style.yaml` pre-registers this control
precisely so that cannot happen: "every Part 2 ID Loss is quoted relative to
this floor".

THE DESIGN
----------
Stylise the reference photographs *themselves* with FLUX img2img, then measure
ID Loss between the stylised reference and the original. Identity is held
perfectly constant -- it is literally the same face, the same photograph -- so
whatever ID Loss appears is entirely cause 2. That is the floor.

8 references x 4 styles x 3 strengths = 96 images.

The photoreal arm is a control inside the control, and is not redundant: img2img
at strength 0.5 alters an image even when the prompt asks for a photograph, so
the photoreal arm separates "damage done by round-tripping through the model"
from "damage done by the style". Without it, reconstruction drift would be
misread as a style effect.

Strengths are swept rather than fixed because the floor is not one number: a
light stylisation should barely move ArcFace and a heavy one should move it a
lot. The curve is what lets a Part 2 result be placed against a comparable
degree of stylisation instead of against an average.

Usage:
  python tools/build_calib_manifest.py
  python tools/build_calib_manifest.py --strengths 0.5 --out benchmark/manifest_calib_small.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifest import derive_seed  # noqa: E402

# Phrases are taken verbatim from the Part 2 style grid so the floor is measured
# against the same words the experiment conditions on. Diverging here would make
# the control describe a different stylisation from the one being corrected for.
STYLE_PHRASES = {
    "photoreal": "a natural colour photograph",
    "oil_painting": "a 19th-century oil painting",
    "render_3d": "a stylised 3D character render",
    "watercolour": "a loose watercolour painting",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default="benchmark/identities.json")
    ap.add_argument("--out", default="benchmark/manifest_calib.parquet")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="img2img strength; higher deviates further from source.")
    ap.add_argument("--styles", nargs="+", default=list(STYLE_PHRASES))
    a = ap.parse_args()

    ids = json.loads((ROOT / a.identities).read_text())
    recs = ids["identities"] if isinstance(ids, dict) and "identities" in ids else ids

    # The floor has to be measured on the identities Part 2 actually uses; a
    # floor from a different set of faces would not be the correction Part 2
    # needs.
    pool = [r for r in recs if r.get("part2") and r.get("status") == "ready"]
    if not pool:
        raise SystemExit("no identities flagged part2 with status=ready -- "
                         "run tools/validate_identities.py first")

    rows = []
    for r in pool:
        for style in a.styles:
            phrase = STYLE_PHRASES[style]
            for s in a.strengths:
                # Strength goes in the cell key, so a re-run at a different
                # sweep never silently overwrites an existing image, and the
                # seed differs per strength rather than being shared across a
                # dimension that is deliberately varied.
                cell = f"{r['iid']}|calib_{style}_s{int(round(s * 100)):03d}"
                rows.append({
                    "cell": cell,
                    "iid": r["iid"],
                    "pid": f"calib_{style}_s{int(round(s * 100)):03d}",
                    # A fixed, subject-neutral frame: the reference supplies the
                    # person, so the prompt must not introduce scene content of
                    # its own or it would be changing two things at once.
                    "prompt": f"a head-and-shoulders portrait, {phrase}",
                    "id_path": r["path"],
                    "seed": derive_seed(cell),
                    "seed_key": cell,
                    "strength": float(s),
                    "style": style,
                    "gender": r.get("gender", "unknown"),
                    # Constant metadata: score_all.py expects these columns and
                    # they carry no information here, since every calibration
                    # image is the same kind of crop of the same kind of photo.
                    "face_size": "closeup",
                    "complexity": "plain",
                    "length": "short",
                })

    man = pd.DataFrame(rows)
    assert man["cell"].duplicated().sum() == 0, "duplicate cells"

    # NOTE: style_phrase is deliberately ABSENT. score_all.py switches on that
    # column to run the Face Masking Index, and FMI is meaningless here -- these
    # images are stylised portraits with no generated scene, so there is no
    # background region for it to contrast the face against. Style and strength
    # are recovered in analysis by joining the scores back to this manifest on
    # `cell`, which needs no change to the scorer.
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    man.to_parquet(out, index=False)

    # NB: man["style"], never man.style -- the latter is pandas' Styler property,
    # not the column, and returns an object that fails with an unrelated message.
    print(f"wrote {out}  ({len(man)} cells)")
    print(f"  identities {man['iid'].nunique()}  styles {sorted(man['style'].unique())}")
    print(f"  strengths  {sorted(man['strength'].unique())}")
    print(f"  seeds unique: {man['seed'].nunique()} of {len(man)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

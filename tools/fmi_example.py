#!/usr/bin/env python
"""The Face Masking Index's crops and scores for one cell, for the talk's example.

For each method and condition of one (identity, base prompt) cell: the face box
the scorer detected, the face and background crops the index compared
(src/metrics/face_masking.py, same rule), and the scores stored in the Part 2
tables. Boxes are in the original image's pixels; the talk's panels are the
same images scaled, so they scale linearly.

Usage (metrics env, where the images are):
  python tools/fmi_example.py --scores $WORK/prl26/results/scores \
      --images $WORK/prl26/results/images --cell id10:p122 \
      --out $WORK/prl26/results/figures/fmi_example.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.face_masking import plan_crops  # noqa: E402
from src.metrics.id_loss import IDScorer          # noqa: E402

STYLES = ["photoreal", "oil_painting", "render_3d", "watercolour"]
METHODS = [("infu", "style_infu"), ("pulid", "style_pulid")]
COLS = ["fmi_photo", "photo_face", "photo_bg", "fmi_reason", "style_score", "id_loss_antelope"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--cell", default="id10:p122")
    ap.add_argument("--id-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    iid, bp = a.cell.split(":")
    scorer = IDScorer(pack="antelopev2", root=a.id_root)
    out = {"cell": a.cell, "methods": {}}
    for key, img_dir in METHODS:
        d = pd.read_parquet(Path(a.scores) / f"style_{key}_manifest_style.parquet")
        d = d[(d["iid"] == iid) & (d["base_pid"] == bp)].set_index("style")
        rows = {}
        for s in STYLES:
            path = Path(a.images) / img_dir / f"{iid}__{bp}_{s}.png"
            bgr = cv2.imread(str(path))
            h, w = bgr.shape[:2]
            _, box = scorer.embed_with_box(bgr)
            plan = plan_crops(box, (w, h)) if box else None
            rec = {c: (None if pd.isna(d.loc[s, c]) else
                       (d.loc[s, c] if isinstance(d.loc[s, c], str) else float(d.loc[s, c])))
                   for c in COLS if c in d.columns}
            rec.update(size=[w, h], face_box=list(box) if box else None,
                       face_crop=list(plan.face) if plan else None,
                       bg_crop=list(plan.background) if plan and plan.background else None)
            rows[s] = rec
        for s in STYLES[1:]:
            f, c = rows[s].get("fmi_photo"), rows["photoreal"].get("fmi_photo")
            rows[s]["fmi_delta"] = None if f is None or c is None else f - c
            rows[s]["style_delta"] = rows[s]["style_score"] - rows["photoreal"]["style_score"]
        out["methods"][key] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for key, rows in out["methods"].items():
        for s, r in rows.items():
            print(f"{key:6s} {s:13s} fmi {r.get('fmi_photo')!s:>22}  delta {r.get('fmi_delta')!s:>22}  "
                  f"face {r['face_crop']}  bg {r['bg_crop']}  {r.get('fmi_reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

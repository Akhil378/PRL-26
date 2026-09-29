#!/usr/bin/env python
"""Contact sheets of the Part 1 images in which no face was found, for inspection.

tools/recover_missed_faces.py measures which missed faces the detector finds on
a second look. What stays missing still needs a person to say what it shows: a
figure seen from behind, a face turned away or hidden, a crowd with no clear
subject, or no person at all. These sheets make that a few minutes' work. Each
tile is labelled with its cell and prompt face size, and, for recovered faces,
the pass that found it. Output goes outside the repository, since the images
show research participants' likenesses.

Usage:
  python tools/missed_contact_sheet.py --scores $WORK/prl26/results/scores
      --images $WORK/prl26/results/images --out $WORK/prl26/results/figures/missed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file  # noqa: E402

TW, TH, COLS, ROWS = 216, 288, 6, 4


def sheet(tiles):
    grid = np.full((ROWS * (TH + 22), COLS * TW, 3), 255, np.uint8)
    for k, (img, label) in enumerate(tiles):
        r, c = divmod(k, COLS)
        y, x = r * (TH + 22), c * TW
        grid[y:y + TH, x:x + TW] = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
        cv2.putText(grid, label, (x + 3, y + TH + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["infu_aes2", "pulid", "pulid_s0"])
    ap.add_argument("--which", default="none,loose", help="found_at values to include")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(a.which.split(","))
    for s in a.sets:
        r = pd.read_csv(Path(a.scores) / f"recovered_{s}_manifest_repro.csv")
        r = r[r.found_at.fillna("none").isin(keep)].sort_values(["face_size", "cell"])
        tiles = []
        for t in r.itertuples():
            img = cv2.imread(str(Path(a.images) / s / cell_to_file(t.cell)))
            if img is not None:
                tag = "" if t.found_at == "none" else f" [{t.found_at}]"
                tiles.append((img, f"{t.cell} {t.face_size}{tag}"))
        per = COLS * ROWS
        for k in range(0, len(tiles), per):
            cv2.imwrite(str(out / f"{s}_{k // per + 1:02d}.jpg"), sheet(tiles[k:k + per]),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{s}: {len(tiles)} images on {-(-len(tiles) // per)} sheets")
    return 0


if __name__ == "__main__":
    sys.exit(main())

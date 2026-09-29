#!/usr/bin/env python
"""Why a face goes missing: one cell per cause, InfU above PuLID (start step 4).

tools/recover_missed_faces.py sorts the cells where antelopev2 found no face at
det_size 640 into those it finds on a second look and those it never finds, and
the contact sheets (tools/missed_contact_sheet.py) show what the second kind
are. Three causes cover almost all of them, and this figure shows one cell of
each, chosen by rule rather than by eye:

  large face   PuLID cells found only at det_size 320 or 160, close-up prompts:
               the face fills the frame and is cut off, and SCRFD at 640 does
               not fire on it
  out of frame PuLID cells with no face on any pass where InfU's image of the
               same cell has one, waist-length prompts: the picture is framed
               on the hands and the object
  back view    cells with no face on any pass for both methods, full-body
               prompts: the person faces away

For each cause the prompt with the most such cells is taken, and within it the
lowest identity number. Runs on the cluster, where the images are.

Usage (infu env):
  python tools/make_missing_figure.py --scores $WORK/prl26/results/scores
      --images $WORK/prl26/results/images --out $WORK/prl26/results/figures/missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file  # noqa: E402


def pick(df):
    """Most frequent prompt, then the lowest identity within it."""
    top = df.groupby("pid").size().sort_values(ascending=False, kind="stable").index[0]
    return df[df.pid == top].sort_values("iid").iloc[0]["cell"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.scores)

    det = {m: pd.read_parquet(S / f"{m}_manifest_repro.parquet").set_index("cell")["face_detected"].astype(bool)
           for m in ("infu_aes2", "pulid")}
    rec = {m: pd.read_csv(S / f"recovered_{m}_manifest_repro.csv").set_index("cell")
           for m in ("infu_aes2", "pulid")}
    P, I = rec["pulid"], rec["infu_aes2"]

    large = P[(P.found_at == "small") & (P.face_size == "closeup")].reset_index()
    frame = P[(P.found_at == "none") & (P.face_size == "waist")].reset_index()
    frame = frame[frame.cell.map(det["infu_aes2"])]
    back = P[(P.found_at == "none") & (P.face_size == "full")].reset_index()
    back = back[back.cell.isin(I.index[I.found_at == "none"])]
    cols = [("Face larger than the frame", pick(large)), ("Face out of frame", pick(frame)),
            ("Seen from behind", pick(back))]

    def status(m, cell):
        if det[m][cell]:
            return "face found at 640"
        r = rec[m].loc[cell]
        if r.found_at in ("small", "full", "pad"):
            return f"missed at 640; found at {'320/160' if r.found_at == 'small' else r.found_at}"
        return "no face on any pass"

    fig, ax = plt.subplots(2, 3, figsize=(7.0, 6.4))
    for j, (title, cell) in enumerate(cols):
        for i, (m, lab) in enumerate((("infu_aes2", "InfU"), ("pulid", "PuLID, start step 4"))):
            im = Image.open(Path(a.images) / m / cell_to_file(cell)).convert("RGB").resize((432, 576))
            ax[i, j].imshow(im)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel(status(m, cell), fontsize=7.5)
            if j == 0:
                ax[i, j].set_ylabel(lab, fontsize=9)
        ax[0, j].set_title(f"{title}\n{cell}", fontsize=8.5)
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_missing.pdf", dpi=200)
    fig.savefig(out / "fig_missing.png", dpi=110)
    (out / "selection.txt").write_text("\n".join(f"{t}: {c}" for t, c in cols) + "\n")
    print("\n".join(f"{t}: {c}" for t, c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())

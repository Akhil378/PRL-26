#!/usr/bin/env python
"""Generate the report's figures as PDF.

The guideline requires figures to be vector or PDF, with every axis labelled and
its units given. Photographic panels cannot be vector, so they are embedded in a
PDF, which is what the guideline asks for.

Two figures:

  fig_floor.pdf   One reference photograph rendered at each filter level, with
                  the measured ID Loss beneath each panel. The domain-shift
                  argument is far more convincing seen than tabulated: the
                  person is visibly the same in every panel, which is the whole
                  claim, while the recogniser's similarity collapses.

  fig_strata.pdf  Detection rate and ID Loss by face size, which localises the
                  deviation from the published figures to one stratum.

Usage (metrics env):
  python tools/make_report_figures.py --scores $WORK/prl26/results/scores \\
      --floor-images $WORK/prl26/results/images/floor --out report/figures
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Order matters: this is the ladder, from unmodified to most abstracted.
LADDER = ["none", "edge_preserve", "stylize_25", "stylize_45", "stylize_60"]
PRETTY = {"none": "unmodified", "edge_preserve": "edge-preserving",
          "stylize_25": "stylised (0.25)", "stylize_45": "stylised (0.45)",
          "stylize_60": "stylised (0.60)"}


def fig_floor(floor_df, img_dir, out, iid=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    iid = iid or sorted(floor_df["iid"].unique())[0]
    means = floor_df.groupby("filter")["id_loss_antelope"].mean()

    fig, axes = plt.subplots(1, len(LADDER), figsize=(11, 2.9))
    for ax, name in zip(axes, LADDER):
        level = LADDER.index(name)
        cand = list(Path(img_dir).glob(f"{iid}__floor_{level}_{name}.png"))
        if not cand:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            continue
        # Crop to the face so the panels show what the recogniser sees rather
        # than the shirt and the backdrop.
        im = Image.open(cand[0]).convert("RGB")
        w, h = im.size
        # Forehead to chin. The panel has to show the whole face: the claim the
        # figure makes is that a reader recognises the same person throughout,
        # and a crop that truncates the jaw undercuts exactly that.
        im = im.crop((int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.85)))
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        v = means.get(name, float("nan"))
        ax.set_title(PRETTY[name], fontsize=9)
        ax.set_xlabel(f"ID Loss {v:.3f}", fontsize=9)
    fig.suptitle("The same photograph, the same person, the same geometry",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def fig_strata(scores, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["closeup", "waist", "full"]
    d = scores[scores["face_size"].isin(order)]
    det = [d[d["face_size"] == s]["face_detected"].astype(float).mean() for s in order]
    idl, err = [], []
    for s in order:
        v = pd.to_numeric(d[d["face_size"] == s]["id_loss_antelope"],
                          errors="coerce").dropna()
        idl.append(v.mean())
        # Standard error, so the bar shows the precision of the mean rather
        # than the spread of the cells.
        err.append(v.std(ddof=1) / np.sqrt(len(v)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a1.bar(order, det, color="0.4")
    a1.set_ylim(0.7, 1.0)
    a1.set_ylabel("Face detection rate")
    a1.set_xlabel("Face size (benchmark stratum)")

    a2.bar(order, idl, yerr=err, capsize=4, color="0.4")
    a2.set_ylabel("ID Loss (antelopev2)")
    a2.set_xlabel("Face size (benchmark stratum)")
    for ax in (a1, a2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--floor-images", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--identity", default=None)
    a = ap.parse_args()

    S = Path(a.scores)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    floor = pd.read_parquet(S / "id_floor.parquet")
    p = fig_floor(floor, a.floor_images, out / "fig_floor.pdf", a.identity)
    print(f"  wrote {p}")

    scores = pd.read_parquet(S / "infu_aes2_manifest_repro.parquet")
    p = fig_strata(scores, out / "fig_strata.pdf")
    print(f"  wrote {p}")

    print("\nBoth are PDF, as the guideline requires. Axes carry labels and the")
    print("stratum bars carry standard errors of the mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

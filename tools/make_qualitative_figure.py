#!/usr/bin/env python
"""Part 2, seen: one (identity, prompt) cell in all four conditions, both methods.

The report states that InfU keeps identity under stylisation largely by declining
to apply the style. That claim rests on a table of paired deltas; this figure
shows what the deltas look like.

Selection is by rule, not by eye, because a hand-picked example of a claimed
effect is not evidence of it. For every (identity, base prompt) cell, each
method's per-style deltas in style adoption and ID Loss are compared with that
method's median delta for the same style (the medians Table 3 reports), each
distance scaled by the metric's interquartile range so the two metrics weigh
equally. The cell with the smallest summed distance, among cells where a face was
detected in all eight images, is the one shown: the most typical cell, not the
most striking one. `--cell iid:base_pid` overrides the rule, and the figure then
says nothing about typicality.

Writes, into --out:
  fig_style_grid.pdf     the report figure (PDF, as the guideline requires)
  panel_*.jpg            one image per panel, for the talk's slide
  selection.txt          the chosen cell, its deltas and the runners-up

Usage (metrics env, on the cluster, where the images are):
  python tools/make_qualitative_figure.py
      --scores     $WORK/prl26/results/scores
      --images     $WORK/prl26/results/images
      --identities benchmark/identities
      --out        $WORK/prl26/results/figures/qual
Write --out outside the repository: the cluster checkout must stay clean or
`git pull --ff-only` stops working (CONTEXT 5.30).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STYLES = ["photoreal", "oil_painting", "render_3d", "watercolour"]
STYLED = STYLES[1:]
PRETTY = {"photoreal": "Photorealistic control", "oil_painting": "Oil painting",
          "render_3d": "3D render", "watercolour": "Watercolour"}
METHODS = [("infu", "InfU", "style_infu"), ("pulid", "PuLID", "style_pulid")]
METRICS = ["style_score", "id_loss_antelope"]


def load(scores: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for key, _, _ in METHODS:
        d = pd.read_parquet(scores / f"style_{key}_manifest_style.parquet")
        for m in METRICS:
            d[m] = pd.to_numeric(d[m], errors="coerce")
        out[key] = d
    return out


def deltas(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Styled minus photoreal, matched on (iid, base_pid): the Part 2 contrast."""
    rows = []
    for key, d in frames.items():
        ctl = d[d["style"] == "photoreal"].set_index(["iid", "base_pid"])
        for s in STYLED:
            st = d[d["style"] == s].set_index(["iid", "base_pid"])
            for m in METRICS:
                for (iid, bp), v in (st[m] - ctl[m]).items():
                    rows.append((key, s, m, iid, bp, v))
    return pd.DataFrame(rows, columns=["method", "style", "metric", "iid",
                                       "base_pid", "delta"])


def select(frames, D):
    med = D.groupby(["method", "style", "metric"])["delta"].median()
    q = D.groupby("metric")["delta"]
    iqr = q.quantile(0.75) - q.quantile(0.25)
    D = D.assign(z=[abs(r.delta - med[(r.method, r.style, r.metric)]) / iqr[r.metric]
                    for r in D.itertuples()])
    both = pd.concat([f.assign(method=k) for k, f in frames.items()])
    all_det = both.groupby(["iid", "base_pid"])["face_detected"].apply(
        lambda x: bool(x.astype(bool).all()))
    s = D.groupby(["iid", "base_pid"])["z"].agg(["sum", "count"])
    s = s[s["count"] == len(METHODS) * len(STYLED) * len(METRICS)]
    s = s.join(all_det.rename("all_detected"))
    s = s[s["all_detected"]].sort_values("sum")
    if s.empty:
        raise SystemExit("no cell has a detected face in all eight images")
    return s, med


def open_panel(path: Path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return im.resize(size, Image.LANCZOS)


def square_reference(path: Path, side: int):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = min(w, h)
    im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
    return im.resize((side, side), Image.LANCZOS)


def build(frames, D, iid, bp, args, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ref_path = Path(args.identities) / f"{iid}.jpg"
    # Panels at 300x400 print at about 240 dpi in a 16 cm-wide figure, which is
    # sharp on paper and keeps the report PDF small: the backend embeds images
    # losslessly, so resolution is paid for in megabytes.
    PW, PH = 300, 400

    fig = plt.figure(figsize=(6.3, 3.95))
    gs = GridSpec(2, 5, figure=fig, wspace=0.06, hspace=0.34,
                  left=0.01, right=0.99, top=0.93, bottom=0.07)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(square_reference(ref_path, 400))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Reference", fontsize=8.5)
    ax.set_xlabel("FRLL, CC BY 4.0", fontsize=7)

    for row, (key, label, img_dir) in enumerate(METHODS):
        d = frames[key]
        cell = d[(d["iid"] == iid) & (d["base_pid"] == bp)].set_index("style")
        for col, s in enumerate(STYLES, start=1):
            a = fig.add_subplot(gs[row, col])
            p = Path(args.images) / img_dir / f"{iid}__{bp}_{s}.png"
            a.imshow(open_panel(p, (PW, PH)))
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
            if row == 0:
                a.set_title(PRETTY[s], fontsize=8.5)
            if col == 1:
                a.set_ylabel(label, fontsize=9.5, fontweight="bold")
            if s == "photoreal":
                a.set_xlabel(f"ID Loss {cell.loc[s, 'id_loss_antelope']:.3f}",
                             fontsize=7.5)
            else:
                q = D[(D["method"] == key) & (D["style"] == s) & (D["iid"] == iid)
                      & (D["base_pid"] == bp)].set_index("metric")["delta"]
                # D.style would be DataFrame.style, pandas' Styler, and match
                # nothing: columns are indexed by name throughout this block.
                a.set_xlabel(f"\u0394style {q['style_score']:+.3f}   "
                             f"\u0394ID {q['id_loss_antelope']:+.3f}",
                             fontsize=7.5)

    pdf = out / "fig_style_grid.pdf"
    fig.savefig(pdf, format="pdf", dpi=240)
    plt.close(fig)

    # The talk lays the panels out itself, in its own typefaces, so each image
    # goes out separately and slightly larger.
    square_reference(ref_path, 640).save(out / "panel_reference.jpg", quality=86)
    for key, _, img_dir in METHODS:
        for s in STYLES:
            p = Path(args.images) / img_dir / f"{iid}__{bp}_{s}.png"
            open_panel(p, (480, 640)).save(out / f"panel_{key}_{s}.jpg", quality=86)
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--identities", default="benchmark/identities")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", help="iid:base_pid, overriding the selection rule")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load(Path(a.scores))
    D = deltas(frames)
    ranking, med = select(frames, D)

    if a.cell:
        iid, bp = a.cell.split(":")
        how = "chosen by hand with --cell; NOT the typical cell"
    else:
        iid, bp = ranking.index[0]
        how = "selected by rule: closest to the median deltas, both methods jointly"

    pdf = build(frames, D, iid, bp, a, out)

    this = D[(D.iid == iid) & (D.base_pid == bp)].pivot_table(
        index=["method", "metric"], columns="style", values="delta")
    lines = [f"cell {iid} x {bp}  ({how})", "",
             "this cell's deltas (styled minus photoreal):", this.round(4).to_string(), "",
             "median deltas over all cells:", med.unstack("metric").round(4).to_string(), "",
             "closest cells (summed IQR-scaled distance):",
             ranking.head(5).to_string()]
    (out / "selection.txt").write_text("\n".join(lines) + "\n", encoding="ascii",
                                       errors="replace")
    print("\n".join(lines))
    print(f"\nwrote {pdf}  ({pdf.stat().st_size / 1e6:.2f} MB) and "
          f"{len(list(out.glob('panel_*.jpg')))} panels")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the identity-strength Pareto frontier: the proposal's headline figure.

Each method is traced as a CURVE rather than compared at a point. The knob
differs by method but means the same thing, namely how hard identity is pushed
into the generation: `infusenet_conditioning_scale` for InfU and `id_weight` for
PuLID. The two are NOT calibrated against each other, and the report must not
treat 0.6 on one as equivalent to 0.6 on the other. That is precisely why the
result is a frontier: each method is compared against its own trade-off, and the
curves are then compared as shapes.

THE 1.0 POINT IS THE MAIN GRID, RESTRICTED
------------------------------------------
`manifest_sweep.parquet` is a strict subset of `manifest_style.parquet`, so the
full-strength arm was already generated and is not regenerated here. It is read
off the grid and **restricted to the sweep's own cells**, because a curve whose
endpoint came from a different set of prompts and identities would confound the
knob with the sample.

Usage (metrics env):
  python tools/make_frontier.py --scores $WORK/prl26/results/scores \\
      --manifest benchmark/manifest_sweep.parquet --out report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "InfU": [(0.6, "sweep_infu_cs06_manifest_sweep.parquet"),
             (0.8, "sweep_infu_cs08_manifest_sweep.parquet"),
             (1.0, "style_infu_manifest_style.parquet")],
    "PuLID": [(0.6, "sweep_pulid_iw06_manifest_sweep.parquet"),
              (0.8, "sweep_pulid_iw08_manifest_sweep.parquet"),
              (1.0, "style_pulid_manifest_style.parquet")],
}

ID = "id_loss_antelope"          # PuLID's encoder
ID_INFU = "id_loss_irse50"       # InfU's encoder, from tools/score_infu_encoder.py
TXT = "clipscore_b32"


def collect(S, cells):
    rows = []
    for method, arms in ARMS.items():
        for scale, fname in arms:
            df = pd.read_parquet(S / fname)
            irf = S / ("irse50_" + fname.replace(".parquet", ".csv"))
            if irf.exists():
                df = df.merge(pd.read_csv(irf)[["cell", ID_INFU]], on="cell", how="left")
            else:
                df[ID_INFU] = np.nan
            # Restrict every arm to the sweep's cells. For 0.6 and 0.8 this is a
            # no-op; for the 1.0 arm, read off the full grid, it is the whole
            # point -- otherwise the endpoint would rest on a larger and
            # different sample than the rest of its own curve.
            df = df[df["cell"].isin(cells)]
            idv = pd.to_numeric(df[ID], errors="coerce").dropna()
            txv = pd.to_numeric(df[TXT], errors="coerce").dropna()
            iiv = pd.to_numeric(df[ID_INFU], errors="coerce").dropna()
            rows.append({
                "method": method, "scale": scale, "n": len(df),
                "n_id": len(idv),
                "detection": float(df["face_detected"].astype(bool).mean()),
                "id_loss": float(idv.mean()),
                "id_sem": float(idv.std(ddof=1) / np.sqrt(len(idv))),
                "id_sd": float(idv.std(ddof=1)),
                "id_infu": float(iiv.mean()) if len(iiv) else float("nan"),
                "id_infu_sem": float(iiv.std(ddof=1) / np.sqrt(len(iiv))) if len(iiv) > 1 else float("nan"),
                "clip": float(txv.mean()),
                "clip_sem": float(txv.std(ddof=1) / np.sqrt(len(txv))),
                "clip_sd": float(txv.std(ddof=1)),
            })
    return pd.DataFrame(rows)


def figure(d, out):
    """Two panels, one per judge of identity, same curves and same x axis.

    The ID axis is the one that moves: under PuLID's encoder PuLID's curve lies
    below InfU's at matched strength, under InfU's encoder above it. Drawing only
    one panel would present a recogniser choice as a property of the methods.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    styles = {"InfU": ("o-", "0.15"), "PuLID": ("s--", "0.45")}
    panels = [("id_loss", "id_sem", "ID Loss, PuLID's encoder (antelopev2)"),
              ("id_infu", "id_infu_sem", "ID Loss, InfU's encoder (IR-SE50)")]
    for ax, (col, sem, label) in zip(axes, panels):
        for method, g in d.groupby("method"):
            g = g.sort_values("scale")
            mk, c = styles[method]
            ax.errorbar(g["clip"], g[col], xerr=g["clip_sem"], yerr=g[sem],
                        fmt=mk, color=c, capsize=3, label=method, linewidth=1.4)
            for _, r in g.iterrows():
                ax.annotate(f"{r['scale']:.1f}", (r["clip"], r[col]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=c)
        ax.set_xlabel("CLIPScore, ViT-B/32 (higher is better)")
        ax.set_ylabel(label + ", lower is better", fontsize=9)
        ax.invert_yaxis()      # up and to the right is better on both axes
        ax.xaxis.set_major_locator(MaxNLocator(5))   # seven 4-decimal ticks collide
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def table(d, out):
    lines = ["\\begin{table}[t]", "  \\centering",
             "  \\caption[Identity-strength frontier]{Identity strength traded "
             "against text alignment, mean $\\pm$ standard deviation over the "
             "192-cell sweep slice. The knob is "
             "\\texttt{infusenet\\_conditioning\\_scale} for InfU and "
             "\\texttt{id\\_weight} for PuLID; the two are not calibrated "
             "against each other, so the curves are comparable as shapes and "
             "not pointwise. The 1.0 arm is the main grid restricted to the "
             "same cells rather than a separate run.}",
             "  \\label{tab:frontier}",
             "  \\begin{tabular}{llrrrr}", "    \\toprule",
             "    Method & Strength & $n$ & Detection & ID Loss & CLIPScore \\\\",
             "    \\midrule"]
    for method, g in d.groupby("method"):
        for _, r in g.sort_values("scale").iterrows():
            lines.append(
                f"    {method} & {r['scale']:.1f} & {int(r['n_id'])} & "
                f"{r['detection']:.3f} & "
                f"${r['id_loss']:.4f} \\pm {r['id_sd']:.4f}$ & "
                f"${r['clip']:.4f} \\pm {r['clip_sd']:.4f}$ \\\\")
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    Path(out).write_text("\n".join(lines), encoding="ascii")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--manifest", default="benchmark/manifest_sweep.parquet")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()

    S = Path(a.scores)
    mp = Path(a.manifest)
    cells = set(pd.read_parquet(mp if mp.is_absolute() else ROOT / mp)["cell"])

    d = collect(S, cells)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"  wrote {figure(d, out / 'figures' / 'fig_frontier.pdf')}")
    print(f"  wrote {table(d, out / 'tables' / 'frontier.tex')}")

    # State the shape comparison the figure is for, so it is not left to the eye.
    print("\n--- what the curves say ---")
    for method, g in d.groupby("method"):
        g = g.sort_values("scale")
        did = g["id_loss"].iloc[0] - g["id_loss"].iloc[-1]
        dtx = g["clip"].iloc[-1] - g["clip"].iloc[0]
        print(f"  {method}: raising strength 0.6 -> 1.0 buys {did:+.4f} ID Loss "
              f"for {dtx:+.4f} CLIPScore")
    return 0


if __name__ == "__main__":
    sys.exit(main())

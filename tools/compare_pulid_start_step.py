#!/usr/bin/env python
"""PuLID at start_step 0 against InfU and PuLID at start 4: every metric, paired.

Reads <scores>/{infu_aes2,pulid,pulid_s0}_manifest_repro.parquet and the matching
irse50_*.csv files (report, Section 5.1).

Usage:  python tools/compare_pulid_start_step.py results-backup/scores
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

S = Path(sys.argv[1])
ID = ["id_loss_antelope", "id_loss_buffalo", "id_loss_irse50"]
TX = ["clipscore_b32", "clipscore_l14", "pickscore_paper"]


def load(method):
    b = pd.read_parquet(S / f"{method}_manifest_repro.parquet")
    f = S / f"irse50_{method}_manifest_repro.csv"
    if f.exists():
        b = b.merge(pd.read_csv(f)[["cell", "id_loss_irse50"]], on="cell", how="left")
    for c in ID + TX:
        if c in b:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["face_detected"] = b["face_detected"].astype(bool)
    return b.set_index("cell")


I, P4, P0 = load("infu_aes2"), load("pulid"), load("pulid_s0")
done = P0.index[P0["score_status"].astype(str).eq("ok")] if "score_status" in P0 else P0.index
print(f"PuLID start 0 cells scored: {len(done)} of {len(P0)}")
cells = done
I, P4, P0 = I.loc[cells], P4.loc[cells], P0.loc[cells]

print("\nmeans over these cells (each method's own detected cells for ID):")
print(f"  {'metric':18} {'InfU':>8} {'PuLID4':>8} {'PuLID0':>8}")
for c in ID + TX:
    if c in P0:
        print(f"  {c:18} {I[c].mean():8.4f} {P4[c].mean():8.4f} {P0[c].mean():8.4f}")
print(f"  {'detection':18} {I.face_detected.mean():8.4f} {P4.face_detected.mean():8.4f} {P0.face_detected.mean():8.4f}")


def paired(a, b, c):
    j = pd.concat([a[c].rename("a"), b[c].rename("b")], axis=1)
    cc = j.dropna()
    d = cc.b - cc.a
    ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / len(d) ** 0.5)
    p = stats.ttest_rel(cc.b, cc.a).pvalue
    w = j.fillna(np.nanmax(j.values)) if c in ID else None
    worst = (w.b - w.a).mean() if w is not None else float("nan")
    return d.mean(), ci, p, len(d), worst


print("\nPuLID(start 0) minus InfU, paired:")
for c in ID + TX:
    if c in P0:
        m, ci, p, n, worst = paired(I, P0, c)
        extra = f"  worst-case {worst:+.4f}" if c in ID else ""
        print(f"  {c:18} {m:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p:.2g}  n={n}{extra}")

print("\nPuLID start 0 minus start 4, paired (the setting's direct effect):")
for c in ID + TX:
    if c in P0:
        m, ci, p, n, _ = paired(P4, P0, c)
        print(f"  {c:18} {m:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p:.2g}  n={n}")

ceil = {"clipscore_b32": 0.3105, "clipscore_l14": 0.2583}
print("\ntext-alignment headroom with PuLID at start 0:")
for c, C in ceil.items():
    i, p0 = I[c].mean(), P0[c].mean()
    print(f"  {c}: ceiling {C:.4f}  InfU {i:.4f}  PuLID0 {p0:.4f}  headroom {C - p0:+.4f}  "
          f"InfU closes {100 * (i - p0) / (C - p0):+.1f}%  published margin/headroom {0.032 / (C - p0):.2f}x")

#!/usr/bin/env python
"""Which identity judges are neutral, and what do the neutral ones say?

Each method conditions on one ArcFace network (PuLID on glintr100, InfU on
IR-SE50), and each wins under its own. A judge is neutral between them if its
verdict on a cell does not follow either method's encoder where the two
encoders disagree. That is measured directly:

    d_j   = ID Loss(PuLID) - ID Loss(InfU) under judge j, per cell
    lean  = corr(d_j, z(d_IR-SE50) - z(d_glintr100)),   z = standardised

The disagreement term is large where InfU's encoder is harder on PuLID than
PuLID's encoder is. A judge that sides with InfU's encoder correlates positively
with it, one that sides with PuLID's negatively, a neutral one not at all; the
lean is proportional to corr(d_j, d_IR-SE50) - corr(d_j, d_glintr100), and the
two encoders themselves mark the ends of the scale. The 95% interval is
Fisher's. buffalo_l, which tracks glintr100 at r = 0.985, is the check that the
measure detects a lean when there is one.

Judges: the three ArcFace networks, FaceNet on VGGFace2 and on CASIA-WebFace
(tools/dump_embeddings.py), SFace, VGG-Face and dlib (tools/score_panel.py).
Cells are those where the paper's pipeline (antelopev2 at 640) found a face in
both methods' images and the judge's own pipeline did too. Per judge the output
gives the lean, the paired difference (mean, 95% t interval, dz, share of cells
where PuLID is lower) and identification (the face is closest to its own
reference among the same-gender references), at both PuLID settings. The neutral
panel is every judge whose lean interval contains zero at both settings; their
per-cell differences, each standardised, are averaged into one panel verdict.

Usage:  python tools/analyse_panel.py <embeddings dir> <panel dir> [<scores dir>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EMB, PANEL = Path(sys.argv[1]), Path(sys.argv[2])
S = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "results-backup/scores"
DUMP = {"glint": "glintr100 (PuLID's)", "w600k": "buffalo_l", "irse50": "IR-SE50 (InfU's)",
        "fn_vgg": "FaceNet, VGGFace2", "fn_casia": "FaceNet, CASIA"}
PAN = {"sface": "SFace", "vggface": "VGG-Face", "dlib": "dlib ResNet"}
ARMS = {"InfU": "infu_aes2", "PuLID 4": "pulid", "PuLID 0": "pulid_s0"}
man = pd.read_parquet(ROOT / "benchmark/manifest_repro.parquet").set_index("cell")


def refs(judge):
    if judge in DUMP:
        R = np.load(EMB / "refs.npz", allow_pickle=True)
        return {str(k): v for k, v in zip(R["key"], R[judge])}
    R = np.load(PANEL / f"{judge}_refs.npz")
    return {str(k): v for k, v in zip(R["key"], R["emb"])}


def emb(judge, arm):
    if judge in DUMP:
        z = np.load(EMB / f"{arm}.npz", allow_pickle=True)
        return pd.DataFrame(z[judge].astype(np.float32), index=[str(k) for k in z["key"]])
    z = np.load(PANEL / f"{judge}_{arm}.npz")
    return pd.DataFrame(z["emb"].astype(np.float32), index=[str(k) for k in z["key"]])


def judge_frame(judge):
    """Per arm: ID Loss and whether the face is identified, NaN where the judge has no face."""
    R = refs(judge)
    frll = [k for k in R if "/identities/" in k]
    G = np.stack([R[k] for k in frll])
    gender = man.groupby("id_path")["gender"].first()
    out = {}
    for lab, arm in ARMS.items():
        E = emb(judge, arm).reindex(man.index)
        sims = E.to_numpy() @ G.T
        own = np.array([frll.index(p) for p in man["id_path"]])
        same = np.array([[gender[k] == g for k in frll] for g in man["id_path"].map(gender)])
        genuine = sims[np.arange(len(own)), own]
        imp = np.where(same, sims, -np.inf)
        imp[np.arange(len(own)), own] = -np.inf
        ok = np.isfinite(genuine)
        out[lab] = pd.DataFrame({"loss": np.where(ok, 1 - genuine, np.nan),
                                 "ident": np.where(ok, (genuine > imp.max(1)).astype(float), np.nan)},
                                index=man.index)
    return out


def main():
    scrfd = {lab: pd.read_parquet(S / f"{arm}_manifest_repro.parquet").set_index("cell")["face_detected"]
             .astype(bool).reindex(man.index) for lab, arm in ARMS.items()}
    J = {j: judge_frame(j) for j in list(DUMP) + [p for p in PAN if (PANEL / f"{p}_refs.npz").exists()]}
    names = {**DUMP, **PAN}
    res, OUT = {}, {}
    for setting in ("PuLID 4", "PuLID 0"):
        common = scrfd["InfU"] & scrfd[setting]
        d = pd.DataFrame({j: J[j][setting]["loss"] - J[j]["InfU"]["loss"] for j in J}).loc[common]
        # Standardise before differencing: otherwise the encoder whose verdicts
        # spread wider dominates the disagreement and even a copy of glintr100
        # would show a lean. With both in sd units, lean is proportional to
        # corr(d_j, d_IR-SE50) - corr(d_j, d_glintr100), and the two encoders
        # themselves sit at -/+ sqrt((1 - r)/2), the ends of the scale.
        z = (d - d.mean()) / d.std()
        disagree = z["irse50"] - z["glint"]
        print(f"\n=== {setting} minus InfU, cells where antelopev2 found both faces ({int(common.sum())})")
        print(f"  {'judge':20} {'n':>5} {'lean':>22} {'mean diff':>28} {'dz':>6} {'PuLID lower':>11}"
              f" {'identified InfU / PuLID':>24}")
        for j in J:
            m = d[j].notna() & disagree.notna()
            r = float(np.corrcoef(d[j][m], disagree[m])[0, 1])
            fz, se = np.arctanh(r), 1 / np.sqrt(m.sum() - 3)
            lo, hi = np.tanh(fz - 1.96 * se), np.tanh(fz + 1.96 * se)
            x = d[j][m]
            ci = stats.t.interval(0.95, len(x) - 1, x.mean(), x.std(ddof=1) / np.sqrt(len(x)))
            p = stats.ttest_1samp(x, 0).pvalue
            both = common & J[j]["InfU"]["ident"].notna() & J[j][setting]["ident"].notna()
            ia, ib = J[j]["InfU"]["ident"][both].mean(), J[j][setting]["ident"][both].mean()
            xa, xb = J[j]["InfU"]["ident"][both] > 0, J[j][setting]["ident"][both] > 0
            n01, n10 = int((~xa & xb).sum()), int((xa & ~xb).sum())
            p_id = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
            lean = f"{r:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(f"  {names[j]:20} {len(x):5d} {lean:>22} {x.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                  f" {x.mean() / x.std(ddof=1):+6.2f} {np.mean(x < 0):11.0%} {ia:11.3f} / {ib:.3f}"
                  f"  McNemar p={p_id:.1e}")
            res.setdefault(j, {})[setting] = {"lean": r, "lean_ci": [lo, hi], "n": int(len(x)),
                                              "mean": x.mean(), "ci": list(ci), "p": p,
                                              "dz": x.mean() / x.std(ddof=1), "pulid_lower": np.mean(x < 0),
                                              "ident_infu": ia, "ident_pulid": ib, "n_ident": int(both.sum()),
                                              "ident_p": p_id}
    neutral = [j for j in res if j not in ("glint", "irse50") and all(
        res[j][s]["lean_ci"][0] <= 0 <= res[j][s]["lean_ci"][1] for s in ("PuLID 4", "PuLID 0"))]
    print(f"\nneutral at both settings (lean interval contains 0): {[names[j] for j in neutral]}")
    for setting in ("PuLID 4", "PuLID 0"):
        common = scrfd["InfU"] & scrfd[setting]
        d = pd.DataFrame({j: J[j][setting]["loss"] - J[j]["InfU"]["loss"] for j in neutral}).loc[common].dropna()
        if d.empty:
            continue
        zd = d / d.std()                   # each judge in its own units of spread
        panel = zd.mean(1)
        ci = stats.t.interval(0.95, len(panel) - 1, panel.mean(), panel.std(ddof=1) / np.sqrt(len(panel)))
        print(f"  panel verdict, {setting} minus InfU, n={len(panel)}: {panel.mean():+.3f} sd units "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}], p={stats.ttest_1samp(panel, 0).pvalue:.1e}, "
              f"PuLID lower in {np.mean(panel < 0):.0%}")
        OUT[f"panel_{setting}"] = {"n": len(panel), "mean_sd_units": panel.mean(), "ci": list(ci),
                                   "p": stats.ttest_1samp(panel, 0).pvalue, "pulid_lower": np.mean(panel < 0)}
    OUT["judges"] = res
    OUT["neutral"] = neutral

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    (S / "panel.json").write_text(json.dumps(clean(OUT), indent=1), encoding="utf-8")
    print(f"\nwrote {S / 'panel.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""The last experiments, analysed: a non-ArcFace judge, recovered faces, style on one phrase.

Reads <scores> (results-backup/scores): the Part 1 and Part 2 score tables, the
irse50_*.csv files, and the outputs of slurm/score_extras.sbatch:
facenet_*.csv, recovered_*_manifest_repro.csv and style_phrases.parquet.
Prints every number the report quotes and writes them to <scores>/extras.json.

Conventions follow the rest of the project: Part 1 differences are PuLID minus
InfU, paired by cell, mean with a 95% t interval; worst-case bounds impute the
largest observed loss for every missing face; Part 2 deltas are styled minus
photoreal within a method, Wilcoxon with Holm across the three styles, the
median with a bootstrap interval (the Hodges-Lehmann location alongside)
(src/analysis/style_deltas.py).

Usage:  python tools/analyse_extras.py results-backup/scores
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.style_deltas import bootstrap_ci, hodges_lehmann, holm  # noqa: E402

S = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results-backup/scores")
JUDGES = {"id_loss_antelope": "glintr100 (PuLID's)", "id_loss_buffalo": "buffalo_l",
          "id_loss_irse50": "IR-SE50 (InfU's)", "id_loss_facenet": "FaceNet (neither)"}
STYLES = ["oil_painting", "render_3d", "watercolour"]
RECOVER = ("full", "small", "pad")          # the fallback rule; "loose" is reported apart
OUT: dict = {}


def tci(d) -> dict:
    d = np.asarray(d, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    lo, hi = stats.t.interval(0.95, n - 1, m, sd / np.sqrt(n))
    return {"mean": m, "lo": lo, "hi": hi, "p": stats.ttest_1samp(d, 0).pvalue, "n": n, "dz": m / sd}


def fmt(r: dict, k=4) -> str:
    return f"{r['mean']:+.{k}f} [{r['lo']:+.{k}f}, {r['hi']:+.{k}f}] n={r['n']} p={r['p']:.1e} dz={r['dz']:+.2f}"


def part1(method: str) -> pd.DataFrame:
    b = pd.read_parquet(S / f"{method}_manifest_repro.parquet")
    b = b[["cell", "iid", "pid", "face_size", "complexity", "face_detected",
           "id_loss_antelope", "id_loss_buffalo"]]
    b = b.merge(pd.read_csv(S / f"irse50_{method}_manifest_repro.csv")[["cell", "id_loss_irse50"]],
                on="cell", how="left")
    b = b.merge(pd.read_csv(S / f"facenet_{method}_manifest_repro.csv")[
        ["cell", "face_detected_facenet", "det_prob_facenet", "face_frac_facenet", "id_loss_facenet"]],
        on="cell", how="left")
    b["face_detected"] = b["face_detected"].astype(bool)
    b["face_detected_facenet"] = b["face_detected_facenet"].astype(bool)
    for c in JUDGES:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    rec = S / f"recovered_{method}_manifest_repro.csv"
    if rec.exists():
        r = pd.read_csv(rec)[["cell", "found_at", "face_h_px", "id_loss_antelope", "id_loss_irse50"]]
        r = r.rename(columns={"id_loss_antelope": "rec_antelope", "id_loss_irse50": "rec_irse50"})
        b = b.merge(r, on="cell", how="left")
    return b.set_index("cell")


def paired(A, B, col, det_a=None, det_b=None):
    """PuLID (B) minus InfU (A) over cells where both have `col`; plus the worst-case bound."""
    j = pd.concat([A[col].rename("a"), B[col].rename("b")], axis=1)
    if det_a is not None:
        j.loc[~det_a, "a"] = np.nan
        j.loc[~det_b, "b"] = np.nan
    cc = j.dropna()
    r = tci(cc.b - cc.a)
    r["mean_a"], r["mean_b"] = cc.a.mean(), cc.b.mean()
    r["win_b"] = float((cc.b < cc.a).mean())
    w = j.fillna(np.nanmax(j.values))
    r["worst"] = float((w.b - w.a).mean())
    return r


def section_judge(I, P4, P0):
    print("\n=== A. FaceNet, a judge from outside the ArcFace family")
    for lab, d in (("InfU", I), ("PuLID s4", P4), ("PuLID s0", P0)):
        print(f"  detection {lab:9} SCRFD@640 {d.face_detected.mean():.3f}   MTCNN {d.face_detected_facenet.mean():.3f}"
              f"   by size MTCNN {d.groupby('face_size').face_detected_facenet.mean().round(3).to_dict()}")
        OUT.setdefault("detection", {})[lab] = {"scrfd": d.face_detected.mean(),
                                                "mtcnn": d.face_detected_facenet.mean()}
        print("   SCRFD x MTCNN:", pd.crosstab(d.face_detected, d.face_detected_facenet).to_dict())

    # Independence: how closely does each judge track the others, per image?
    pool = pd.concat([I, P4, P0])
    pool = pool[list(JUDGES)].dropna()
    corr = pool.corr(method="pearson")
    print(f"\n  per-image Pearson r, {len(pool)} images of all three arms where every judge has a face:")
    print(corr.round(3).to_string())
    OUT["judge_corr"] = corr.round(4).to_dict()
    for lab, d in (("InfU", I), ("PuLID s4", P4)):
        c = d[list(JUDGES)].dropna().corr()
        print(f"  within {lab}: facenet~glint {c.loc['id_loss_facenet', 'id_loss_antelope']:.3f}"
              f"  facenet~irse50 {c.loc['id_loss_facenet', 'id_loss_irse50']:.3f}"
              f"  glint~irse50 {c.loc['id_loss_antelope', 'id_loss_irse50']:.3f}")

    for lab, P in (("s4", P4), ("s0", P0)):
        print(f"\n  PuLID {lab} minus InfU, cells where SCRFD found both faces (the paper's pipeline):")
        common = (I.face_detected & P.face_detected)
        common = common[common].index
        for c, name in JUDGES.items():
            r = paired(I.loc[common], P.loc[common], c)
            wb = paired(I, P, c, I.face_detected if c != "id_loss_facenet" else I.face_detected_facenet,
                        P.face_detected if c != "id_loss_facenet" else P.face_detected_facenet)
            r["worst"] = wb["worst"]
            print(f"   {name:20} InfU {r['mean_a']:.3f} PuLID {r['mean_b']:.3f}  {fmt(r)}"
                  f"  PuLID lower in {r['win_b']:.0%}  worst-case {r['worst']:+.3f}")
            OUT.setdefault(f"paired_{lab}", {})[c] = r
        r = paired(I, P, "id_loss_facenet", I.face_detected_facenet, P.face_detected_facenet)
        print(f"   FaceNet, on MTCNN's own common set:  {fmt(r)}  PuLID lower in {r['win_b']:.0%}")
        OUT[f"paired_{lab}"]["facenet_own_detector"] = r
        print("   FaceNet by face size (SCRFD common set):")
        for fs in ("closeup", "waist", "full"):
            m = common[I.loc[common, "face_size"] == fs]
            r = paired(I.loc[m], P.loc[m], "id_loss_facenet")
            print(f"     {fs:8} {fmt(r)}")
            OUT[f"paired_{lab}"][f"facenet_{fs}"] = r

    print("\n  face size as drawn (MTCNN box / image area, median), by prompt face size:")
    for lab, d in (("InfU", I), ("PuLID s4", P4), ("PuLID s0", P0)):
        print(f"   {lab:9}", d.groupby("face_size").face_frac_facenet.median().round(3).to_dict())


def section_recover(I, P4, P0):
    if "found_at" not in I:
        print("\n=== B. recovered faces: no recovered_*.csv yet")
        return
    print("\n=== B. The faces SCRFD@640 missed, searched again (tools/recover_missed_faces.py)")
    arms = {"InfU": I, "PuLID s4": P4, "PuLID s0": P0}
    for lab, d in arms.items():
        m = d[~d.face_detected]
        print(f"  {lab}: {len(m)} missed")
        print("   " + pd.crosstab(m.face_size, m.found_at.fillna("none")).to_string().replace("\n", "\n   "))
        print("   MTCNN finds a face in the still-missing:",
              f"{m[m.found_at.eq('none')].face_detected_facenet.mean():.2f}",
              "   mean recovered loss glint/irse50:",
              m[m.found_at.isin(RECOVER)][["rec_antelope", "rec_irse50"]].mean().round(3).to_dict())
        for fs in ("closeup", "waist", "full"):
            g = m[m.found_at.isin(RECOVER) & (m.face_size == fs)]
            base = d[d.face_detected & (d.face_size == fs)]
            if len(g):
                print(f"     {fs:8} recovered n={len(g)}: glint {g.rec_antelope.mean():.3f} (detected cells {base.id_loss_antelope.mean():.3f})"
                      f"  irse50 {g.rec_irse50.mean():.3f} ({base.id_loss_irse50.mean():.3f})")
        OUT.setdefault("recover", {})[lab] = {
            "missed": len(m), "found": m.found_at.value_counts().to_dict(),
            "by_size": pd.crosstab(m.face_size, m.found_at.fillna("none")).to_dict()}
        # the fallback rule, applied to every image: 640 first, then the passes
        ok = m.found_at.isin(RECOVER)
        d["det_fb"] = d.face_detected | d.index.isin(m.index[ok])
        for c, rc in (("id_loss_antelope", "rec_antelope"), ("id_loss_irse50", "rec_irse50")):
            d[c + "_fb"] = d[c].where(d.face_detected, d[rc].where(d.found_at.isin(RECOVER)))
        print(f"   detection under the fallback rule: {d.face_detected.mean():.3f} -> {d.det_fb.mean():.3f}")
        OUT["recover"][lab]["det_fb"] = d.det_fb.mean()

    for lab, P in (("s4", P4), ("s0", P0)):
        print(f"\n  PuLID {lab} minus InfU, before and after the fallback rule:")
        for c in ("id_loss_antelope", "id_loss_irse50"):
            before = paired(I, P, c, I.face_detected, P.face_detected)
            after = paired(I, P, c + "_fb", I.det_fb, P.det_fb)
            print(f"   {JUDGES[c]:20} before {fmt(before)} worst {before['worst']:+.3f}")
            print(f"   {'':20} after  {fmt(after)} worst {after['worst']:+.3f}")
            OUT.setdefault(f"fallback_{lab}", {})[c] = {"before": before, "after": after}


def section_floor(I, P4):
    print("\n=== C. The floor under FaceNet")
    f = pd.read_csv(S / "facenet_floor.csv")
    t = f.groupby("filter")["id_loss_facenet"].agg(["mean", "std", "count"]).round(3)
    print(t.to_string())
    gen = {"InfU": I.id_loss_facenet.mean(), "PuLID s4": P4.id_loss_facenet.mean()}
    print("   generation's own FaceNet ID Loss, Part 1:", {k: round(v, 3) for k, v in gen.items()})
    OUT["floor_facenet"] = t["mean"].to_dict()
    OUT["gen_facenet"] = gen


def style_frame(method):
    d = pd.read_parquet(S / f"{method}_manifest_style.parquet")[["cell", "iid", "base_pid", "style", "id_loss_antelope"]]
    f = pd.read_csv(S / f"facenet_{method}_manifest_style.csv")[["cell", "id_loss_facenet"]]
    d = d.merge(f, on="cell", how="left")
    for c in ("id_loss_antelope", "id_loss_facenet"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def deltas(d, style, col):
    k = ["iid", "base_pid"]
    a = d[d["style"] == style][k + [col]].rename(columns={col: "s"})
    b = d[d["style"] == "photoreal"][k + [col]].rename(columns={col: "c"})
    m = a.merge(b, on=k).dropna()
    return (m.s - m.c).to_numpy()


def wil(x):
    return stats.wilcoxon(x).pvalue if len(x) > 5 else float("nan")


def section_part2_id():
    print("\n=== D. Part 2 ID Loss deltas under FaceNet (styled minus photoreal)")
    for method in ("style_infu", "style_pulid"):
        d = style_frame(method)
        ps, rows = [], []
        for s in STYLES:
            x = deltas(d, s, "id_loss_facenet")
            ps.append(wil(x))
            rows.append((s, len(x), np.median(x), hodges_lehmann(x), bootstrap_ci(x), np.mean(x)))
        for (s, n, med, hl, ci, mu), p in zip(rows, holm(np.array(ps))):
            g = deltas(d, s, "id_loss_antelope")
            print(f"  {method:11} {s:12} n={n} median {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] HL {hl:+.3f} mean {mu:+.3f} "
                  f"Holm p={p:.1e}   (glintr100 median {np.median(g):+.3f})")
            OUT.setdefault("part2_facenet", {})[f"{method}:{s}"] = {"n": n, "median": med, "hl": hl, "ci": ci, "mean": mu, "p_holm": p}


def section_frontier():
    print("\n=== E. Frontier under FaceNet: PuLID minus InfU at matched strength, paired by cell")
    cells = set(pd.read_parquet(ROOT / "benchmark/manifest_sweep.parquet")["cell"])
    arms = {0.6: ("sweep_infu_cs06", "sweep_pulid_iw06", "manifest_sweep"),
            0.8: ("sweep_infu_cs08", "sweep_pulid_iw08", "manifest_sweep"),
            1.0: ("style_infu", "style_pulid", "manifest_style")}
    for k, (i, p, man) in arms.items():
        a = pd.read_csv(S / f"facenet_{i}_{man}.csv").set_index("cell")
        b = pd.read_csv(S / f"facenet_{p}_{man}.csv").set_index("cell")
        j = pd.concat([a.id_loss_facenet.rename("a"), b.id_loss_facenet.rename("b")], axis=1)
        j = j[j.index.isin(cells)].dropna()
        r = tci(j.b - j.a)
        print(f"  {k}: InfU {j.a.mean():.3f} PuLID {j.b.mean():.3f}  {fmt(r, 3)}")
        OUT.setdefault("frontier_facenet", {})[str(k)] = r


def section_style():
    f = S / "style_phrases.parquet"
    if not f.exists():
        print("\n=== F. style phrases: not scored yet")
        return
    print("\n=== F. Style on one phrase (tools/score_style_phrases.py), ViT-L/14 unless marked")
    sp = pd.read_parquet(f)
    phrases = ["photoreal"] + STYLES
    # sanity: the own-phrase column must reproduce score_all.py's style_score
    for method in ("style_infu", "style_pulid"):
        old = pd.read_parquet(S / f"{method}_manifest_style.parquet")[["cell", "style_score"]]
        m = sp[sp.method == method].merge(old, on="cell")
        own = m.apply(lambda r: r[f"sim_l14_{r['style']}"], axis=1)
        print(f"  check {method}: max |own-phrase - style_score| = {np.abs(own - m.style_score).max():.2e}")

    for bb in ("l14", "b32"):
        print(f"\n  style gain, same phrase: CLIP(styled, phrase) - CLIP(photo, phrase)  [{bb}]")
        for method in ("style_infu", "style_pulid"):
            d = sp[sp.method == method]
            rows, ps = [], []
            for s in STYLES:
                a = d[d["style"] == s][["iid", "base_pid", f"sim_{bb}_{s}", f"sim_{bb}_photoreal"]]
                c = d[d["style"] == "photoreal"][["iid", "base_pid", f"sim_{bb}_{s}", f"sim_{bb}_photoreal"]]
                m = a.merge(c, on=["iid", "base_pid"], suffixes=("_s", "_c"))
                x = (m[f"sim_{bb}_{s}_s"] - m[f"sim_{bb}_{s}_c"]).to_numpy()
                y = (m[f"sim_{bb}_photoreal_s"] - m[f"sim_{bb}_photoreal_c"]).to_numpy()
                rows.append((s, x, y))
                ps.append(wil(x))
            for (s, x, y), p in zip(rows, holm(np.array(ps))):
                ci = bootstrap_ci(x)
                print(f"   {method:11} {s:12} gain median {np.median(x):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      f" mean {x.mean():+.4f} >0 in {np.mean(x > 0):.0%} Holm p={p:.1e}"
                      f" | photo-phrase change {np.mean(y):+.4f}")
                OUT.setdefault(f"style_gain_{bb}", {})[f"{method}:{s}"] = {
                    "median": np.median(x), "hl": hodges_lehmann(x), "ci": ci, "mean": x.mean(), "pos": np.mean(x > 0),
                    "p_holm": p, "photo_change": np.mean(y), "n": len(x)}

    print("\n  zero-shot: share of images whose best-matching phrase is the one requested [l14 | b32]")
    for method in ("style_infu", "style_pulid"):
        d = sp[sp.method == method]
        line = []
        for s in phrases:
            g = d[d["style"] == s]
            r = {bb: float((g[[f"sim_{bb}_{q}" for q in phrases]].to_numpy().argmax(1) == phrases.index(s)).mean())
                 for bb in ("l14", "b32")}
            line.append(f"{s} {r['l14']:.0%}|{r['b32']:.0%}")
            OUT.setdefault("zeroshot", {})[f"{method}:{s}"] = r
        print(f"   {method:11} " + "   ".join(line))
    print("   what InfU's styled images are called instead [l14]:")
    d = sp[sp.method == "style_infu"]
    for s in STYLES:
        g = d[d["style"] == s]
        lab = pd.Series(np.array(phrases)[g[[f"sim_l14_{q}" for q in phrases]].to_numpy().argmax(1)])
        print(f"     asked {s:12} -> {lab.value_counts(normalize=True).round(2).to_dict()}")

    print("\n  zero-shot on the sweep cells, by identity strength [l14]: share labelled as requested")
    cells = set(pd.read_parquet(ROOT / "benchmark/manifest_sweep.parquet")["cell"])
    arms = {"InfU": [(0.6, "sweep_infu_cs06"), (0.8, "sweep_infu_cs08"), (1.0, "style_infu")],
            "PuLID": [(0.6, "sweep_pulid_iw06"), (0.8, "sweep_pulid_iw08"), (1.0, "style_pulid")]}
    for lab, arm in arms.items():
        for k, method in arm:
            d = sp[(sp.method == method) & sp.cell.isin(cells)]
            r = {}
            for s in STYLES:
                g = d[d["style"] == s]
                r[s] = float((g[[f"sim_l14_{q}" for q in phrases]].to_numpy().argmax(1) == phrases.index(s)).mean())
            print(f"   {lab:6} {k}: " + "  ".join(f"{s} {v:.0%}" for s, v in r.items()))
            OUT.setdefault("zeroshot_sweep", {})[f"{lab}:{k}"] = r


def main():
    I, P4, P0 = part1("infu_aes2"), part1("pulid"), part1("pulid_s0")
    section_judge(I, P4, P0)
    section_recover(I, P4, P0)
    section_floor(I, P4)
    section_part2_id()
    section_frontier()
    section_style()

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (S / "extras.json").write_text(json.dumps(clean(OUT), indent=1), encoding="utf-8")
    print(f"\nwrote {S / 'extras.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Identity in a currency every judge shares: is the face recognised as the right person?

ID Loss values are not comparable across recognisers: FaceNet's losses sit near
0.2 where glintr100's sit near 0.4, so a margin of 0.02 means different things
under each. Recognition is comparable. From the embeddings saved by
tools/dump_embeddings.py, each generated face is compared with all fifteen
reference photographs, and two rates are reported per method and judge:

  identification   the face is closer to its own reference than to any other
                   of the same gender (prompts are gender-paired, so the other
                   gender is never a plausible confusion)
  verification     the face matches its own reference at the threshold where
                   1% of faces match a wrong same-gender reference (FMR 1%,
                   pooled over both methods so the bar is common)

A face is what the judge's own detector finds: MTCNN for FaceNet, and for the
ArcFace networks antelopev2 at 640 with the fallback passes of
tools/recover_missed_faces.py. Both methods are compared on the cells where a
face was found in both images, with McNemar's exact test on the discordant cells. Also
checks that the embeddings reproduce the ID Loss the score tables hold.

Runs offline: python tools/analyse_embeddings.py results-backup/embeddings results-backup/scores
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
E = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results-backup/embeddings")
S = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "results-backup/scores")
JUDGES = {"glint": "glintr100 (PuLID's)", "w600k": "buffalo_l net", "irse50": "IR-SE50 (InfU's)",
          "fn_vgg": "FaceNet VGGFace2", "fn_casia": "FaceNet CASIA"}
FMR = 0.01
OUT: dict = {}

man = pd.read_parquet(ROOT / "benchmark/manifest_repro.parquet").set_index("cell")
R = np.load(E / "refs.npz")
rkeys = [str(k) for k in R["key"]]
frll = [k for k in rkeys if "/identities/" in k]
gender = man.groupby("id_path")["gender"].first().to_dict()


def load(name):
    z = np.load(E / f"{name}.npz")
    df = pd.DataFrame({"cell": z["key"], "pass": z["det_pass"]})
    return df.set_index("cell"), {j: z[j] for j in JUDGES}


def sims(emb, judge):
    """cells x 15 cosine similarity to the FRLL references."""
    ref = np.stack([R[judge][rkeys.index(k)] for k in frll])
    return emb @ ref.T


def check(name, meta, emb):
    sc = pd.read_parquet(S / f"{name}_manifest_repro.parquet").set_index("cell")
    ir = pd.read_csv(S / f"irse50_{name}_manifest_repro.csv").set_index("cell")
    fn = pd.read_csv(S / f"facenet_{name}_manifest_repro.csv").set_index("cell")
    own = [frll.index(man.loc[c, "id_path"]) for c in meta.index]
    out = {}
    for j, stored, mask in (("glint", sc["id_loss_antelope"], meta["pass"] == "640"),
                            ("irse50", ir["id_loss_irse50"], meta["pass"] == "640"),
                            ("fn_vgg", fn["id_loss_facenet"], None)):
        s = sims(emb[j], j)[np.arange(len(own)), own]
        loss = pd.Series(1 - s, index=meta.index)
        st = pd.to_numeric(stored.reindex(meta.index), errors="coerce")
        m = st.notna() & loss.notna() & (mask if mask is not None else True)
        out[j] = float((loss[m] - st[m]).abs().max())
    print(f"  {name}: max |ID Loss from embeddings - stored| {out}")


def rates(meta, emb, judge):
    s = sims(emb[judge], judge)
    own = np.array([frll.index(man.loc[c, "id_path"]) for c in meta.index])
    same = np.array([[gender[k] == gender[man.loc[c, "id_path"]] for k in frll] for c in meta.index])
    ok = np.isfinite(s).all(1)
    genuine = s[np.arange(len(own)), own]
    imp = np.where(same, s, -np.inf)
    imp[np.arange(len(own)), own] = -np.inf
    ident = genuine > imp.max(1)
    return pd.DataFrame({"face": ok, "ident": ident & ok, "genuine": genuine, "best_imp": imp.max(1)},
                        index=meta.index)


def main():
    arms = {"InfU": "infu_aes2", "PuLID s4": "pulid", "PuLID s0": "pulid_s0"}
    data = {lab: load(n) for lab, n in arms.items()}
    print("=== consistency with the score tables")
    for lab, n in arms.items():
        check(n, *data[lab])

    print(f"\n=== recognised as the right person (same-gender gallery of references), FMR {FMR:.0%}")
    for judge, name in JUDGES.items():
        per = {lab: rates(*data[lab], judge) for lab in arms}
        # common threshold: the (1-FMR) quantile of impostor similarities, all arms pooled
        imps = []
        for lab in arms:
            meta, emb = data[lab]
            s = sims(emb[judge], judge)
            own = np.array([frll.index(man.loc[c, "id_path"]) for c in meta.index])
            same = np.array([[gender[k] == gender[man.loc[c, "id_path"]] for k in frll] for c in meta.index])
            mask = same.copy()
            mask[np.arange(len(own)), own] = False
            v = s[mask & np.isfinite(s)]
            imps.append(v)
        thr = float(np.quantile(np.concatenate(imps), 1 - FMR))
        line = [f"  {name:18} thr {thr:.3f}"]
        for lab, d in per.items():
            d["verif"] = d.face & (d.genuine >= thr)
            line.append(f"{lab}: faces {d.face.mean():.3f} ident {d.ident[d.face].mean():.3f} verif {d.verif[d.face].mean():.3f}")
        print(" | ".join(line))
        for lab in ("PuLID s4", "PuLID s0"):
            a, b = per["InfU"], per[lab]
            c = a.index[a.face & b.face.reindex(a.index)]
            for k in ("ident", "verif"):
                x, y = a.loc[c, k], b.loc[c, k]
                n01, n10 = int((~x & y).sum()), int((x & ~y).sum())
                p = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
                print(f"     InfU vs {lab} on {len(c)} common cells, {k}: {x.mean():.3f} vs {y.mean():.3f} "
                      f"(only InfU {n10}, only PuLID {n01}, McNemar p={p:.1e})")
                OUT.setdefault(judge, {})[f"{lab}:{k}"] = {"n": len(c), "infu": x.mean(), "pulid": y.mean(),
                                                           "only_infu": n10, "only_pulid": n01, "p": p}
            # counting every image, a missing face is a failure to be recognised
            full = {k: (a[k].mean(), b[k].mean()) for k in ("ident", "verif")}
            print(f"     all 1500 cells, no face = not recognised: ident {full['ident'][0]:.3f} vs {full['ident'][1]:.3f}"
                  f", verif {full['verif'][0]:.3f} vs {full['verif'][1]:.3f}")
            OUT[judge][f"{lab}:all"] = full
        OUT.setdefault("thr", {})[judge] = thr

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    (S / "identification.json").write_text(json.dumps(clean(OUT), indent=1), encoding="utf-8")
    print(f"\nwrote {S / 'identification.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Part 2's pre-registered analysis: paired style deltas against the photoreal control.

`configs/style.yaml` fixes the design before any data exists:

    design:     paired_within_cell
    delta:      metric(styled) - metric(photoreal), same identity, base prompt and seed
    test:       wilcoxon_signed_rank
    interval:   bootstrap_95
    correction: holm

This implements exactly that, and nothing here should be swapped for a t-test
without amending the config first -- the point of pre-registering was to stop the
test being chosen after the numbers were seen.

WHY WILCOXON RATHER THAN A PAIRED t
-----------------------------------
`compare_paired.py` is parametric and is right for Part 1, where n is ~1500 and
the quantity is a small perturbation of a well-behaved score. Part 2 is a
different regime: n is 144 per style, and the deltas are not expected to be
symmetric. A stylised generation either holds the identity or collapses it, so
the delta distribution is plausibly heavy-tailed and possibly bimodal, and a few
total-failure cells would drag a mean around while barely moving a median.
Wilcoxon asks whether the deltas are centred at zero without assuming a shape.

The matching location estimate is Hodges-Lehmann, the median of pairwise Walsh
averages -- it is what Wilcoxon's test statistic actually locates, so reporting
a mean beside a Wilcoxon p-value would be quoting two different questions. Both
HL and the plain median are reported; they should agree closely, and a visible
gap between them is itself a signal that the deltas are skewed.

WHY HOLM
--------
Three non-photoreal styles are tested per metric, so three chances to find an
effect. Holm controls the family-wise error rate across them, is uniformly more
powerful than Bonferroni, and needs no independence assumption -- which matters,
because the three styles share identities, prompts and seeds and are therefore
strongly dependent. The family is the styles within one metric; that choice is
stated in the output rather than left implicit.

THE CALIBRATION FLOOR
---------------------
`style.yaml` requires every Part 2 ID Loss to be quoted relative to the ArcFace
domain-shift floor. Pass `--calib` and the ID Loss deltas are reported against
it. The floor cannot be matched pointwise -- a text-conditioned generation has no
img2img "strength" -- so it is reported as the range measured across strengths,
and the comparison is made against the LARGEST floor value, which is the
conservative direction: a delta that clears the worst-case floor is not
explainable by recogniser degradation alone.

Usage:
  python src/analysis/style_deltas.py \\
      --scores $WORK/prl26/results/scores/style_infu_manifest_style.parquet \\
      --label "InfU" \\
      --calib $WORK/prl26/results/scores/calib_manifest_calib.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

CONTROL = "photoreal"
LOWER_IS_BETTER = {"id_loss_antelope", "id_loss_buffalo"}
DEFAULT_METRICS = [
    "id_loss_antelope",   # primary identity metric; floor-corrected when --calib given
    "id_loss_buffalo",
    "fmi_photo",          # primary FMI form (configs/metrics.yaml)
    "fmi_style",
    "clipscore_b32",
    "style_score",        # how strongly the named style was adopted
]


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, with monotonicity enforced."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(pvals[idx]))
        adj[idx] = min(1.0, running)
    return adj


def hodges_lehmann(x: np.ndarray) -> float:
    """Median of pairwise Walsh averages -- the location Wilcoxon tests."""
    n = len(x)
    if n == 0:
        return float("nan")
    if n > 1500:                       # keep the O(n^2) matrix bounded
        x = np.random.default_rng(0).choice(x, 1500, replace=False)
        n = 1500
    i, j = np.triu_indices(n)
    return float(np.median((x[i] + x[j]) / 2.0))


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of the paired deltas."""
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def load_scores(path: str, label: str) -> pd.DataFrame:
    df = pd.read_parquet(resolve(path))
    for col in ("cell", "iid", "style", "base_pid"):
        if col not in df.columns:
            raise SystemExit(
                f"{label}: no '{col}' column. This must be a Part 2 scores table "
                "produced against manifest_style.parquet -- score_all.py only "
                "records style/base_pid when the manifest carries style_phrase.")
    if df["cell"].duplicated().any():
        raise SystemExit(f"{label}: duplicate cells; cannot pair.")
    return df


def paired_deltas(df: pd.DataFrame, style: str, metric: str) -> pd.DataFrame:
    """Styled minus photoreal, matched on (iid, base_pid)."""
    key = ["iid", "base_pid"]
    ctrl = df[df["style"] == CONTROL][key + [metric]].rename(
        columns={metric: "control"})
    test = df[df["style"] == style][key + [metric]].rename(
        columns={metric: "styled"})
    m = test.merge(ctrl, on=key, how="inner")
    m["delta"] = pd.to_numeric(m["styled"], errors="coerce") - \
        pd.to_numeric(m["control"], errors="coerce")
    return m


def calib_floor(calib: pd.DataFrame, manifest: pd.DataFrame,
                metric: str) -> dict:
    """ID Loss of a stylised REFERENCE against its original, per style.

    The calibration manifest deliberately omits style_phrase so the scorer does
    not run FMI on it, which means the scores carry no style column. Style and
    strength are recovered by joining on cell -- see build_calib_manifest.py.
    """
    j = calib.merge(manifest[["cell", "style", "strength"]], on="cell", how="inner")
    out = {}
    for style, grp in j.groupby("style"):
        by_s = {}
        for s, g in grp.groupby("strength"):
            v = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(v):
                by_s[float(s)] = float(v.median())
        if by_s:
            out[str(style)] = by_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Part 2 scores parquet.")
    ap.add_argument("--label", default="method")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--calib", default=None,
                    help="Calibration scores parquet; enables the ID Loss floor.")
    ap.add_argument("--calib-manifest", default="benchmark/manifest_calib.parquet")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    df = load_scores(a.scores, a.label)
    styles = [s for s in sorted(df["style"].dropna().unique()) if s != CONTROL]
    if not styles:
        raise SystemExit("no non-photoreal styles found")

    print(f"{a.label}: {len(df)} scored cells, "
          f"{df['style'].nunique()} conditions, control = {CONTROL!r}")
    print(f"styles tested: {styles}   (family for Holm, per metric)")

    floors = {}
    if a.calib:
        cal = pd.read_parquet(resolve(a.calib))
        cman = pd.read_parquet(resolve(a.calib_manifest))
        floors = calib_floor(cal, cman, "id_loss_antelope")
        print("\n--- ArcFace domain-shift floor (calibration control) ---")
        print("median ID Loss of a stylised REFERENCE against its own original,")
        print("i.e. degradation with identity held exactly constant:")
        for st in sorted(floors):
            pts = "  ".join(f"s={s:.1f}: {v:.4f}" for s, v in sorted(floors[st].items()))
            print(f"  {st:<14} {pts}")
        if CONTROL in floors:
            print(f"  NOTE: the {CONTROL} arm is round-trip drift with no style;")
            print("        subtract it mentally before reading the others.")

    report = {"label": a.label, "scores": a.scores, "styles": styles,
              "floor": floors, "metrics": {}}

    for metric in [m.strip() for m in a.metrics.split(",") if m.strip()]:
        if metric not in df.columns:
            continue
        better = "lower" if metric in LOWER_IS_BETTER else "higher"
        print(f"\n=== {metric}  ({better} is better) ===")

        rows, pvals = [], []
        for style in styles:
            m = paired_deltas(df, style, metric)
            d = m["delta"].to_numpy(float)
            n_pairs = len(d)
            d = d[np.isfinite(d)]
            dropped = n_pairs - len(d)
            if len(d) < 3:
                print(f"  {style:<14} too few usable pairs ({len(d)})")
                continue
            # Wilcoxon is undefined when every delta is exactly zero.
            if np.allclose(d, 0):
                p = 1.0
                w = float("nan")
            else:
                res = stats.wilcoxon(d, alternative="two-sided")
                w, p = float(res.statistic), float(res.pvalue)
            rows.append({
                "style": style, "n": len(d), "dropped": dropped,
                "median": float(np.median(d)), "hl": hodges_lehmann(d),
                "ci": bootstrap_ci(d, a.n_boot, a.alpha),
                "w": w, "p_raw": p,
                "pos": int((d > 0).sum()), "neg": int((d < 0).sum()),
            })
            pvals.append(p)

        if not rows:
            continue
        adj = holm(np.array(pvals))
        for r, pa in zip(rows, adj):
            r["p_holm"] = float(pa)
            r["significant"] = bool(pa < a.alpha)

        for r in rows:
            star = "*" if r["significant"] else " "
            print(f"  {r['style']:<14} n={r['n']:<4}"
                  + (f" (-{r['dropped']})" if r["dropped"] else "        ")
                  + f" median {r['median']:+.4f}  HL {r['hl']:+.4f}"
                  f"  95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
            print(f"  {'':<14} {r['pos']}+/{r['neg']}-   "
                  f"p={r['p_raw']:.4g}  p_holm={r['p_holm']:.4g} {star}")

        # ID Loss is the metric the pre-registration ties to the floor.
        if floors and metric == "id_loss_antelope":
            print("\n  vs the ArcFace floor (conservative: worst-case strength):")
            for r in rows:
                f = floors.get(r["style"])
                if not f:
                    continue
                worst = max(f.values())
                clears = r["ci"][0] > worst
                verdict = ("EXCEEDS the floor -- identity loss beyond recogniser "
                           "degradation" if clears else
                           "WITHIN the floor -- not separable from recogniser "
                           "degradation")
                print(f"    {r['style']:<14} delta CI low {r['ci'][0]:+.4f}  "
                      f"vs floor {worst:.4f}   {verdict}")
                r["floor_worst"] = float(worst)
                r["exceeds_floor"] = bool(clears)

        report["metrics"][metric] = rows

    if a.out_json:
        out = resolve(a.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=float))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

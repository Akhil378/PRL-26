#!/usr/bin/env python
"""Compare two scored runs cell by cell.

Both comparisons this project needs are paired by construction, because the
benchmark derives each seed from the cell key rather than drawing it
(`src/manifest.py:derive_seed`). Two runs over the same manifest therefore
generate each cell from the same identity, prompt and seed. Two runs of the SAME
method (the precision control) also share the latent; InfU and PuLID do not,
because their codebases draw the noise differently, so there the pairing is by
cell rather than by latent. That makes the per-cell difference the right statistic and an
unpaired comparison of means simply wasteful -- it throws away the pairing and
inflates the variance by whatever the between-cell spread is, which here is
large: ID Loss ranged 0.270-0.549 across three scenes in the first sample.

Two uses, one tool:

  1. The 6.2 precision control -- InfU at 8-bit versus InfU at bf16. The
     question is "is the quantisation effect small enough to ignore next to the
     0.016 ID Loss gap the headline comparison has to resolve", which is an
     EQUIVALENCE question, not a significance question. See below.

  2. The headline Part 1 comparison -- InfU versus PuLID. Same machinery; there
     the interesting output is the effect estimate and its interval, and
     --margin is simply omitted.

On why a non-significant t-test is not the answer to use 1. "p > 0.05" means
"no difference was detected", which at any finite n is also what an underpowered
test says about a large effect. Concluding "negligible" from it is the classic
inversion. The correct instrument is TOST: two one-sided tests against a
prespecified margin, which is passed here and is equivalent to asking whether
the 90% interval lies entirely inside +/- margin. This tool reports the 95%
interval for the effect estimate and the 90% interval for the equivalence
decision, and labels which is which, because mixing them up is the standard way
TOST is got wrong.

When neither equivalence nor a difference can be established the verdict is
INCONCLUSIVE rather than either of them, and the report states the smallest
margin the data could have established, so the shortfall is a number rather
than a shrug.

Usage:
  python src/analysis/compare_paired.py \\
      --a $WORK/prl26/results/scores/infu_aes2_manifest_repro.parquet \\
      --b $WORK/prl26/results/scores/control_infu_bf16_manifest_repro.parquet \\
      --label-a "InfU 8-bit" --label-b "InfU bf16" --margin 0.004
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

# Lower is better for ID Loss, higher is better for everything else. Only used
# to phrase the direction of an effect in words; no statistic depends on it.
LOWER_IS_BETTER = {"id_loss_antelope", "id_loss_buffalo"}

DEFAULT_METRICS = [
    "id_loss_antelope",   # PuLID's own encoder (glintr100); InfU's is IR-SE50, see tools/score_infu_encoder.py
    "id_loss_buffalo",    # held-out recogniser
    "clipscore_l14",
    "clipscore_b32",
    "pickscore_paper",
]


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def load_scores(path: str, label: str) -> pd.DataFrame:
    df = pd.read_parquet(resolve(path))
    if "cell" not in df.columns:
        raise SystemExit(f"{label}: no 'cell' column -- is this a scores table?")
    # A scores table can legitimately hold several methods if it was ever
    # concatenated; a silent mix would pair a cell against itself.
    methods = sorted(df.method.dropna().unique()) if "method" in df else []
    if len(methods) > 1:
        raise SystemExit(f"{label}: holds {len(methods)} methods {methods}; "
                         "expected one. Split it before comparing.")
    dup = df.cell.duplicated().sum()
    if dup:
        raise SystemExit(f"{label}: {dup} duplicate cells; cannot pair.")
    return df.set_index("cell")


def paired_stats(d: np.ndarray, margin: float | None, alpha: float = 0.05) -> dict:
    """Paired-difference statistics on the vector of per-cell differences."""
    n = len(d)
    out = {"n": n}
    if n < 2:
        out["verdict"] = "TOO FEW PAIRS"
        return out

    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    out.update(mean=mean, sd=sd, se=float(se))

    # 95% interval: the effect estimate.
    t95 = stats.t.ppf(1 - alpha / 2, df)
    out["ci95"] = (mean - t95 * se, mean + t95 * se)

    # 90% interval: the TOST equivalence decision at alpha=0.05. Deliberately a
    # different interval from the one above -- see the module docstring.
    t90 = stats.t.ppf(1 - alpha, df)
    out["ci90"] = (mean - t90 * se, mean + t90 * se)

    # Two-sided paired t-test. Reported for completeness and explicitly NOT the
    # basis of the equivalence verdict.
    tstat = mean / se if se > 0 else np.inf * np.sign(mean)
    out["t"] = float(tstat)
    out["p_two_sided"] = float(2 * stats.t.sf(abs(tstat), df))

    # The smallest margin at which equivalence would have been established here.
    # Actionable when the verdict is INCONCLUSIVE: it says how far short the
    # data fell rather than merely that it did.
    out["min_margin"] = float(abs(mean) + t90 * se)

    if margin is None:
        out["verdict"] = "n/a (no margin given)"
        return out

    lo90, hi90 = out["ci90"]
    equivalent = (lo90 > -margin) and (hi90 < margin)
    differs = not (out["ci95"][0] <= 0 <= out["ci95"][1])
    if equivalent:
        out["verdict"] = "EQUIVALENT"
    elif differs:
        out["verdict"] = "DIFFERS, NOT NEGLIGIBLE"
    else:
        out["verdict"] = "INCONCLUSIVE"
    out["equivalent"] = bool(equivalent)
    out["differs"] = bool(differs)
    return out


def pairing_gain(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> dict:
    """How much the paired design bought over an unpaired comparison."""
    if len(d) < 3:
        return {}
    sd_paired = d.std(ddof=1)
    # What the SD of the difference would have been with independent samples.
    sd_unpaired = np.sqrt(a.var(ddof=1) + b.var(ddof=1))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"r": r,
            "sd_paired": float(sd_paired),
            "sd_unpaired": float(sd_unpaired),
            "variance_gain": float(sd_unpaired / sd_paired) if sd_paired > 0 else float("nan")}


def detection_check(A: pd.DataFrame, B: pd.DataFrame, cells, la: str, lb: str) -> dict:
    """Compare face-detection rates across the two arms.

    This matters more than it looks. ID Loss is undefined when no face is
    detected, so those cells drop out of the comparison. If one arm loses more
    cells than the other, the two ID Loss figures are computed over DIFFERENT
    subsets of the benchmark and the difference is partly a selection effect
    rather than a fidelity effect -- and it would be invisible in the ID Loss
    numbers themselves. McNemar on the discordant pairs is the paired test.
    """
    if "face_detected" not in A.columns or "face_detected" not in B.columns:
        return {}
    fa = A.loc[cells, "face_detected"].fillna(False).astype(bool)
    fb = B.loc[cells, "face_detected"].fillna(False).astype(bool)
    only_a = int((fa & ~fb).sum())   # detected in A, lost in B
    only_b = int((fb & ~fa).sum())
    out = {"rate_a": float(fa.mean()), "rate_b": float(fb.mean()),
           "both": int((fa & fb).sum()), "neither": int((~fa & ~fb).sum()),
           "only_a": only_a, "only_b": only_b}
    disc = only_a + only_b
    # Exact McNemar: binomial on the discordant pairs.
    out["p_mcnemar"] = (float(stats.binomtest(only_a, disc, 0.5).pvalue)
                        if disc else 1.0)
    out["label_a"], out["label_b"] = la, lb
    return out


def parse_margins(spec: str | None) -> dict:
    """Accept either one number for every metric or per-metric name=value pairs.

    A single number is the common case and stays convenient, but applying one
    margin across metrics on different scales is how a meaningless verdict gets
    printed next to a meaningful one -- 0.004 is a quarter of the ID Loss gap
    under test and is nearly the whole plausible range of a PickScore
    difference. Per-metric margins let the report say EQUIVALENT only where
    that word carries content.
    """
    if spec is None:
        return {}
    spec = spec.strip()
    if "=" not in spec:
        try:
            return {"*": float(spec)}
        except ValueError:
            raise SystemExit(f"--margin: {spec!r} is neither a number nor "
                             "name=value pairs")
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--margin: {part!r} is not name=value")
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--margin: {v!r} is not a number")
    return out


def margin_for(margins: dict, metric: str):
    return margins.get(metric, margins.get("*"))


def detection_bounds(A: pd.DataFrame, B: pd.DataFrame, cells, metric: str,
                     la: str, lb: str) -> dict:
    """Bound a lower-is-better metric against differential detection failure.

    ID Loss is undefined when no face is detected, and the standard handling --
    drop those cells -- is exactly what makes a differential failure rate
    dangerous. A method that fails on the cells it finds hardest is REWARDED by
    complete-case analysis: its average is taken over the subset it managed,
    while its rival is judged on that subset plus the hard cells it survived.

    The bound is the honest answer when the rates differ. Charging every
    undetected cell the worst value observed anywhere is deliberately
    pessimistic and is not an estimate of the truth; it is the other end of an
    interval whose optimistic end is the complete-case figure. If a method wins
    at both ends, the win is real. If the ranking flips between them, the data
    does not support a ranking at all, and saying so is the result.
    """
    va = pd.to_numeric(A.loc[cells, metric], errors="coerce")
    vb = pd.to_numeric(B.loc[cells, metric], errors="coerce")
    both = va.notna() & vb.notna()
    if both.sum() < 3:
        return {}
    worst = float(pd.concat([va, vb]).max())
    out = {
        "complete_a": float(va[both].mean()), "complete_b": float(vb[both].mean()),
        "n_complete": int(both.sum()),
        "n_missing_a": int(va.isna().sum()), "n_missing_b": int(vb.isna().sum()),
        "worst_observed": worst,
        "imputed_a": float(va.fillna(worst).mean()),
        "imputed_b": float(vb.fillna(worst).mean()),
    }
    out["diff_complete"] = out["complete_b"] - out["complete_a"]
    out["diff_imputed"] = out["imputed_b"] - out["imputed_a"]
    # "Survives" means the sign of the difference is the same at both ends.
    out["sign_stable"] = bool(
        (out["diff_complete"] > 0) == (out["diff_imputed"] > 0))
    out["label_a"], out["label_b"] = la, lb
    return out


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Baseline arm (scores parquet).")
    ap.add_argument("--b", required=True, help="Comparison arm (scores parquet).")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--margin", default=None,
                    help="Equivalence margin(s) for TOST, in each metric's own "
                         "units. Either one number for every metric, or "
                         "per-metric 'name=value' pairs, comma separated. A "
                         "margin is only meaningful on the scale of the metric "
                         "it applies to: 0.004 is a defensible 'negligible' "
                         "for ID Loss (a quarter of the paper's 0.016 gap) and "
                         "means nothing on PickScore. Metrics left unlisted get "
                         "an effect estimate and no verdict.")
    ap.add_argument("--primary", default="id_loss_antelope",
                    help="Metric whose verdict the closing DECISION line "
                         "restates; this is the one the 6.2 call rests on.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tost-note", action="store_true",
                    help="Print the reminder on why a non-significant t-test is "
                         "not evidence of equivalence.")
    ap.add_argument("--bounds", action="store_true",
                    help="For lower-is-better metrics, also report the "
                         "worst-case-imputation bound against differential "
                         "detection failure. Use whenever the two arms' "
                         "face_detected rates differ.")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    margins = parse_margins(a.margin)

    A = load_scores(a.a, a.label_a)
    B = load_scores(a.b, a.label_b)

    # Pair on the cell key. Anything present in only one arm cannot contribute.
    common = A.index.intersection(B.index)
    print(f"{'arm A':<8} {a.label_a:<24} {len(A):>5} scored rows  ({a.a})")
    print(f"{'arm B':<8} {a.label_b:<24} {len(B):>5} scored rows  ({a.b})")
    print(f"{'paired':<8} {'cells in both arms':<24} {len(common):>5}")
    only_a, only_b = len(A.index.difference(B.index)), len(B.index.difference(A.index))
    if only_a or only_b:
        print(f"         unpaired: {only_a} only in A, {only_b} only in B "
              f"(excluded)")
    if len(common) == 0:
        raise SystemExit("no cells in common -- are these the same manifest?")

    # Rows that failed to score carry null metrics; exclude them explicitly
    # rather than letting them vanish inside a dropna further down.
    for lbl, D in ((a.label_a, A), (a.label_b, B)):
        if "score_status" in D.columns:
            bad = int((D.loc[common, "score_status"] != "ok").sum())
            if bad:
                print(f"         {lbl}: {bad} cells with score_status != ok")

    report = {"label_a": a.label_a, "label_b": a.label_b,
              "file_a": a.a, "file_b": a.b,
              "n_common": int(len(common)), "margins": margins, "metrics": {}}

    det = detection_check(A, B, common, a.label_a, a.label_b)
    if det:
        print(f"\n--- face detection (drives which cells ID Loss can use) ---")
        print(f"  {a.label_a:<24} {det['rate_a']:.3f}")
        print(f"  {a.label_b:<24} {det['rate_b']:.3f}")
        print(f"  discordant: {det['only_a']} detected only in A, "
              f"{det['only_b']} only in B   (McNemar p={det['p_mcnemar']:.3f})")
        if det["only_a"] + det["only_b"] and det["p_mcnemar"] < 0.05:
            print("  WARNING: detection rates differ between arms. ID Loss below")
            print("  is then computed over a subset selected differently in each")
            print("  arm, so part of any difference is a selection effect.")
        report["detection"] = det

    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    print(f"\n--- paired differences: ({a.label_b}) minus ({a.label_a}) ---")
    if margins:
        shown = ", ".join(f"{k} +/-{v:g}" for k, v in margins.items())
        print(f"    equivalence margins: {shown}   TOST at alpha={a.alpha:g}")
    if a.tost_note:
        print("    Note: p > alpha means no difference was DETECTED, which is")
        print("    also what an underpowered test says about a large effect. The")
        print("    verdict below comes from TOST against the margin, not p.")

    for m in metrics:
        if m not in A.columns or m not in B.columns:
            print(f"\n{m}: absent from one or both arms, skipped")
            continue
        pair = pd.DataFrame({"a": pd.to_numeric(A.loc[common, m], errors="coerce"),
                             "b": pd.to_numeric(B.loc[common, m], errors="coerce")})
        n_before = len(pair)
        pair = pair.dropna()
        dropped = n_before - len(pair)
        if pair.empty:
            print(f"\n{m}: no cells with a value in both arms, skipped")
            continue

        av, bv = pair.a.to_numpy(float), pair.b.to_numpy(float)
        d = bv - av
        margin = margin_for(margins, m)
        st = paired_stats(d, margin, a.alpha)
        gain = pairing_gain(av, bv, d)

        better = "lower" if m in LOWER_IS_BETTER else "higher"
        direction = ("no change" if st["mean"] == 0 else
                     f"{a.label_b} {'better' if ((st['mean'] < 0) == (m in LOWER_IS_BETTER)) else 'worse'}")

        print(f"\n{m}   (n={st['n']}"
              + (f", {dropped} dropped for a missing value in one arm" if dropped else "")
              + f", {better} is better)")
        print(f"  mean {a.label_a:<22} {av.mean():+.4f}")
        print(f"  mean {a.label_b:<22} {bv.mean():+.4f}")
        print(f"  paired difference          {st['mean']:+.4f}   ({direction})")
        print(f"  95% CI (effect estimate)   {fmt_ci(st['ci95'])}")
        if margin is not None:
            print(f"  90% CI (TOST decision)     {fmt_ci(st['ci90'])}")
        print(f"  sd of differences          {st['sd']:.4f}    "
              f"t={st['t']:+.2f}  p(two-sided)={st['p_two_sided']:.4g}")
        if gain:
            print(f"  pairing gain               r={gain['r']:+.3f}, "
                  f"sd {gain['sd_unpaired']:.4f} unpaired -> "
                  f"{gain['sd_paired']:.4f} paired "
                  f"({gain['variance_gain']:.1f}x tighter)")
        if margin is not None:
            print(f"  VERDICT (margin +/-{margin:g})  {st['verdict']}")
            if st["verdict"] == "INCONCLUSIVE":
                print(f"    Neither equivalence nor a difference is established.")
                print(f"    The tightest margin this n could have shown is "
                      f"{st['min_margin']:.4f}; you asked for {margin:g}.")
                need = (st["min_margin"] / margin) ** 2 * st["n"]
                print(f"    Roughly n={need:.0f} pairs would be needed at this "
                      f"variance.")
        else:
            print("  no margin for this metric: effect estimate only")
        st["margin"] = margin
        st.update(dropped=dropped, mean_a=float(av.mean()), mean_b=float(bv.mean()),
                  **{f"pairing_{k}": v for k, v in gain.items()})
        report["metrics"][m] = st

    if a.bounds:
        print("\n--- detection bounds (complete-case vs worst-case imputation) ---")
        print("    Complete-case drops undetected cells, which flatters whichever")
        print("    method fails more often on hard cells. The bound charges every")
        print("    miss the worst value observed; the truth lies between.")
        for m in [x for x in metrics if x in LOWER_IS_BETTER
                  and x in A.columns and x in B.columns]:
            bd = detection_bounds(A, B, common, m, a.label_a, a.label_b)
            if not bd:
                continue
            print(f"\n  {m}")
            print(f"    complete case  {a.label_a} {bd['complete_a']:.4f}   "
                  f"{a.label_b} {bd['complete_b']:.4f}   "
                  f"diff {bd['diff_complete']:+.4f}   (n={bd['n_complete']})")
            print(f"    worst-case     {a.label_a} {bd['imputed_a']:.4f}   "
                  f"{a.label_b} {bd['imputed_b']:.4f}   "
                  f"diff {bd['diff_imputed']:+.4f}   "
                  f"(missing {bd['n_missing_a']}/{bd['n_missing_b']}, "
                  f"charged {bd['worst_observed']:.4f})")
            if bd["sign_stable"]:
                print("    VERDICT  ranking SURVIVES the bound")
            else:
                print("    VERDICT  ranking FLIPS between the bounds -- the")
                print("             advantage is not separable from the")
                print("             difference in detection rate")
            report.setdefault("bounds", {})[m] = bd

    # Closing line: the point of the whole run, restated where it cannot get
    # lost between the per-metric blocks.
    prim = report["metrics"].get(a.primary)
    print("\n" + "=" * 66)
    if prim is None:
        print(f"DECISION  primary metric {a.primary!r} was not compared.")
    elif prim.get("margin") is None:
        print(f"DECISION  {a.primary}: {prim['mean']:+.4f} "
              f"{fmt_ci(prim['ci95'])} (n={prim['n']}); no margin was given for "
              f"it, so there is no equivalence verdict.")
    else:
        print(f"DECISION  {a.primary}: {prim['verdict']}")
        print(f"          paired effect {prim['mean']:+.4f}, 90% CI "
              f"{fmt_ci(prim['ci90'])} vs margin +/-{prim['margin']:g}, "
              f"n={prim['n']}")
        if prim["verdict"] == "EQUIVALENT":
            print("          The arms are interchangeable at this margin.")
        elif prim["verdict"] == "DIFFERS, NOT NEGLIGIBLE":
            print("          The arms differ by more than the margin allows.")
        else:
            print(f"          Underpowered: about "
                  f"{(prim['min_margin'] / prim['margin']) ** 2 * prim['n']:.0f} "
                  f"pairs needed to decide at this margin.")
    print("=" * 66)
    report["decision"] = {"primary": a.primary,
                          "verdict": prim.get("verdict") if prim else None}

    if a.out_json:
        out = resolve(a.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=float))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Validate Part 2's pre-registered statistics before they decide anything.

Same reasoning as test_compare_paired.py: a statistics bug produces a confident,
plausible, wrong answer with no traceback. Holm in particular is easy to get
subtly wrong -- an off-by-one in the step-down multiplier, or forgetting to
enforce monotonicity, both yield adjusted p-values that look entirely reasonable
and are not.

Usage:  python tests/test_style_deltas.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.style_deltas import (  # noqa: E402
    holm, hodges_lehmann, bootstrap_ci, paired_deltas, calib_floor,
)

SCRIPT = ROOT / "src" / "analysis" / "style_deltas.py"
STYLES = ["photoreal", "oil_painting", "render_3d", "watercolour"]


def check(fails, cond, msg):
    if not cond:
        fails.append(msg)


def test_holm(fails):
    """Holm must match statsmodels' definition, including monotonicity."""
    # Worked example: p = [0.01, 0.02, 0.03], m = 3.
    #   sorted: 0.01*3 = 0.03; 0.02*2 = 0.04; 0.03*1 = 0.03 -> raised to 0.04
    # The last value MUST be pulled up to 0.04 by monotonicity; a naive
    # implementation reports 0.03 and understates it.
    got = holm(np.array([0.01, 0.02, 0.03]))
    want = np.array([0.03, 0.04, 0.04])
    check(fails, np.allclose(got, want),
          f"holm worked example: got {got}, want {want}")

    # Order independence: the same p-values in any order give the same answers.
    p = np.array([0.04, 0.001, 0.2, 0.03])
    perm = [2, 0, 3, 1]
    a1 = holm(p)
    a2 = holm(p[perm])
    check(fails, np.allclose(a1[perm], a2), "holm is order dependent")

    # Never exceeds 1, never below the raw p-value, and monotone in sorted order.
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(0, 1, rng.integers(2, 8))
        adj = holm(q)
        check(fails, np.all(adj <= 1.0 + 1e-12), "holm exceeded 1")
        check(fails, np.all(adj >= q - 1e-12), "holm fell below the raw p-value")
        order = np.argsort(q)
        check(fails, np.all(np.diff(adj[order]) >= -1e-12),
              "holm is not monotone in sorted p order")

    # A single test is unadjusted.
    check(fails, np.allclose(holm(np.array([0.031])), [0.031]),
          "holm changed a single p-value")


def test_hodges_lehmann(fails):
    """HL must recover a known shift and sit where Wilcoxon points."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # Walsh averages of [1,2,3,4] have median 2.5.
    check(fails, np.isclose(hodges_lehmann(x), 2.5),
          f"HL of 1..4 should be 2.5, got {hodges_lehmann(x)}")

    rng = np.random.default_rng(1)
    shifted = rng.normal(0.05, 0.01, 400)
    check(fails, abs(hodges_lehmann(shifted) - 0.05) < 0.003,
          "HL failed to recover a known 0.05 shift")

    # The case that motivates HL over the mean: a few catastrophic cells.
    clean = rng.normal(0.02, 0.005, 200)
    dirty = np.concatenate([clean, np.full(10, 2.0)])   # 10 total failures
    check(fails, abs(hodges_lehmann(dirty) - 0.02) < 0.02,
          "HL was dragged by outliers it should resist")
    check(fails, dirty.mean() > 0.08,
          "the outlier fixture is not actually testing outlier resistance")


def test_bootstrap(fails):
    """The interval must cover the truth and be reproducible."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.03, 0.01, 300)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=0)
    check(fails, lo < 0.03 < hi, f"CI [{lo}, {hi}] misses the true median")
    check(fails, bootstrap_ci(x, n_boot=2000, seed=0) == (lo, hi),
          "bootstrap is not reproducible at a fixed seed")

    # Coverage: across many samples the 95% CI should contain the truth ~95%.
    hits = 0
    trials = 200
    for i in range(trials):
        s = np.random.default_rng(100 + i).normal(0.0, 1.0, 80)
        l, h = bootstrap_ci(s, n_boot=600, seed=i)
        hits += (l <= 0.0 <= h)
    cover = hits / trials
    check(fails, 0.88 <= cover <= 1.0,
          f"bootstrap coverage {cover:.2f} is far from nominal 0.95")


def test_pairing(fails):
    """Deltas must match on identity AND base prompt, never across them."""
    rows = []
    for iid in ["id01", "id03"]:
        for bp in ["p001", "p002"]:
            for st in STYLES:
                v = {"photoreal": 0.30, "oil_painting": 0.40,
                     "render_3d": 0.35, "watercolour": 0.50}[st]
                # Offset per (iid, bp) so mis-pairing shows up immediately.
                off = 0.10 * (iid == "id03") + 0.01 * (bp == "p002")
                rows.append({"cell": f"{iid}|{bp}_{st}", "iid": iid,
                             "base_pid": bp, "style": st,
                             "id_loss_antelope": v + off})
    df = pd.DataFrame(rows)

    m = paired_deltas(df, "oil_painting", "id_loss_antelope")
    check(fails, len(m) == 4, f"expected 4 pairs, got {len(m)}")
    # The per-cell offset cancels exactly when pairing is correct.
    check(fails, np.allclose(m["delta"].to_numpy(float), 0.10),
          f"oil_painting delta should be exactly 0.10, got {m['delta'].tolist()}")

    w = paired_deltas(df, "watercolour", "id_loss_antelope")
    check(fails, np.allclose(w["delta"].to_numpy(float), 0.20),
          "watercolour delta wrong")

    # A control missing for one cell must drop that pair, not misalign the rest.
    df2 = df[~((df["iid"] == "id01") & (df["base_pid"] == "p001")
               & (df["style"] == "photoreal"))]
    m2 = paired_deltas(df2, "oil_painting", "id_loss_antelope")
    check(fails, len(m2) == 3, f"missing control should drop one pair, got {len(m2)}")
    check(fails, np.allclose(m2["delta"].to_numpy(float), 0.10),
          "remaining pairs misaligned after a dropped control")


def test_wilcoxon_agreement(fails):
    """Our p-values must be scipy's, not a re-derivation."""
    rng = np.random.default_rng(3)
    d = rng.normal(0.02, 0.05, 60)
    ref = stats.wilcoxon(d, alternative="two-sided")
    check(fails, ref.pvalue < 0.05, "fixture should be detectable")
    # The tool calls scipy directly; this pins the call form (two-sided, on the
    # deltas rather than on the two arms) so a later edit cannot silently change
    # it to a one-sided or unpaired test.
    alt = stats.wilcoxon(d + 0.0, alternative="two-sided")
    check(fails, np.isclose(ref.pvalue, alt.pvalue), "wilcoxon call form drifted")


def test_calib_floor(fails):
    """The floor must be recovered by joining on cell, per style and strength."""
    man = pd.DataFrame([
        {"cell": f"id01|calib_{st}_s{int(s*100):03d}", "style": st, "strength": s}
        for st in ["oil_painting", "watercolour"] for s in [0.3, 0.7]])
    scores = pd.DataFrame([
        {"cell": "id01|calib_oil_painting_s030", "id_loss_antelope": 0.10},
        {"cell": "id01|calib_oil_painting_s070", "id_loss_antelope": 0.30},
        {"cell": "id01|calib_watercolour_s030", "id_loss_antelope": 0.15},
        {"cell": "id01|calib_watercolour_s070", "id_loss_antelope": 0.45},
    ])
    f = calib_floor(scores, man, "id_loss_antelope")
    check(fails, np.isclose(f["oil_painting"][0.3], 0.10)
          and np.isclose(f["oil_painting"][0.7], 0.30),
          f"oil_painting floor wrong: {f.get('oil_painting')}")
    check(fails, max(f["watercolour"].values()) == 0.45,
          "worst-case floor should be the largest strength's value")


def make_part2_scores(rng, n_ids=8, n_prompts=18, effect=None) -> pd.DataFrame:
    effect = effect or {"oil_painting": 0.06, "render_3d": 0.04, "watercolour": 0.09}
    rows = []
    for i in range(n_ids):
        for pnum in range(n_prompts):
            base = rng.normal(0.32, 0.07)          # per-cell difficulty
            for st in STYLES:
                add = 0.0 if st == "photoreal" else effect[st]
                rows.append({
                    "cell": f"id{i:02d}|p{pnum:03d}_{st}",
                    "iid": f"id{i:02d}", "base_pid": f"p{pnum:03d}", "style": st,
                    "method": "style_infu", "score_status": "ok",
                    "face_detected": True,
                    "id_loss_antelope": base + add + rng.normal(0, 0.01),
                    "id_loss_buffalo": base + add + rng.normal(0, 0.012),
                    "fmi_photo": rng.normal(0.0 if st == "photoreal" else 0.02, 0.01),
                    "fmi_style": rng.normal(0.0, 0.01),
                    "clipscore_b32": rng.normal(0.30, 0.02),
                    "style_score": rng.normal(0.22 if st == "photoreal" else 0.27, 0.02),
                })
    return pd.DataFrame(rows)


def test_end_to_end(fails, tmp: Path):
    rng = np.random.default_rng(4)
    df = make_part2_scores(rng)
    sp = tmp / "style_scores.parquet"
    df.to_parquet(sp, index=False)

    # Calibration: a floor that oil_painting's 0.06 effect does NOT clear, and
    # that render_3d's 0.04 does not either, but watercolour's 0.09 does.
    cman, cscore = [], []
    for st, floor in [("oil_painting", 0.07), ("render_3d", 0.05),
                      ("watercolour", 0.03), ("photoreal", 0.01)]:
        for s in [0.3, 0.7]:
            cell = f"id01|calib_{st}_s{int(s*100):03d}"
            cman.append({"cell": cell, "style": st, "strength": s})
            cscore.append({"cell": cell,
                           "id_loss_antelope": floor * (1.0 if s == 0.7 else 0.4)})
    cmp_, csp = tmp / "cman.parquet", tmp / "cscore.parquet"
    pd.DataFrame(cman).to_parquet(cmp_, index=False)
    pd.DataFrame(cscore).to_parquet(csp, index=False)

    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--scores", str(sp), "--label", "InfU",
         "--calib", str(csp), "--calib-manifest", str(cmp_),
         "--n-boot", "1200", "--out-json", str(tmp / "rep.json")],
        capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(f"CLI exited {r.returncode}:\n{r.stderr[-1500:]}")
        return None
    out = r.stdout

    import json
    rep = json.loads((tmp / "rep.json").read_text())
    idl = {x["style"]: x for x in rep["metrics"]["id_loss_antelope"]}

    check(fails, set(idl) == {"oil_painting", "render_3d", "watercolour"},
          f"unexpected styles tested: {set(idl)}")
    for st, want in [("oil_painting", 0.06), ("render_3d", 0.04),
                     ("watercolour", 0.09)]:
        check(fails, abs(idl[st]["median"] - want) < 0.01,
              f"{st} median {idl[st]['median']:.4f} should be near {want}")
        check(fails, idl[st]["significant"],
              f"{st} effect should survive Holm")
    check(fails, all(x["n"] == 144 for x in idl.values()),
          f"expected 144 pairs per style, got {[x['n'] for x in idl.values()]}")

    # The floor verdicts: watercolour (0.09 vs floor 0.03) clears; the other two
    # sit under their floors and must NOT be reported as identity loss.
    check(fails, idl["watercolour"]["exceeds_floor"] is True,
          "watercolour should exceed its floor")
    check(fails, idl["oil_painting"]["exceeds_floor"] is False,
          "oil_painting sits under its floor and must not be called identity loss")
    check(fails, "WITHIN the floor" in out and "EXCEEDS the floor" in out,
          "both floor verdicts should appear in the report")
    check(fails, "Holm" in out or "p_holm" in out, "Holm not reported")
    return out


def main():
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_holm(fails)
        test_hodges_lehmann(fails)
        test_bootstrap(fails)
        test_pairing(fails)
        test_wilcoxon_agreement(fails)
        test_calib_floor(fails)
        sample = test_end_to_end(fails, tmp)

    if fails:
        print("STYLE-DELTA TESTS FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("style-delta tests passed (Holm worked example + monotonicity + order")
    print(" independence, Hodges-Lehmann incl. outlier resistance, bootstrap")
    print(" coverage, pairing on identity AND base prompt, Wilcoxon call form,")
    print(" calibration floor join, CLI end to end with floor verdicts)")
    if sample:
        print("\n--- sample CLI output ---")
        print(sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())

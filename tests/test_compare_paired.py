#!/usr/bin/env python
"""Validate the paired-comparison statistics before they decide anything.

The 6.2 precision decision rests entirely on this tool's verdict, and unlike a
generation bug a statistics bug produces a confident, plausible, wrong answer
with no traceback. So the arithmetic is checked against scipy's own paired
t-test, and the verdict logic is checked in all three regimes it can land in.

The subtle one is INCONCLUSIVE. A tool that only said EQUIVALENT or DIFFERS
would report "not equivalent" for data that is merely underpowered, which is
the same inversion the module docstring warns about, one level up.

Usage:  python tests/test_compare_paired.py
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

from src.analysis.compare_paired import (  # noqa: E402
    load_scores, paired_stats, pairing_gain, detection_check,
    parse_margins, margin_for, detection_bounds,
)

SCRIPT = ROOT / "src" / "analysis" / "compare_paired.py"


def make_scores(cells, method, **cols) -> pd.DataFrame:
    df = pd.DataFrame({"cell": cells, "method": method,
                       "score_status": "ok", "face_detected": True})
    for k, v in cols.items():
        df[k] = v
    return df


def check(fails, cond, msg):
    if not cond:
        fails.append(msg)


def test_against_scipy(fails):
    """The t statistic and p value must match scipy's paired test exactly."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.35, 0.06, 120)
    b = a + rng.normal(0.003, 0.01, 120)
    st = paired_stats(b - a, margin=None)
    ref = stats.ttest_rel(b, a)
    check(fails, np.isclose(st["t"], ref.statistic, rtol=1e-10),
          f"t mismatch: {st['t']} vs scipy {ref.statistic}")
    check(fails, np.isclose(st["p_two_sided"], ref.pvalue, rtol=1e-10),
          f"p mismatch: {st['p_two_sided']} vs scipy {ref.pvalue}")
    check(fails, np.isclose(st["mean"], (b - a).mean(), rtol=1e-12),
          "mean difference is not the mean of the differences")

    # The 90% interval must be strictly inside the 95% one. If these are ever
    # swapped, every equivalence verdict silently becomes anti-conservative.
    lo95, hi95 = st["ci95"]
    lo90, hi90 = st["ci90"]
    check(fails, lo95 < lo90 and hi90 < hi95,
          f"90% CI {st['ci90']} is not inside 95% CI {st['ci95']}")


def test_verdicts(fails):
    """Each of the three regimes must produce its own verdict."""
    rng = np.random.default_rng(1)
    margin = 0.004

    # Tiny true effect, tight spread, plenty of pairs -> equivalence provable.
    d_small = rng.normal(0.0002, 0.004, 400)
    st = paired_stats(d_small, margin)
    check(fails, st["verdict"] == "EQUIVALENT",
          f"tight null should be EQUIVALENT, got {st['verdict']} ci90={st['ci90']}")

    # Effect far larger than the margin -> a real, non-negligible difference.
    d_big = rng.normal(0.02, 0.004, 400)
    st = paired_stats(d_big, margin)
    check(fails, st["verdict"] == "DIFFERS, NOT NEGLIGIBLE",
          f"large effect should DIFFER, got {st['verdict']}")

    # Small n, wide spread -> neither conclusion is available.
    d_noisy = rng.normal(0.001, 0.05, 8)
    st = paired_stats(d_noisy, margin)
    check(fails, st["verdict"] == "INCONCLUSIVE",
          f"underpowered data should be INCONCLUSIVE, got {st['verdict']}")
    check(fails, st["min_margin"] > margin,
          "INCONCLUSIVE must imply the achievable margin exceeds the one asked for")

    # min_margin must be honest: just above it, the verdict flips.
    st2 = paired_stats(d_noisy, st["min_margin"] * 1.001)
    check(fails, st2["verdict"] == "EQUIVALENT",
          f"at margin just above min_margin the verdict should flip to "
          f"EQUIVALENT, got {st2['verdict']}")


def test_pairing_gain(fails):
    """Correlated arms must show the pairing buying real variance reduction."""
    rng = np.random.default_rng(2)
    base = rng.normal(0.35, 0.08, 200)          # between-cell spread, the big term
    a = base + rng.normal(0, 0.004, 200)
    b = base + rng.normal(0.001, 0.004, 200)
    g = pairing_gain(a, b, b - a)
    check(fails, g["r"] > 0.9, f"arms should correlate strongly, r={g['r']:.3f}")
    check(fails, g["variance_gain"] > 5,
          f"pairing should tighten sd substantially, got {g['variance_gain']:.2f}x")
    check(fails, g["sd_paired"] < g["sd_unpaired"],
          "paired sd must be below the unpaired sd")


def test_detection_selection_effect(fails):
    """An asymmetric detection failure must be flagged, not absorbed."""
    cells = [f"id{i:02d}|p001" for i in range(40)]
    A = make_scores(cells, "a", id_loss_antelope=0.3).set_index("cell")
    B = make_scores(cells, "b", id_loss_antelope=0.3).set_index("cell")
    # Arm B loses 10 faces that arm A found: a one-sided selection effect.
    B.loc[cells[:10], "face_detected"] = False
    det = detection_check(A, B, pd.Index(cells), "a", "b")
    check(fails, det["only_a"] == 10 and det["only_b"] == 0,
          f"discordant counts wrong: {det}")
    check(fails, det["p_mcnemar"] < 0.05,
          f"a 10-0 split should be significant, p={det['p_mcnemar']}")

    # Symmetric losses are not a selection effect and must not be flagged.
    B2 = make_scores(cells, "b", id_loss_antelope=0.3).set_index("cell")
    A2 = make_scores(cells, "a", id_loss_antelope=0.3).set_index("cell")
    A2.loc[cells[:5], "face_detected"] = False
    B2.loc[cells[5:10], "face_detected"] = False
    det2 = detection_check(A2, B2, pd.Index(cells), "a", "b")
    check(fails, det2["p_mcnemar"] > 0.05,
          f"a 5-5 split should not be significant, p={det2['p_mcnemar']}")


def test_margin_parsing(fails):
    """A margin is only meaningful on its own metric's scale.

    One number applied to every metric is how 'EQUIVALENT' ends up printed
    beside a metric the margin means nothing on, so per-metric margins have to
    parse correctly and unlisted metrics have to fall through to no verdict.
    """
    check(fails, parse_margins(None) == {}, "no margin should parse to empty")
    check(fails, parse_margins("0.004") == {"*": 0.004},
          "a bare number should apply to every metric")

    m = parse_margins("id_loss_antelope=0.004, clipscore_b32=0.008")
    check(fails, m == {"id_loss_antelope": 0.004, "clipscore_b32": 0.008},
          f"per-metric parse wrong: {m}")
    check(fails, margin_for(m, "id_loss_antelope") == 0.004, "lookup wrong")
    check(fails, margin_for(m, "pickscore_paper") is None,
          "an unlisted metric must get no margin, not a borrowed one")
    # A bare number is a fallback for everything, including unlisted metrics.
    check(fails, margin_for(parse_margins("0.01"), "anything") == 0.01,
          "the '*' fallback should cover unlisted metrics")

    for bad in ("nonsense", "id_loss=abc", "id_loss"):
        try:
            parse_margins(bad)
            fails.append(f"--margin {bad!r} was accepted")
        except SystemExit:
            pass

    # No margin -> an effect estimate but explicitly no verdict.
    st = paired_stats(np.zeros(20) + 0.001, margin=None)
    check(fails, "n/a" in st["verdict"],
          f"a metric with no margin must not get a verdict, got {st['verdict']}")


def test_detection_bounds(fails):
    """A win built on dropping hard cells must not survive the bound.

    This is the pilot's actual failure mode reduced to a fixture. Note what the
    complete-case figure is NOT: it is already matched on the cells both arms
    survived, so it is a fair comparison *on that subset*. The question the
    bound asks is whether that subset generalises to the cells one arm lost --
    and when the lost cells are the hard ones, it does not.
    """
    cells = [f"id{i:02d}|p001" for i in range(21)]
    # 0-4 hard, 5-19 easy, 20 a catastrophic cell both arms survive. That last
    # one sets the worst observed value, which is what a miss gets charged.
    loss_a = [0.9] * 5 + [0.30] * 15 + [1.00]
    loss_b = [0.85] * 5 + [0.28] * 15 + [1.00]
    A = pd.DataFrame({"cell": cells, "id_loss_antelope": loss_a,
                      "face_detected": True}).set_index("cell")
    B = pd.DataFrame({"cell": cells, "id_loss_antelope": loss_b,
                      "face_detected": True}).set_index("cell")
    idx = pd.Index(cells)

    bd = detection_bounds(A, B, idx, "id_loss_antelope", "A", "B")
    check(fails, bd["diff_complete"] < 0, "B should win on complete data")
    check(fails, bd["sign_stable"],
          "with nothing missing the bound cannot flip")

    # B now fails to produce a face on the 5 hard cells.
    B2 = B.copy()
    B2.loc[cells[:5], "id_loss_antelope"] = np.nan
    B2.loc[cells[:5], "face_detected"] = False
    bd2 = detection_bounds(A, B2, idx, "id_loss_antelope", "A", "B")

    check(fails, bd2["n_missing_b"] == 5 and bd2["n_missing_a"] == 0,
          f"missing counts wrong: {bd2['n_missing_a']}/{bd2['n_missing_b']}")
    check(fails, bd2["n_complete"] == 16,
          f"complete-case n should be 16, got {bd2['n_complete']}")
    check(fails, bd2["diff_complete"] < 0,
          "complete case should still show B ahead -- that is the trap")
    check(fails, bd2["diff_imputed"] > 0,
          f"worst-case imputation should reverse it, got "
          f"{bd2['diff_imputed']:+.4f}")
    check(fails, not bd2["sign_stable"],
          "a ranking that flips between the bounds must be flagged unstable")
    check(fails, np.isclose(bd2["worst_observed"], 1.00),
          f"worst observed should be 1.00, got {bd2['worst_observed']}")


def test_guards(fails, tmp: Path):
    """Malformed inputs must fail loudly rather than pair the wrong things."""
    dup = pd.DataFrame({"cell": ["x|p1", "x|p1"], "method": "m",
                        "id_loss_antelope": [0.3, 0.4]})
    p = tmp / "dup.parquet"
    dup.to_parquet(p, index=False)
    try:
        load_scores(str(p), "dup")
        fails.append("duplicate cells were accepted; pairing would be arbitrary")
    except SystemExit:
        pass

    mixed = pd.DataFrame({"cell": ["x|p1", "y|p1"], "method": ["m1", "m2"],
                          "id_loss_antelope": [0.3, 0.4]})
    p2 = tmp / "mixed.parquet"
    mixed.to_parquet(p2, index=False)
    try:
        load_scores(str(p2), "mixed")
        fails.append("a two-method table was accepted; arms would be conflated")
    except SystemExit:
        pass


def test_end_to_end(fails, tmp: Path):
    """Run the actual CLI, including the NaN-drop and unpaired-cell paths."""
    rng = np.random.default_rng(3)
    cells = [f"id{i:02d}|p{j:03d}" for i in range(20) for j in range(10)]
    base = rng.normal(0.35, 0.08, len(cells))

    a = make_scores(cells, "infu_q8",
                    id_loss_antelope=base + rng.normal(0, 0.004, len(cells)),
                    id_loss_buffalo=base + rng.normal(0, 0.005, len(cells)),
                    clipscore_l14=rng.normal(0.25, 0.02, len(cells)),
                    clipscore_b32=rng.normal(0.30, 0.02, len(cells)),
                    pickscore_paper=rng.normal(0.22, 0.01, len(cells)))
    b = make_scores(cells, "infu_bf16",
                    id_loss_antelope=base + rng.normal(0.0005, 0.004, len(cells)),
                    id_loss_buffalo=base + rng.normal(0.0005, 0.005, len(cells)),
                    clipscore_l14=rng.normal(0.25, 0.02, len(cells)),
                    clipscore_b32=rng.normal(0.30, 0.02, len(cells)),
                    pickscore_paper=rng.normal(0.22, 0.01, len(cells)))

    # A few undetected faces (null ID Loss) and one cell missing from arm B.
    a.loc[:4, "id_loss_antelope"] = np.nan
    b = b.iloc[:-1]

    pa, pb = tmp / "arm_a.parquet", tmp / "arm_b.parquet"
    a.to_parquet(pa, index=False)
    b.to_parquet(pb, index=False)

    out_json = tmp / "report.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--a", str(pa), "--b", str(pb),
         "--label-a", "InfU 8-bit", "--label-b", "InfU bf16",
         "--margin", "0.004", "--out-json", str(out_json)],
        capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(f"CLI exited {r.returncode}:\n{r.stderr[-1500:]}")
        return
    out = r.stdout
    check(fails, "1 only in A" in out or "unpaired" in out,
          "the cell missing from arm B was not reported as unpaired")
    check(fails, "5 dropped" in out,
          "the 5 null ID Loss cells were not reported as dropped")
    check(fails, "EQUIVALENT" in out,
          "a 0.0005 effect against a 0.004 margin should be EQUIVALENT")
    check(fails, out_json.exists(), "--out-json wrote nothing")
    if out_json.exists():
        import json
        rep = json.loads(out_json.read_text())
        check(fails, rep["metrics"]["id_loss_antelope"]["n"] == len(cells) - 1 - 5,
              f"paired n wrong: {rep['metrics']['id_loss_antelope']['n']}")
    return out


def main():
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_against_scipy(fails)
        test_verdicts(fails)
        test_pairing_gain(fails)
        test_detection_selection_effect(fails)
        test_margin_parsing(fails)
        test_detection_bounds(fails)
        test_guards(fails, tmp)
        sample = test_end_to_end(fails, tmp)

    if fails:
        print("PAIRED-COMPARISON TESTS FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("paired-comparison tests passed (scipy agreement, all three "
          "verdicts, min_margin,")
    print(" pairing gain, McNemar selection check, per-metric margins,")
    print(" detection bounds, malformed-input guards, CLI end to end)")
    if sample:
        print("\n--- sample CLI output ---")
        print(sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())

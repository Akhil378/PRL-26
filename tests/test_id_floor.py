#!/usr/bin/env python
"""Verify the ID-floor filters preserve geometry, without trusting a detector.

The floor measured by `tools/measure_id_floor.py` is only a floor if the filters
change appearance WITHOUT moving anything. That claim is the whole control, so
it needs a test that can actually fail.

The obvious check -- run the face detector before and after and compare boxes --
is the one that must NOT be used, and the tool learned this the hard way: the
detector's box moved by up to 44 px on filtered references, which looked like a
geometry failure and was not. A detected box is an ESTIMATE produced from
appearance; change the appearance and the estimate moves even though every pixel
stayed put. Testing geometry with an appearance-sensitive instrument cannot
distinguish the two.

So this measures the filters directly, against fiducial marks whose true
positions are known by construction. If a filter warped, resampled or shifted
the image, the centroids move and the test fails. If it merely repainted, they
do not.

Usage:  python tests/test_id_floor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.measure_id_floor import filter_ladder  # noqa: E402


def fiducial_image(w=512, h=640):
    """Dark field with bright discs at known centres.

    Discs are large and high-contrast so that even the most aggressive filter
    leaves something to locate; the test is about WHERE the mass is, not whether
    the filter dimmed it.
    """
    import cv2
    img = np.full((h, w, 3), 20, np.uint8)
    centres = [(96, 120), (416, 120), (256, 320), (96, 520), (416, 520)]
    for (x, y) in centres:
        cv2.circle(img, (x, y), 40, (240, 240, 240), -1)
    return img, centres


def centroids(img, n_expected):
    """Locate bright blobs and return their centres, sorted for comparison."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Midpoint threshold: robust to a filter changing overall brightness.
    thr = (int(gray.min()) + int(gray.max())) // 2
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs = [(cents[i][0], cents[i][1])
             for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 300]
    if len(blobs) != n_expected:
        return None
    return sorted(blobs, key=lambda c: (round(c[1] / 50), round(c[0] / 50)))


def textured_image(w=512, h=640, seed=0):
    """Rich, filter-survivable structure for phase correlation.

    The disc fixture cannot speak for a filter that dissolves filled shapes,
    such as pencilSketch. Phase correlation does not care what the image looks
    like, only how its structure lines up, so it covers every filter including
    the ones that change appearance most.
    """
    import cv2
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    for i in range(6):
        x, y = int(rng.integers(40, w - 120)), int(rng.integers(40, h - 120))
        cv2.rectangle(img, (x, y), (x + 80, y + 60),
                      (int(rng.integers(0, 255)),) * 3, -1)
    return img


def global_shift(a, b):
    """Sub-pixel translation between two images, appearance-independent."""
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    (dx, dy), _resp = cv2.phaseCorrelate(ga, gb)
    return max(abs(dx), abs(dy))


def check(fails, cond, msg):
    if not cond:
        fails.append(msg)


def main():
    fails: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("opencv not available here; run this in the metrics env")
        return 0

    src, centres = fiducial_image()
    ref = centroids(src, len(centres))
    check(fails, ref is not None, "could not locate fiducials in the source")
    if ref is None:
        print("FAILED: fiducial setup broken")
        return 1

    # The centroids must match where the discs were actually drawn.
    drawn = sorted(centres, key=lambda c: (round(c[1] / 50), round(c[0] / 50)))
    for (gx, gy), (dx, dy) in zip(ref, drawn):
        check(fails, abs(gx - dx) < 1.5 and abs(gy - dy) < 1.5,
              f"fiducial setup: found ({gx:.1f},{gy:.1f}) want ({dx},{dy})")

    tex = textured_image()
    print(f"{'filter':<16}{'shape kept':>12}{'fiducial px':>20}"
          f"{'phase px':>16}")
    for name, fn in filter_ladder():
        out = fn(src.copy())
        same_shape = out.shape == src.shape
        check(fails, same_shape,
              f"{name}: shape changed {src.shape} -> {out.shape} (resampling!)")

        got = centroids(out, len(centres))
        if got is None:
            # A filter that dissolves filled shapes (pencilSketch) defeats this
            # fixture. That is a fixture limit, not a geometry failure, and the
            # phase-correlation check below covers it.
            shown = "n/a (dissolved)"
        else:
            shift = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                        for a, b in zip(ref, got))
            shown = "%.2f" % shift
            # One pixel of slack for thresholding, not for warping. A genuine
            # warp or resample moves a centroid by far more than this.
            check(fails, shift < 1.0,
                  f"{name}: centroid moved {shift:.2f} px -- not appearance-only")

        # Appearance-independent check, applied to EVERY filter.
        tshift = global_shift(tex, fn(tex.copy()))
        print(f"{name:<16}{str(same_shape):>12}{shown:>20}{tshift:>16.3f}")
        check(fails, tshift < 0.5,
              f"{name}: phase correlation shows a {tshift:.3f} px translation")

    if fails:
        print("\nID-FLOOR GEOMETRY TESTS FAILED:")
        for f in fails:
            print("  -", f)
        print("\nA filter that moves geometry cannot supply a domain-shift floor:")
        print("its ID Loss would mix appearance with misalignment.")
        return 1

    print("\ngeometry preserved by every filter, verified two ways and by")
    print("neither a detector nor any appearance-sensitive instrument:")
    print("fiducial centroids where the marks survive, and phase correlation")
    print("everywhere. The ID Loss these filters produce is therefore")
    print("attributable to appearance alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

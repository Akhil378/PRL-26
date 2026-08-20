#!/usr/bin/env python
"""Unit tests for the FMI crop geometry. No GPU, no models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.face_masking import (area, background_crop, dilate, overlaps,
                                      plan_crops)

W, H = 864, 1152
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


print("dilate")
b = (400, 300, 500, 450)
d = dilate(b, 1.2, W, H)
check("grows about the centre", d == (390, 285, 510, 465), str(d))
check("centre is preserved",
      (d[0] + d[2]) / 2 == (b[0] + b[2]) / 2 and (d[1] + d[3]) / 2 == (b[1] + b[3]) / 2)
edge = dilate((0, 0, 100, 100), 2.0, W, H)
check("clips at the image edge", edge[0] == 0 and edge[1] == 0, str(edge))
check("clips at the far edge", dilate((W - 50, H - 50, W, H), 3.0, W, H)[2:] == (W, H))

print("\noverlaps")
check("touching edges do not overlap", not overlaps((0, 0, 10, 10), (10, 0, 20, 10)))
check("one pixel of shared area overlaps", overlaps((0, 0, 10, 10), (9, 9, 20, 20)))
check("containment overlaps", overlaps((0, 0, 100, 100), (10, 10, 20, 20)))

print("\nbackground_crop")
face = (350, 200, 500, 400)
bg = background_crop(face, W, H)
check("returns a box", bg is not None)
check("does not touch the face", bg is not None and not overlaps(bg, face))
check("stays inside the image",
      bg is not None and bg[0] >= 0 and bg[1] >= 0 and bg[2] <= W and bg[3] <= H, str(bg))
check("matches the face area at full scale",
      bg is not None and area(bg) == area(face), f"{area(bg)} vs {area(face)}")

# a face filling almost the whole frame leaves nowhere to sample
big = (5, 5, W - 5, H - 5)
check("gives up when the face fills the frame", background_crop(big, W, H) is None)

# a tall narrow face still leaves room beside it
tall = (400, 0, 500, H)
bt = background_crop(tall, W, H)
check("finds a region beside a full-height face",
      bt is not None and not overlaps(bt, tall), str(bt))

print("\nplan_crops")
p = plan_crops((350, 200, 500, 400), (W, H))
check("normal portrait yields both crops",
      p.background is not None and p.reason == "ok")
p2 = plan_crops((10, 10, W - 10, H - 10), (W, H))
check("close-up is reported, not imputed",
      p2.background is None and p2.reason == "face_dominates_frame", p2.reason)
p3 = plan_crops((100, 100, 100, 100), (W, H))
check("degenerate box is caught",
      p3.background is None and p3.reason == "degenerate_face_box", p3.reason)

# determinism: the metric must not move between scoring runs
runs = {background_crop((350, 200, 500, 400), W, H) for _ in range(5)}
print("\ndeterminism")
check("identical across repeated calls", len(runs) == 1)

print()
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    sys.exit(1)
print("all crop-geometry tests passed")

#!/usr/bin/env python
"""Check the curated identity set before it is frozen into a manifest.

Enforces what GENERATION.md requires: licence recorded, exactly one detectable
face, adequate face resolution, and no source drawn from InfU's stage-1 training
corpora. Marks passing rows "ready" with --mark-ready, which is what unlocks
src/manifest.py.

Runs without insightface, in which case the face checks are skipped and the row
is not marked ready.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "benchmark" / "identities.json"

# Datasets named in InfiniteYou section 4.1 as stage-1 pretraining data.
FORBIDDEN = ["ffhq", "celeba", "celebv", "celebvhq", "celebv-hq", "celebv-text",
             "vggface", "vggface2", "millioncelebs", "vfhq", "easyportrait",
             "cosmicman", "cosmicmanhq"]

# ArcFace consumes a 112x112 aligned crop, so the only real requirement is that
# the detected face is comfortably above that before downsampling. 256 px is 2x
# the model input in each dimension; the earlier 512 was an invented number that
# rejected 1350x1350 studio portraits whose faces occupy ~35% of the frame.
MIN_FACE_PX = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mark-ready", action="store_true")
    ap.add_argument("--insightface-root", default=None)
    args = ap.parse_args()

    rows = json.load(open(IDS, encoding="utf-8"))
    app = None
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=args.insightface_root,
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:                     # noqa: BLE001
        print(f"! insightface unavailable ({type(e).__name__}) - face checks skipped\n")

    ok_count = 0
    for r in rows:
        problems = []
        p = ROOT / r["path"]

        if not p.exists():
            problems.append("image missing")
        if str(r.get("source", "")).upper() in ("", "TODO"):
            problems.append("source not recorded")
        if str(r.get("licence", "")).upper() in ("", "TODO"):
            problems.append("licence not recorded")

        src = str(r.get("source", "")).lower()
        for bad in FORBIDDEN:
            if bad in src.replace("-", "").replace("_", ""):
                problems.append(f"source is InfU training data ({bad})")
                break

        if p.exists() and app is not None:
            import cv2
            img = cv2.imread(str(p))
            if img is None:
                problems.append("unreadable image")
            else:
                faces = app.get(img)
                if not faces:
                    problems.append("no face detected")
                elif len(faces) > 1:
                    problems.append(f"{len(faces)} faces detected, need exactly 1")
                else:
                    b = faces[0].bbox
                    side = min(b[2] - b[0], b[3] - b[1])
                    if side < MIN_FACE_PX:
                        problems.append(f"face crop {side:.0f}px < {MIN_FACE_PX}px")

        if problems:
            print(f"  FAIL  {r['iid']}  " + "; ".join(problems))
        else:
            ok_count += 1
            group = r.get("ethnicity", r.get("skin_tone", "?"))
            print(f"  OK    {r['iid']}  {r['gender']:<6} {r['age_band']:<6} "
                  f"{group:<12} {r.get('licence','')}")
            if args.mark_ready and app is not None:
                r["status"] = "ready"

    print(f"\n{ok_count}/{len(rows)} identities pass")
    for f in ("gender", "age_band", "ethnicity"):
        if any(f in r for r in rows):
            print(f"  {f:<10}", dict(Counter(r.get(f, "?") for r in rows)))
    n2 = sum(1 for r in rows if r.get("part2"))
    print(f"  part2 subset: {n2} "
          f"({dict(Counter(r['gender'] for r in rows if r.get('part2')))})")

    if args.mark_ready:
        if app is None:
            print("\nNot marking ready: face checks did not run.")
            return 1
        IDS.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"\nmarked {sum(1 for r in rows if r.get('status')=='ready')} ready")
    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

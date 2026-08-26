#!/usr/bin/env python
"""Measure the ArcFace domain-shift floor with geometry-preserving filters.

REPLACES the img2img calibration control, which failed (CONTEXT 6.9): FLUX
img2img never applied the style, and its "no style" arm still destroyed more
identity than the actual generations it was meant to calibrate. The design error
was assuming a generative pass preserves identity while changing appearance. It
re-synthesises geometry, and geometry is most of what ArcFace measures.

WHAT THIS MEASURES INSTEAD
--------------------------
A non-photorealistic *filter* repaints texture without moving a pixel. Landmark
geometry is therefore identical by construction -- this script verifies that
rather than assuming it -- so any drop in ArcFace similarity is attributable to
appearance alone. That is exactly the floor Part 2 needs: how much identity
similarity is lost purely because the image stopped looking like a photograph.

AND IT IS PARAMETERISED BY THE RIGHT AXIS
-----------------------------------------
The failed design tried to match FLUX's named styles one-for-one, which no
filter can do. This sweeps the quantity that actually matters instead: the
**loss of photorealism**, measured as CLIP similarity to "a natural colour
photograph" -- the identical phrase `configs/metrics.yaml` already uses as FMI's
primary axis.

That makes the floor transferable rather than style-matched. The output is a
curve

    ID Loss floor  =  f(photorealism drop)

so a Part 2 generation is calibrated by measuring how far its own photorealism
fell and reading the floor off at that point, whatever style produced it. A
style-name lookup could never have done this, because the filter's "oil
painting" is not FLUX's.

HONEST LIMITATION, WHICH BELONGS IN THE REPORT
----------------------------------------------
These filters abstract texture while leaving the underlying photograph's shading
and structure intact. A diffusion model asked for an oil painting also
reinterprets shape, lighting and material. So at equal photorealism drop, a
filter is the *gentler* transformation and this curve is a **lower bound** on
the true floor. A Part 2 effect that clears it is real; one that falls below it
is not demonstrated. `configs/metrics.yaml` already carries the same
lower-bound caveat for FMI's synthetic controls.

Usage (metrics env; CPU only, no GPU needed):
  python tools/measure_id_floor.py --out $WORK/prl26/results/scores/id_floor.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHOTO_PHRASE = "a natural colour photograph"      # same as configs/metrics.yaml


def filter_ladder():
    """Increasingly non-photographic renderings, all geometry-preserving.

    Every one of these is a per-pixel or edge-aware neighbourhood operation.
    None warps, resamples or re-synthesises, so a landmark at (x, y) stays at
    (x, y). That is the property the whole control rests on, and it is checked
    at runtime below.
    """
    import cv2

    def ident(a):
        return a

    def edge_preserve(a):
        return cv2.edgePreservingFilter(a, flags=1, sigma_s=60, sigma_r=0.4)

    def sty(sr):
        def f(a):
            return cv2.stylization(a, sigma_s=60, sigma_r=sr)
        return f

    def pencil_colour(a):
        _gray, colour = cv2.pencilSketch(a, sigma_s=60, sigma_r=0.07,
                                         shade_factor=0.05)
        return colour

    return [
        ("none", ident),                 # zero point: the reference itself
        ("edge_preserve", edge_preserve),
        ("stylize_25", sty(0.25)),
        ("stylize_45", sty(0.45)),       # the setting used by the FMI controls
        ("stylize_60", sty(0.60)),
        ("pencil_colour", pencil_colour),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default="benchmark/identities.json")
    ap.add_argument("--out", required=True, help="Parquet for the floor curve.")
    ap.add_argument("--save-images", default=None,
                    help="Optional directory; writes one PNG per cell for the "
                         "report's qualitative figure.")
    ap.add_argument("--id-root", default=None, help="INSIGHTFACE_HOME")
    ap.add_argument("--backbone", default="B32")
    ap.add_argument("--all-identities", action="store_true",
                    help="Use all identities, not just the part2 subset.")
    a = ap.parse_args()

    import cv2
    from src.metrics.clip_backend import CLIPBackend
    from src.metrics.id_loss import IDScorer

    ids = json.loads((ROOT / a.identities).read_text())
    recs = ids["identities"] if isinstance(ids, dict) and "identities" in ids else ids
    pool = [r for r in recs
            if r.get("status") == "ready" and (a.all_identities or r.get("part2"))]
    if not pool:
        raise SystemExit("no ready identities")

    print(f"loading models (clip={a.backbone}, insightface)...", flush=True)
    clip = CLIPBackend(a.backbone)
    scorers = {p: IDScorer(pack=p, root=a.id_root)
               for p in ("antelopev2", "buffalo_l")}
    ladder = filter_ladder()

    save_dir = Path(a.save_images) if a.save_images else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows, geom_fail = [], []
    for r in pool:
        ref_path = ROOT / r["path"]
        bgr = cv2.imread(str(ref_path))
        if bgr is None:
            raise SystemExit(f"cannot read {ref_path}")

        # Reference geometry, to verify the filters do not move it.
        _, ref_box = scorers["antelopev2"].embed_with_box(bgr)

        for level, (name, fn) in enumerate(ladder):
            out_bgr = fn(bgr.copy())
            tmp = (save_dir / f"{r['iid']}__floor_{level}_{name}.png"
                   if save_dir else Path("/tmp") / f"_floor_{r['iid']}_{name}.png")
            cv2.imwrite(str(tmp), out_bgr)

            loss_a, det = scorers["antelopev2"].id_loss(str(tmp), str(ref_path))
            loss_b, _ = scorers["buffalo_l"].id_loss(str(tmp), str(ref_path))
            img = Image.open(tmp).convert("RGB")
            photo = clip.clipscore(img, PHOTO_PHRASE)

            # GEOMETRY CHECK -- the claim the control depends on. A filter must
            # leave the detected face box where it was; if it does not, the
            # operation is not appearance-only and its ID Loss is contaminated.
            _, box = scorers["antelopev2"].embed_with_box(out_bgr)
            if ref_box is not None and box is not None:
                shift = float(np.abs(np.asarray(box, float)
                                     - np.asarray(ref_box, float)).max())
            else:
                shift = float("nan")
            if shift == shift and shift > 8.0:
                geom_fail.append((r["iid"], name, shift))

            rows.append({"iid": r["iid"], "level": level, "filter": name,
                         "id_loss_antelope": loss_a if det else None,
                         "id_loss_buffalo": loss_b if det else None,
                         "face_detected": bool(det),
                         "clip_photo": photo, "box_shift_px": shift})
            if not save_dir:
                tmp.unlink(missing_ok=True)
        print(f"  {r['iid']} done", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # The photorealism drop is measured against each identity's OWN unfiltered
    # score, so a face that simply photographs unusually does not shift the axis.
    # df["filter"], never df.filter -- the latter is DataFrame.filter, a method,
    # and the comparison would silently produce nonsense rather than an error.
    base = df[df["filter"] == "none"].set_index("iid")["clip_photo"]
    df["photo_drop"] = df.apply(
        lambda x: base[x["iid"]] - x["clip_photo"], axis=1)

    g = df.groupby(["level", "filter"]).agg(
        n=("iid", "size"),
        detected=("face_detected", "mean"),
        photo_drop=("photo_drop", "mean"),
        id_floor=("id_loss_antelope", "mean"),
        id_floor_sd=("id_loss_antelope", "std"),
        id_floor_holdout=("id_loss_buffalo", "mean"),
        max_shift=("box_shift_px", "max"),
    ).reset_index()

    print(f"\nwrote {out}  ({len(df)} rows, {df.iid.nunique()} identities)")
    print("\n--- ArcFace domain-shift floor, geometry held fixed ---")
    print(f"{'filter':<16}{'photo drop':>12}{'ID floor':>10}{'sd':>8}"
          f"{'holdout':>9}{'det':>6}{'max px shift':>14}")
    for _, x in g.iterrows():
        print(f"{x['filter']:<16}{x['photo_drop']:>12.4f}{x['id_floor']:>10.4f}"
              f"{x['id_floor_sd']:>8.4f}{x['id_floor_holdout']:>9.4f}"
              f"{x['detected']:>6.2f}{x['max_shift']:>14.1f}")

    if geom_fail:
        print("\n*** GEOMETRY CHECK FAILED -- these filters moved the face box:")
        for iid, name, sh in geom_fail:
            print(f"      {iid} {name}: {sh:.1f} px")
        print("    Their ID Loss is not appearance-only and must not be used")
        print("    as a floor. Drop those levels or replace the filter.")
        return 1

    print("\nGeometry check passed: every filter left the detected face box "
          "within 8 px,")
    print("so these ID Loss values are attributable to appearance alone.")
    print("\nREAD THIS AS A LOWER BOUND. A filter abstracts texture but keeps the")
    print("photograph's shading and structure; a diffusion model asked for a")
    print("painting also reinterprets shape and lighting. At equal photorealism")
    print("drop the filter is the gentler transformation, so a Part 2 effect that")
    print("clears this curve is real, and one that falls below it is not shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

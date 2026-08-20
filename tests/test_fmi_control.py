#!/usr/bin/env python
"""Validate the Face Masking Index against constructed positive/negative controls.

configs/metrics.yaml requires this before FMI is allowed to support any claim.
Three composites are built from one real portrait, all scored against the same
style phrase:

  masked     photoreal face on a stylised background   -> FMI should be clearly positive
  blended    stylised face on a stylised background    -> FMI should be near zero
  photoreal  photoreal face on a photoreal background  -> FMI should be near zero

If masked does not separate from the other two, FMI is not measuring what it
claims to and must not be reported.

Usage:  python tests/test_fmi_control.py [--backbone B32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gen_stub import canvas, stylise                  # noqa: E402
from src.metrics.clip_backend import CLIPBackend          # noqa: E402
from src.metrics.face_masking import face_masking_index   # noqa: E402
from src.metrics.id_stub import StubIDScorer              # noqa: E402

W, H = 864, 1152
STYLE_PHRASE = "an oil painting"


def compose(ref: Image.Image, style_bg: bool, style_face: bool) -> Image.Image:
    bg = canvas(W, H, seed=7)
    if style_bg:
        bg = stylise(bg)
    side = int(min(W, H) * 0.42)
    face = ref.convert("RGB").resize((side, side), Image.LANCZOS)
    if style_face:
        face = stylise(face)
    bg.paste(face, ((W - side) // 2, int(H * 0.10)))
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="B32")
    a = ap.parse_args()

    ref_path = ROOT / "tests" / "fixtures" / "identities" / "id_t01.jpg"
    if not ref_path.exists():
        print("run tools/make_fixtures.py first")
        return 1
    ref = Image.open(ref_path)

    clip = CLIPBackend(a.backbone, device="cpu")
    det = StubIDScorer()

    import cv2
    import numpy as np

    cases = {
        "masked":    compose(ref, style_bg=True,  style_face=False),
        "blended":   compose(ref, style_bg=True,  style_face=True),
        "photoreal": compose(ref, style_bg=False, style_face=False),
    }

    outdir = ROOT / "tests" / "fixtures" / "out" / "fmi_control"
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    print(f"style phrase: {STYLE_PHRASE!r}   backbone: {a.backbone}\n")
    print(f"  {'case':<11}{'FMI_photo':>11}{'FMI_style':>11}{'ph_face':>10}{'ph_bg':>9}  reason")
    for name, img in cases.items():
        img.save(outdir / f"{name}.png")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, box = det.embed_with_box(bgr)
        if box is None:
            print(f"  {name:<11}{'--':>9}   no face detected in composite")
            results[name] = None
            continue
        m = face_masking_index(img, STYLE_PHRASE, box, clip)
        results[name] = m
        f = lambda v, w=11: f"{v:{w}.4f}" if v is not None else f"{'--':>{w}}"
        print(f"  {name:<11}{f(m['fmi_photo'])}{f(m['fmi_style'])}"
              f"{f(m['photo_face'],10)}{f(m['photo_bg'],9)}  {m['fmi_reason']}")

    print()
    fails = []
    need = [k for k in ("masked", "blended", "photoreal")
            if results.get(k) is None or results[k].get("fmi_photo") is None]
    if need:
        fails.append(f"no FMI produced for: {', '.join(need)}")
    else:
        mk, bl, ph = (results[k]["fmi_photo"] for k in ("masked", "blended", "photoreal"))
        # Absolute FMI is crop-content biased, so the photoreal condition is the
        # baseline and every comparison is a difference against it.
        did_masked, did_blended = mk - ph, bl - ph
        print("  difference vs photoreal control (the form that must be reported):")
        print(f"    masked  - photoreal = {did_masked:+.4f}   should be clearly positive")
        print(f"    blended - photoreal = {did_blended:+.4f}   should not be positive")
        print(f"    separation          = {did_masked - did_blended:+.4f}")
        if did_masked <= 0:
            fails.append(f"masked does not exceed the photoreal control ({did_masked:+.4f})")
        if did_masked <= did_blended:
            fails.append(f"masked ({did_masked:+.4f}) does not exceed blended ({did_blended:+.4f})")

    if fails:
        print("\nFMI CONTROL FAILED:")
        for f in fails:
            print("  -", f)
        print("\nFMI must not be used to support a claim until this passes.")
        return 1
    print("\nFMI control passed: the index separates a masked face from a blended one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

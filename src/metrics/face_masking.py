#!/usr/bin/env python
"""Face Masking Index.

Tests the hypothesis that identity preservation under a style prompt is bought
by leaving the face region photoreal while the background stylises around it.
A whole-image CLIPScore cannot see this: "stylised everywhere" and "stylised
except the face" score alike. FMI separates them.

    FMI = style_score(background crop) - style_score(face crop)

scored against the bare style phrase ("a 19th-century oil painting"). Positive
means the background adopted the style more strongly than the face did.

The crop geometry lives here and is deliberately free of any model dependency,
so it can be unit-tested without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class CropResult:
    face: Box
    background: Optional[Box]
    reason: str = "ok"


def dilate(box: Box, factor: float, w: int, h: int) -> Box:
    """Scale a box about its centre, clipped to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    return (
        max(0, int(round(cx - bw / 2))),
        max(0, int(round(cy - bh / 2))),
        min(w, int(round(cx + bw / 2))),
        min(h, int(round(cy + bh / 2))),
    )


def overlaps(a: Box, b: Box) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def background_crop(face: Box, w: int, h: int, stride_div: int = 16,
                    shrink: Sequence[float] = (1.0, 0.8, 0.6)) -> Optional[Box]:
    """Largest-distance face-free box of the same size as `face`.

    Candidates are placed on a regular grid; any that touches the face box is
    rejected, and of the survivors the one whose centre is farthest from the
    face centre wins. Deterministic, and it degrades by shrinking the box rather
    than by silently returning something that overlaps the subject.

    Returns None when the face dominates the frame -- common for close-up
    prompts, and a case that must be reported rather than imputed.
    """
    fw, fh = face[2] - face[0], face[3] - face[1]
    if fw <= 0 or fh <= 0:
        return None
    fcx, fcy = (face[0] + face[2]) / 2.0, (face[1] + face[3]) / 2.0
    sx, sy = max(1, w // stride_div), max(1, h // stride_div)

    for s in shrink:
        cw, ch = max(1, int(fw * s)), max(1, int(fh * s))
        if cw > w or ch > h:
            continue
        best, best_d = None, -1.0
        for y in range(0, h - ch + 1, sy):
            for x in range(0, w - cw + 1, sx):
                cand = (x, y, x + cw, y + ch)
                if overlaps(cand, face):
                    continue
                d = ((x + cw / 2) - fcx) ** 2 + ((y + ch / 2) - fcy) ** 2
                if d > best_d:
                    best, best_d = cand, d
        if best is not None:
            return best
    return None


def plan_crops(face_bbox: Box, image_size: Tuple[int, int],
               dilate_factor: float = 1.2) -> CropResult:
    """Face and background boxes for one image."""
    w, h = image_size
    face = dilate(face_bbox, dilate_factor, w, h)
    if area(face) == 0:
        return CropResult(face, None, "degenerate_face_box")
    if area(face) / float(w * h) > 0.6:
        return CropResult(face, None, "face_dominates_frame")
    bg = background_crop(face, w, h)
    return CropResult(face, bg, "ok" if bg else "no_background_region")


PHOTO_PHRASE = "a natural colour photograph"


def face_masking_index(image, style_phrase: str, face_bbox: Box, backend,
                       dilate_factor: float = 1.2,
                       photo_phrase: str = PHOTO_PHRASE) -> dict:
    """FMI for one image. `backend` supplies .clip_similarity(PIL.Image, str).

    Two formulations are returned, because local controls showed the obvious one
    is the weaker of the two:

      fmi_style = style(background) - style(face)
      fmi_photo = photo(face) - photo(background)

    Both are positive when the face resisted stylisation. `fmi_photo` is the
    more sensitive of the two: an NPR-stylised image moved CLIP's "a natural
    colour photograph" score by -0.0163 while moving "an oil painting" by only
    +0.0015, roughly a tenfold difference in signal. Losing photorealism is
    easier for CLIP to see than acquiring a specific painterly style.

    IMPORTANT -- neither number is interpretable in absolute terms. A face crop
    and a background crop have different content, and that alone shifts the
    style score: with no stylisation anywhere, a face crop scored 0.2233 against
    "an oil painting" and a background crop 0.2148, an 0.0085 offset comparable
    in size to the effects being measured. Always report the paired difference
    against the photoreal control condition for the same identity, base prompt
    and seed; that is what cancels the crop-content bias, and it is the reason
    the photoreal condition is in the design.

    Returns a dict; the fmi_* entries are None when no valid background exists.
    """
    plan = plan_crops(face_bbox, image.size, dilate_factor)
    face_img = image.crop(plan.face)
    out = {
        "style_face": backend.clip_similarity(face_img, style_phrase),
        "photo_face": backend.clip_similarity(face_img, photo_phrase),
        "style_bg": None, "photo_bg": None,
        "fmi_style": None, "fmi_photo": None,
        "fmi_reason": plan.reason,
    }
    if plan.background is None:
        return out
    bg_img = image.crop(plan.background)
    out["style_bg"] = backend.clip_similarity(bg_img, style_phrase)
    out["photo_bg"] = backend.clip_similarity(bg_img, photo_phrase)
    out["fmi_style"] = out["style_bg"] - out["style_face"]
    out["fmi_photo"] = out["photo_face"] - out["photo_bg"]
    out["fmi_reason"] = "ok"
    return out

#!/usr/bin/env python
"""ArcFace identity loss, scored with two recognisers.

    ID Loss = 1 - cos(ArcFace(generated), ArcFace(reference))

Both InfU and PuLID-FLUX condition on a 512-d ArcFace embedding from InsightFace
`antelopev2`. Scoring with that same network measures each method against the
encoder it was optimised for, which flatters both. `antelopev2` is kept because
it is what makes the numbers comparable to Table 1, and `buffalo_l` is carried
alongside as a held-out check. Any claim that survives only under antelopev2 is
a claim about the encoder, not about identity.

When no face is detected in a generated image the loss is undefined. It is
returned as None and the caller records the failure; the detection rate is a
first-class result, not a row to drop quietly. Under heavy stylisation it is one
of the more informative numbers in Part 2.
"""
from __future__ import annotations

import functools
from typing import Optional, Tuple

import cv2
import numpy as np

PACKS = ("antelopev2", "buffalo_l")


class IDScorer:
    def __init__(self, pack: str = "antelopev2", root: Optional[str] = None,
                 det_size: int = 640, ctx_id: int = 0):
        from insightface.app import FaceAnalysis
        self.pack = pack
        self.app = FaceAnalysis(
            name=pack, root=root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def _largest(self, bgr: np.ndarray):
        faces = self.app.get(bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        f = self._largest(bgr)
        return None if f is None else f.normed_embedding

    def embed_with_box(self, bgr: np.ndarray):
        """Embedding plus integer bbox -- FMI needs the box from the same detection."""
        f = self._largest(bgr)
        if f is None:
            return None, None
        x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
        return f.normed_embedding, (x1, y1, x2, y2)

    @functools.lru_cache(maxsize=64)
    def _ref_embedding(self, ref_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(ref_path)
        if img is None:
            raise FileNotFoundError(ref_path)
        e = self.embed(img)
        if e is None:
            raise ValueError(f"no face detected in reference image {ref_path}")
        return e

    def id_loss(self, gen_path: str, ref_path: str) -> Tuple[Optional[float], bool]:
        """Returns (id_loss, face_detected). Loss is None when detection fails."""
        gen = cv2.imread(gen_path)
        if gen is None:
            raise FileNotFoundError(gen_path)
        g = self.embed(gen)
        if g is None:
            return None, False
        return float(1.0 - np.dot(g, self._ref_embedding(ref_path))), True

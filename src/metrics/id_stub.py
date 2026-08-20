#!/usr/bin/env python
"""Local identity backend for pipeline testing. Never use for reported results.

insightface does not build on Windows without MSVC build tools, so the real
IDScorer cannot run on the development laptop. This provides the same interface
using OpenCV's bundled YuNet detector and SFace recogniser, which are genuine
models -- real detection, real 128-d embeddings, real cosine similarity -- just
not the ArcFace variants the paper and both methods use.

That makes it a proper test of our plumbing: the undefined-loss branch, the
two-recogniser columns, the bbox handover to FMI, the aggregation. It is not a
substitute for antelopev2 and must never appear in the report.

Weights (~40 MB total) download once from the OpenCV model zoo.
"""
from __future__ import annotations

import functools
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
DETECTOR = ("face_detection_yunet_2023mar.onnx",
            f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx")
RECOGNISER = ("face_recognition_sface_2021dec.onnx",
              f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx")


def _ensure(name: str, url: str) -> str:
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS / name
    if not dest.exists():
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    return str(dest)


class StubIDScorer:
    backend = "opencv-yunet-sface"

    def __init__(self, pack: str = "stub", root: Optional[str] = None, **kw):
        self.pack = pack
        self.det = cv2.FaceDetectorYN.create(
            _ensure(*DETECTOR), "", (320, 320), score_threshold=0.7)
        self.rec = cv2.FaceRecognizerSF.create(_ensure(*RECOGNISER), "")
        # the two "recognisers" must not be bit-identical, mirroring the way
        # antelopev2 and buffalo_l disagree slightly on the same image
        self._blur = pack not in ("stub", "antelopev2")

    def _detect_raw(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _feature(self, bgr: np.ndarray, face_row) -> Optional[np.ndarray]:
        src = cv2.GaussianBlur(bgr, (3, 3), 0) if self._blur else bgr
        aligned = self.rec.alignCrop(src, face_row)
        f = self.rec.feature(aligned).flatten().astype(np.float64)
        n = np.linalg.norm(f)
        return None if n == 0 else f / n

    def embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        row = self._detect_raw(bgr)
        return None if row is None else self._feature(bgr, row)

    def embed_with_box(self, bgr: np.ndarray):
        row = self._detect_raw(bgr)
        if row is None:
            return None, None
        x, y, w, h = (int(round(v)) for v in row[:4])
        return self._feature(bgr, row), (x, y, x + w, y + h)

    @functools.lru_cache(maxsize=64)
    def _ref_embedding(self, ref_path: str):
        img = cv2.imread(ref_path)
        if img is None:
            raise FileNotFoundError(ref_path)
        e = self.embed(img)
        if e is None:
            raise ValueError(f"no face detected in reference image {ref_path}")
        return e

    def id_loss(self, gen_path: str, ref_path: str) -> Tuple[Optional[float], bool]:
        gen = cv2.imread(gen_path)
        if gen is None:
            raise FileNotFoundError(gen_path)
        g = self.embed(gen)
        if g is None:
            return None, False
        return float(1.0 - np.dot(g, self._ref_embedding(ref_path))), True

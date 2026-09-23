#!/usr/bin/env python
"""ID Loss measured with the encoder InfU itself conditions on.

InfU does NOT condition on antelopev2's recogniser. Its pipeline uses antelopev2
only to find the face and its five landmarks, then embeds a 112x112 crop with
facexlib's IR-SE50 ArcFace (`init_recognition_model('arcface')`,
recognition_arcface_ir_se50.pth). PuLID-FLUX does condition on antelopev2's
glintr100. Part 1's primary ID Loss is glintr100, so it measures PuLID against
its own encoder and InfU against a foreign one. buffalo_l is foreign to both.
This adds the missing third column: each method measured against InfU's encoder.

The embedding path is InfU's, copied from pipelines/pipeline_infu_flux.py
(extract_arcface_bgr_embedding): norm_crop of the BGR image at the detector's
five keypoints, scaled to [-1, 1], channels left in BGR order exactly as InfU
leaves them. Only the detection rule differs, deliberately: generated images use
det_size 640 and the largest face, the same rule as src/metrics/id_loss.py, so
the set of detected cells is identical to Part 1's and the new column differs
from the old ones in the recognition network alone. References use InfU's
640/320/160 fallback, as InfU does when it reads them.

Runs in the `infu` env, which has facexlib and its weights staged offline.

Usage:
  python tools/score_infu_encoder.py --manifest benchmark/manifest_repro.parquet
      --images $WORK/prl26/results/images/infu_aes2 --method infu_aes2
      --out $WORK/prl26/results/scores/irse50_infu_aes2_manifest_repro.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file  # noqa: E402


class InfUEncoder:
    def __init__(self, insightface_root: str):
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
        # Detection only: the kps InfU uses come from the detector, and skipping
        # the other antelopev2 heads makes CPU detection several times faster.
        self.apps = {}
        for s in (640, 320, 160):
            app = FaceAnalysis(name="antelopev2", root=insightface_root,
                               allowed_modules=["detection"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(s, s))
            self.apps[s] = app
        self.model = init_recognition_model("arcface", device="cuda")

    @staticmethod
    def _largest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @torch.no_grad()
    def _embed(self, bgr, kps):
        from insightface.utils import face_align
        crop = face_align.norm_crop(bgr, landmark=np.array(kps), image_size=112)
        x = torch.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        x = (2 * x - 1).cuda().contiguous()
        e = self.model(x)[0].float().cpu().numpy()
        return e / np.linalg.norm(e)

    def embed_generated(self, bgr):
        faces = self.apps[640].get(bgr)
        return None if not faces else self._embed(bgr, self._largest(faces).kps)

    def embed_reference(self, bgr):
        for s in (640, 320, 160):
            faces = self.apps[s].get(bgr)
            if faces:
                return self._embed(bgr, self._largest(faces).kps)
        return None


def score_floor(enc, a):
    """The floor: filtered reference photographs scored against the originals."""
    rows, refs = [], {}
    for p in sorted(Path(a.images).glob("*__floor_*.png")):
        iid, rest = p.stem.split("__floor_", 1)
        level, name = rest.split("_", 1)
        if iid not in refs:
            refs[iid] = enc.embed_reference(cv2.imread(str(ROOT / "benchmark/identities" / f"{iid}.jpg")))
        g = enc.embed_generated(cv2.imread(str(p)))
        rows.append({"iid": iid, "level": int(level), "filter": name,
                     "id_loss_irse50": None if g is None else float(1.0 - np.dot(g, refs[iid]))})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["level", "filter"])["id_loss_irse50"].agg(["mean", "std", "count"]).round(4))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insightface-root", default=None)
    ap.add_argument("--floor", action="store_true",
                    help="--images holds the domain-shift floor, <iid>__floor_<k>_<name>.png, "
                         "each scored against benchmark/identities/<iid>.jpg")
    a = ap.parse_args()

    enc = InfUEncoder(a.insightface_root)
    if a.floor:
        return score_floor(enc, a)
    m = pd.read_parquet(ROOT / a.manifest)
    refs, rows, t0 = {}, [], time.time()
    for i, r in enumerate(m.itertuples(), 1):
        if r.id_path not in refs:
            e = enc.embed_reference(cv2.imread(str(ROOT / r.id_path)))
            if e is None:
                raise SystemExit(f"no face in reference {r.id_path}")
            refs[r.id_path] = e
        p = Path(a.images) / cell_to_file(r.cell)
        img = cv2.imread(str(p)) if p.exists() else None
        g = None if img is None else enc.embed_generated(img)
        rows.append({"cell": r.cell, "method": a.method, "iid": r.iid, "pid": r.pid,
                     "image_found": img is not None, "face_detected_irse50": g is not None,
                     "id_loss_irse50": None if g is None else float(1.0 - np.dot(g, refs[r.id_path]))})
        if i % 250 == 0:
            print(f"  {i}/{len(m)}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    v = df["id_loss_irse50"].dropna()
    print(f"wrote {a.out}: {len(df)} cells, {df.image_found.sum()} images, "
          f"{df.face_detected_irse50.sum()} faces, mean ID Loss (IR-SE50) {v.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Are the faces the detector missed really absent? Look again at full resolution.

Every ID Loss in Part 1 comes from antelopev2's SCRFD detector at det_size 640:
an 864x1152 image is shrunk to fit 640x640 first, so a face in a full-body shot
loses almost half its width before the detector sees it. A cell with no face is
dropped from that method's mean, and the two methods lose different cells (InfU
119, PuLID 206 at start 4, 74 at start 0). The report bounds what that can do
by imputing the worst loss. This measures it instead: each missed image is
searched again by the same detector at det_size 1152, where the image is not
shrunk at all, and a face found there is scored by both methods' encoders
exactly as Part 1 scores faces (largest face; glintr100 via antelopev2,
IR-SE50 via InfU's own crop, see tools/score_infu_encoder.py). A second, looser
pass at detection threshold 0.3 records how much more a lower bar finds; it is
reported separately, since a lower bar also admits things that are not faces.

A face found this way is a face the image has and the metric never counted.
What remains missing after both passes is a candidate for "no usable face".

Runs in the `infu` env (insightface on CPU, facexlib on the GPU).

Usage:
  python tools/recover_missed_faces.py --scores $WORK/prl26/results/scores
      --images $WORK/prl26/results/images --out-dir $WORK/prl26/results/scores
      --sets infu_aes2 pulid pulid_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file  # noqa: E402

FULL = 1152          # the long side of every generated image: no downscaling


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


class Judges:
    def __init__(self, root):
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis

        def app(size, thresh=0.5):
            a = FaceAnalysis(name="antelopev2", root=root,
                             allowed_modules=["detection", "recognition"],
                             providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=-1, det_size=(size, size), det_thresh=thresh)
            return a
        self.ref640, self.ref320, self.ref160 = app(640), app(320), app(160)
        self.full, self.loose = app(FULL), app(FULL, 0.3)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.irse = init_recognition_model("arcface", device=self.dev)

    @torch.no_grad()
    def irse50(self, bgr, kps):
        from insightface.utils import face_align
        crop = face_align.norm_crop(bgr, landmark=np.array(kps), image_size=112)
        x = torch.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        e = self.irse((2 * x - 1).to(self.dev).contiguous())[0].float().cpu().numpy()
        return e / np.linalg.norm(e)

    def reference(self, bgr):
        """glintr100 at 640 as id_loss.py reads references; IR-SE50 with InfU's fallback."""
        f = self.ref640.get(bgr)
        glint = largest(f).normed_embedding
        for a in (self.ref640, self.ref320, self.ref160):
            f = a.get(bgr)
            if f:
                return glint, self.irse50(bgr, largest(f).kps)
        raise SystemExit("no face in a reference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", nargs="+", default=["infu_aes2", "pulid", "pulid_s0"])
    ap.add_argument("--manifest", default="manifest_repro")
    ap.add_argument("--insightface-root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N missed cells per set")
    a = ap.parse_args()

    J = Judges(a.insightface_root)
    man = pd.read_parquet(ROOT / "benchmark" / f"{a.manifest}.parquet").set_index("cell")
    refs = {}
    for s in a.sets:
        sc = pd.read_parquet(Path(a.scores) / f"{s}_{a.manifest}.parquet")
        missed = sc.loc[~sc["face_detected"].astype(bool), "cell"].tolist()
        if a.limit:
            missed = missed[:a.limit]
        rows = []
        for cell in missed:
            r = man.loc[cell]
            if r.id_path not in refs:
                refs[r.id_path] = J.reference(cv2.imread(str(ROOT / r.id_path)))
            g_ref, i_ref = refs[r.id_path]
            bgr = cv2.imread(str(Path(a.images) / s / cell_to_file(cell)))
            rec = {"cell": cell, "method": s, "iid": r.iid, "pid": r.pid,
                   "face_size": r.face_size, "complexity": r.complexity,
                   "image_found": bgr is not None, "found_at": "none"}
            if bgr is not None:
                for tag, app in (("full", J.full), ("loose", J.loose)):
                    faces = app.get(bgr)
                    if faces:
                        f = largest(faces)
                        x1, y1, x2, y2 = f.bbox
                        rec.update(found_at=tag, det_score=float(f.det_score),
                                   face_frac=float((x2 - x1) * (y2 - y1) / (bgr.shape[0] * bgr.shape[1])),
                                   face_w_px=float(x2 - x1),
                                   id_loss_antelope=float(1.0 - np.dot(f.normed_embedding, g_ref)),
                                   id_loss_irse50=float(1.0 - np.dot(J.irse50(bgr, f.kps), i_ref)))
                        break
            rows.append(rec)
        df = pd.DataFrame(rows)
        out = Path(a.out_dir) / f"recovered_{s}_{a.manifest}.csv"
        df.to_csv(out, index=False)
        print(f"{s}: {len(df)} missed at 640 -> {df.found_at.value_counts().to_dict()}")
        if len(df):
            print(df.groupby(["face_size", "found_at"]).size().unstack(fill_value=0))
            print(df.groupby("found_at")[["id_loss_antelope", "id_loss_irse50", "face_w_px"]].mean().round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())

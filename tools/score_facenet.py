#!/usr/bin/env python
"""ID Loss under a judge from outside the ArcFace family: FaceNet on VGGFace2.

Every recogniser in Part 1 is an ArcFace network, and each method conditions on
one of them: PuLID on antelopev2's glintr100, InfU on facexlib's IR-SE50. Each
method wins under its own encoder, and buffalo_l, meant as the neutral check,
agrees with glintr100 at r = 0.985 per cell. This scores the same
images with facenet-pytorch's Inception-ResNet-v1 (Sandberg's 20180402-114759,
LFW 0.9965): a different architecture, a softmax classifier rather than an
additive angular margin, and VGGFace2 rather than MS1M, WebFace or Glint360K.

The pipeline is the model's own, not ours: its MTCNN finds the faces, the
largest is kept (the rule of src/metrics/id_loss.py), and a 160x160 crop with a
14 px margin is cut and standardised, the settings of facenet-pytorch's LFW
evaluation (examples/lfw_evaluate.ipynb). Detection is therefore independent
too, and is recorded per image: a second detector's view of the detection gap.

Runs in the `infu` env with facenet-pytorch 2.6.0 installed --no-deps (it pins
an older torch; the model code needs only torch, torchvision and PIL). The
weights are staged in $TORCH_HOME/checkpoints; compute nodes have no internet.

Usage:
  python tools/score_facenet.py --manifest benchmark/manifest_repro.parquet
      --images $WORK/prl26/results/images/infu_aes2 --method infu_aes2
      --out $WORK/prl26/results/scores/facenet_infu_aes2_manifest_repro.csv
  python tools/score_facenet.py --floor --images $WORK/prl26/results/images/floor
      --out $WORK/prl26/results/scores/facenet_floor.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file  # noqa: E402

MARGIN, SIZE = 14, 160


class FaceNet:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(image_size=SIZE, margin=MARGIN, select_largest=True,
                           post_process=True, device=self.dev)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.dev)

    @torch.no_grad()
    def embed(self, img: Image.Image):
        """(unit embedding, detector probability, face box area / image area), or Nones."""
        from facenet_pytorch import extract_face, fixed_image_standardization
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None, None, None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        face = fixed_image_standardization(extract_face(img, boxes[i], SIZE, MARGIN))
        e = self.net(face.unsqueeze(0).to(self.dev))[0].float().cpu().numpy()
        return e / np.linalg.norm(e), float(probs[i]), float(areas[i] / (img.width * img.height))


def load(p: Path):
    try:
        im = Image.open(p)
        im.load()
        return im.convert("RGB")
    except (FileNotFoundError, OSError):
        return None


def score_floor(fn, a):
    """The floor: filtered reference photographs scored against the originals."""
    rows, refs = [], {}
    for p in sorted(Path(a.images).glob("*__floor_*.png")):
        iid, rest = p.stem.split("__floor_", 1)
        level, name = rest.split("_", 1)
        if iid not in refs:
            refs[iid] = fn.embed(load(ROOT / "benchmark/identities" / f"{iid}.jpg"))[0]
        g, prob, _ = fn.embed(load(p))
        rows.append({"iid": iid, "level": int(level), "filter": name, "det_prob_facenet": prob,
                     "id_loss_facenet": None if g is None else float(1.0 - np.dot(g, refs[iid]))})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["level", "filter"])["id_loss_facenet"].agg(["mean", "std", "count"]).round(4))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="floor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", action="store_true",
                    help="--images holds the domain-shift floor, <iid>__floor_<k>_<name>.png")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N cells")
    a = ap.parse_args()

    fn = FaceNet()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.floor:
        return score_floor(fn, a)
    m = pd.read_parquet(ROOT / a.manifest)
    if a.limit:
        m = m.head(a.limit)
    refs, rows, t0 = {}, [], time.time()
    for i, r in enumerate(m.itertuples(), 1):
        if r.id_path not in refs:
            e = fn.embed(load(ROOT / r.id_path))[0]
            if e is None:
                raise SystemExit(f"no face in reference {r.id_path}")
            refs[r.id_path] = e
        img = load(Path(a.images) / cell_to_file(r.cell))
        g, prob, frac = (None, None, None) if img is None else fn.embed(img)
        rows.append({"cell": r.cell, "method": a.method, "iid": r.iid, "pid": r.pid,
                     "image_found": img is not None, "face_detected_facenet": g is not None,
                     "det_prob_facenet": prob, "face_frac_facenet": frac,
                     "id_loss_facenet": None if g is None else float(1.0 - np.dot(g, refs[r.id_path]))})
        if i % 250 == 0:
            print(f"  {i}/{len(m)}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    v = df["id_loss_facenet"].dropna()
    print(f"wrote {a.out}: {len(df)} cells, {df.image_found.sum()} images, "
          f"{df.face_detected_facenet.sum()} faces, mean ID Loss (FaceNet) {v.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

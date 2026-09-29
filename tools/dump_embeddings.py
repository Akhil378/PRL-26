#!/usr/bin/env python
"""Every image's face and CLIP embeddings, saved, so analysis survives the cluster.

Cluster access ends on 1 Oct 2026, and the images and weights stay there. The
score tables keep one number per image and judge, which answers the questions
already asked and no others. This keeps the embeddings instead, so that any
later question -- identification among the fifteen identities, a new text
phrase, a different pairing -- can be answered offline from a few hundred
megabytes.

Per image, in one pass:
  detection   antelopev2 SCRFD at det_size 640, largest face, as every ID Loss
              in the project; where it finds nothing, the fallback passes of
              tools/recover_missed_faces.py (320, 160, a padded 640, 1152), with
              the pass recorded and the landmarks in original coordinates
  glintr100   PuLID's encoder, from that detection (antelopev2 recognition)
  w600k_r50   buffalo_l's recognition network on the same five landmarks. The
              buffalo_l scores in the tables used buffalo_l's own detector, so
              these embeddings can differ slightly from them
  IR-SE50     InfU's encoder, on the same landmarks, exactly as
              tools/score_infu_encoder.py
  FaceNet     facenet-pytorch Inception-ResNet-v1 trained on VGGFace2, and the
              same architecture trained on CASIA-WebFace, on the crop of their
              own MTCNN (largest face, 160 px, 14 px margin; tools/score_facenet.py)
  CLIP        ViT-L/14 and ViT-B/32 image embeddings, unit length, as
              src/metrics/clip_backend.py computes them
References get the same treatment, with InfU's 640/320/160 fallback for them.
Face embeddings are unit length and NaN where no face was found.

Runs in the `infu` env on one GPU.

Usage:
  python tools/dump_embeddings.py --sets infu_aes2:manifest_repro floor ceiling
      --images $WORK/prl26/results/images --out $WORK/prl26/results/embeddings
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.clip_backend import CLIPBackend  # noqa: E402
from src.runner_common import cell_to_file  # noqa: E402

D = 512
PAD = 0.25


def largest_idx(boxes):
    return int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))


def unit(e):
    e = np.asarray(e, np.float32).ravel()
    return e / np.linalg.norm(e)


class Embedder:
    def __init__(self, root):
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        def app(size):
            a = FaceAnalysis(name="antelopev2", root=root, allowed_modules=["detection", "recognition"],
                             providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=-1, det_size=(size, size))
            return a
        self.det = {s: app(s) for s in (640, 320, 160, 1152)}
        self.w600k = get_model(str(Path(root) / "models/buffalo_l/w600k_r50.onnx"),
                               providers=["CPUExecutionProvider"])
        self.w600k.prepare(ctx_id=-1)
        self.irse = init_recognition_model("arcface", device="cuda")
        self.mtcnn = MTCNN(image_size=160, margin=14, select_largest=True, post_process=True, device="cuda")
        self.fn = {k: InceptionResnetV1(pretrained=k).eval().cuda() for k in ("vggface2", "casia-webface")}
        self.clip = {b: CLIPBackend(b) for b in ("L14", "B32")}

    def detect(self, bgr, reference=False):
        """(face, pass name, the image searched, x offset, y offset); face None if none found."""
        order = [(640, None), (320, None), (160, None)] if reference else \
                [(640, None), (320, None), (160, None), (640, "pad"), (1152, None)]
        for size, tf in order:
            img, dx, dy = bgr, 0, 0
            if tf == "pad":
                dy, dx = int(PAD * bgr.shape[0]), int(PAD * bgr.shape[1])
                img = cv2.copyMakeBorder(bgr, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.det[size].get(img)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                name = "640" if size == 640 and tf is None else ("pad" if tf else str(size))
                return f, name, img, dx, dy
        return None, "none", bgr, 0, 0

    @torch.no_grad()
    def irse50(self, img, kps):
        from insightface.utils import face_align
        crop = face_align.norm_crop(img, landmark=np.array(kps), image_size=112)
        x = torch.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        return unit(self.irse((2 * x - 1).cuda().contiguous())[0].float().cpu().numpy())

    @torch.no_grad()
    def facenet(self, pil):
        from facenet_pytorch import extract_face, fixed_image_standardization
        boxes, probs = self.mtcnn.detect(pil)
        if boxes is None or len(boxes) == 0:
            return None
        i = largest_idx(boxes)
        x = fixed_image_standardization(extract_face(pil, boxes[i], 160, 14)).unsqueeze(0).cuda()
        return {k: unit(m(x)[0].float().cpu().numpy()) for k, m in self.fn.items()}, boxes[i], float(probs[i])

    def embed(self, path: Path, reference=False) -> dict:
        nan = np.full(D, np.nan, np.float32)
        out = {"found": False, "det_pass": "none", "det_score": np.nan, "bbox": np.full(4, np.nan),
               "kps": np.full((5, 2), np.nan), "glint": nan, "w600k": nan, "irse50": nan,
               "fn_vgg": nan, "fn_casia": nan, "mtcnn_box": np.full(4, np.nan), "mtcnn_prob": np.nan,
               "clip_l14": np.full(768, np.nan, np.float32), "clip_b32": nan}
        bgr = cv2.imread(str(path))
        if bgr is None:
            return out
        out["found"] = True
        f, name, img, dx, dy = self.detect(bgr, reference)
        if f is not None:
            glint = unit(f.normed_embedding)       # before w600k.get, which overwrites f.embedding
            out.update(det_pass=name, det_score=float(f.det_score),
                       bbox=np.array(f.bbox) - [dx, dy, dx, dy], kps=np.array(f.kps) - [dx, dy],
                       glint=glint, irse50=self.irse50(img, f.kps), w600k=unit(self.w600k.get(img, f)))
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        r = self.facenet(pil)
        if r is not None:
            e, box, prob = r
            out.update(fn_vgg=e["vggface2"], fn_casia=e["casia-webface"], mtcnn_box=box, mtcnn_prob=prob)
        if not reference:
            for b, c in self.clip.items():
                out[f"clip_{b.lower()}"] = c.encode_image(pil)[0].float().cpu().numpy()
        return out


def items_for(spec, images):
    """(set name, [(key, image path, reference path or '')])."""
    name, _, manifest = spec.partition(":")
    d = Path(images) / name
    if manifest:
        m = pd.read_parquet(ROOT / "benchmark" / f"{manifest}.parquet")
        return name, [(r.cell, d / cell_to_file(r.cell), r.id_path) for r in m.itertuples()]
    rows = []
    for p in sorted(d.glob("*.png")):
        ref = ""
        if name == "floor":
            ref = f"benchmark/identities/{p.stem.split('__floor_')[0]}.jpg"
        rows.append((p.stem, p, ref))
    return name, rows


def save(path, recs, keys, refs):
    arr = {k: np.stack([r[k] for r in recs]) for k in recs[0]}
    arr["key"] = np.array(keys)
    arr["ref"] = np.array(refs)
    np.savez_compressed(path, **arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="name:manifest, or a bare name to glob *.png")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insightface-root", required=True)
    ap.add_argument("--refs", action="store_true", help="also embed every identity photograph")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E = Embedder(a.insightface_root)

    if a.refs:
        paths = sorted((ROOT / "benchmark/identities").glob("*.jpg")) + \
                sorted((ROOT / "benchmark/identities_paper").glob("*.jpg"))
        recs = [E.embed(p, reference=True) for p in paths]
        save(out / "refs.npz", recs, [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths],
             [""] * len(paths))
        print(f"refs: {len(paths)} photographs, faces {sum(r['det_pass'] != 'none' for r in recs)}")

    for spec in a.sets:
        name, items = items_for(spec, a.images)
        if a.limit:
            items = items[:a.limit]
        t0, recs = time.time(), []
        for i, (key, p, ref) in enumerate(items, 1):
            recs.append(E.embed(p))
            if i % 250 == 0:
                print(f"  {name} {i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        save(out / f"{name}.npz", recs, [k for k, _, _ in items], [r for _, _, r in items])
        passes = pd.Series([r["det_pass"] for r in recs]).value_counts().to_dict()
        print(f"{name}: {len(items)} images, SCRFD passes {passes}, "
              f"MTCNN faces {sum(np.isfinite(r['mtcnn_prob']) for r in recs)}, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

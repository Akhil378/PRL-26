#!/usr/bin/env python
"""Three more identity judges from outside the ArcFace lineage, for a neutral panel.

FaceNet (tools/score_facenet.py) was the first judge from outside the ArcFace
family, and on raw correlation it sits equally far from both methods' encoders.
A stricter test says it leans: regressing its per-cell verdict (PuLID minus InfU)
on the two encoders' verdicts puts more weight on InfU's IR-SE50 than on PuLID's
glintr100 (tools/analyse_panel.py). One judge's lean cannot be told from its
verdict, so this adds three more, each different from ArcFace in network,
training loss or data, and each run through its own published pipeline:

  sface    SFace (Zhong et al.), MobileFaceNet with the sigmoid-constrained
           hypersphere loss, from the OpenCV model zoo, on the crop its YuNet
           detector's five landmarks give (cv2.FaceRecognizerSF.alignCrop).
           YuNet looks at the image with its long side at 640, then at full
           size, then at 320, and keeps the first hit.
  vggface  VGG-Face (Parkhi et al., 2015), VGG16 trained on the VGGFace set with
           a softmax classifier, Albanie's PyTorch conversion; the L2-normalised
           fc7 output of a 224x224 crop of the MTCNN box grown by 30%, as the
           paper extends its boxes to take the whole head.
  dlib     dlib's ResNet face descriptor (metric learning), on the face chip its
           five-point shape predictor aligns inside the MTCNN box. Runs in the
           `dlib` env; the others in `infu`.

MTCNN boxes are the ones tools/dump_embeddings.py saved, so VGG-Face and dlib see
the faces FaceNet saw. ID Loss is 1 - cos of unit embeddings for every judge.

Usage:
  python tools/score_panel.py --judge sface --emb $WORK/prl26/results/embeddings
      --images $WORK/prl26/results/images --models $WORK/facejudges
      --out $WORK/prl26/results/panel
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"refs", "ceiling"}


def unit(e):
    e = np.asarray(e, np.float32).ravel()
    return e / np.linalg.norm(e)


def cell_file(key):
    return key.replace("|", "__") + ".png"


class SFace:
    def __init__(self, models):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(models / "face_detection_yunet_2023mar.onnx"), "",
                                             (320, 320), 0.9, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(models / "face_recognition_sface_2021dec.onnx"), "")

    def __call__(self, path, box=None):
        cv2 = self.cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        H, W = bgr.shape[:2]
        for target in (640, None, 320):
            s = 1.0 if target is None else target / max(H, W)
            img = bgr if s == 1.0 else cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            self.det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.det.detect(img)
            if faces is not None and len(faces):
                f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))].copy()
                f[:14] /= s
                return unit(self.rec.feature(self.rec.alignCrop(bgr, f)))
        return None


class VGGFace:
    def __init__(self, models):
        import importlib.util
        import torch
        from PIL import Image
        self.torch, self.Image = torch, Image
        spec = importlib.util.spec_from_file_location("vgg_face_dag", models / "vgg_face_dag.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.m = mod.Vgg_face_dag()
        self.m.load_state_dict(torch.load(models / "vgg_face_dag.pth", map_location="cpu"))
        self.m = self.m.eval().cuda()
        self.mean = torch.tensor(self.m.meta["mean"], dtype=torch.float32).view(1, 3, 1, 1).cuda()

    def __call__(self, path, box):
        torch = self.torch
        if box is None or not np.isfinite(box).all():
            return None
        im = self.Image.open(path).convert("RGB")
        x1, y1, x2, y2 = box
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, 1.3 * max(x2 - x1, y2 - y1)
        crop = im.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)).resize((224, 224), self.Image.BILINEAR)
        x = torch.from_numpy(np.asarray(crop, np.float32)).permute(2, 0, 1).unsqueeze(0).cuda() - self.mean
        with torch.no_grad():
            for name, layer in self.m.named_children():
                if name == "fc6":
                    x = x.flatten(1)
                x = layer(x)
                if name == "relu7":
                    break
        return unit(x[0].cpu().numpy())


class Dlib:
    def __init__(self, models):
        import dlib
        self.dlib = dlib
        self.sp = dlib.shape_predictor(str(models / "shape_predictor_5_face_landmarks.dat"))
        self.fr = dlib.face_recognition_model_v1(str(models / "dlib_face_recognition_resnet_model_v1.dat"))

    def __call__(self, path, box):
        if box is None or not np.isfinite(box).all():
            return None
        img = self.dlib.load_rgb_image(str(path))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        shape = self.sp(img, self.dlib.rectangle(x1, y1, x2, y2))
        return unit(np.array(self.fr.compute_face_descriptor(img, shape)))


JUDGES = {"sface": SFace, "vggface": VGGFace, "dlib": Dlib}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--emb", required=True, help="tools/dump_embeddings.py output: <set>.npz, refs.npz")
    ap.add_argument("--images", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", help="default: every <set>.npz but refs and ceiling")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    J = JUDGES[a.judge](Path(a.models))
    emb, out = Path(a.emb), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    R = np.load(emb / "refs.npz")
    refs = {}
    for k, box in zip(R["key"], R["mtcnn_box"]):
        e = J(ROOT / str(k), box)
        if e is None:
            raise SystemExit(f"{a.judge}: no face in reference {k}")
        refs[str(k)] = e
    print(f"{a.judge}: {len(refs)} references embedded", flush=True)

    sets = a.sets or sorted(p.stem for p in emb.glob("*.npz") if p.stem not in SKIP)
    for s in sets:
        z = np.load(emb / f"{s}.npz")
        keys, rkeys, boxes = [str(k) for k in z["key"]], [str(r) for r in z["ref"]], z["mtcnn_box"]
        n = a.limit or len(keys)
        rows, E, t0 = [], np.full((n, 0), np.nan), time.time()
        vecs = []
        for i in range(n):
            e = J(Path(a.images) / s / cell_file(keys[i]), boxes[i])
            ref = refs.get(rkeys[i])
            loss = None if e is None or ref is None else float(1.0 - np.dot(e, ref))
            rows.append({"key": keys[i], f"face_{a.judge}": e is not None, f"id_loss_{a.judge}": loss})
            vecs.append(e)
            if (i + 1) % 500 == 0:
                print(f"  {s} {i + 1}/{n} {time.time() - t0:.0f}s", flush=True)
        dim = next((len(v) for v in vecs if v is not None), 1)
        E = np.stack([v if v is not None else np.full(dim, np.nan, np.float32) for v in vecs])
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{a.judge}_{s}.csv", index=False)
        np.savez_compressed(out / f"{a.judge}_{s}.npz", key=np.array(keys[:n]), emb=E)
        print(f"{a.judge} {s}: {n} images, faces {df[f'face_{a.judge}'].sum()}, "
              f"mean ID Loss {pd.to_numeric(df[f'id_loss_{a.judge}']).mean():.4f}, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / f"{a.judge}_refs.npz", key=np.array(list(refs)), emb=np.stack(list(refs.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Score a directory of generated images into one table.

Every table and figure in the report is a groupby over the Parquet file this
writes. Resumable: already-scored cells are skipped, so scoring can run on a
small GPU alongside generation on the A100 and be re-run as images land.

Usage:
  python src/score_all.py --manifest benchmark/manifest_repro.parquet \\
      --images $WORK/prl26/results/images/infu_aes2 --method infu_aes2 \\
      --out $WORK/prl26/results/scores/infu_aes2.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.clip_backend import CLIPBackend          # noqa: E402
from src.metrics.face_masking import face_masking_index   # noqa: E402
from src.metrics.id_loss import IDScorer                  # noqa: E402
from src.metrics.pickscore import PAPER_SCALE, PickScorer  # noqa: E402


def cell_to_file(cell: str) -> str:
    return cell.replace("|", "__") + ".png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-root", default=None, help="INSIGHTFACE_HOME/models")
    ap.add_argument("--skip-b32", action="store_true")
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="score N rows then stop")
    args = ap.parse_args()

    man = pd.read_parquet(ROOT / args.manifest) if not Path(args.manifest).is_absolute() \
        else pd.read_parquet(args.manifest)
    img_dir = Path(args.images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        prev = pd.read_parquet(out)
        done = set(prev.cell)
        print(f"resuming: {len(done)} cells already scored")
    else:
        prev = None

    man["file"] = man.cell.map(cell_to_file)
    todo = man[~man.cell.isin(done)].copy()
    todo["exists"] = todo.file.map(lambda f: (img_dir / f).exists())
    missing = int((~todo.exists).sum())
    todo = todo[todo.exists]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} to score, {missing} not yet generated")
    if todo.empty:
        return 0

    is_style = "style_phrase" in todo.columns
    print("loading models...")
    id_a = IDScorer("antelopev2", root=args.id_root)
    id_b = IDScorer("buffalo_l", root=args.id_root)
    clip_l = CLIPBackend("L14")
    clip_b = None if args.skip_b32 else CLIPBackend("B32")
    picker = PickScorer()

    rows, t0 = [], time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        path = str(img_dir / r.file)
        ref = str(ROOT / r.id_path)
        img = Image.open(path).convert("RGB")

        loss_a, det_a = id_a.id_loss(path, ref)
        loss_b, _ = id_b.id_loss(path, ref)

        rec = {
            "cell": r.cell, "method": args.method, "iid": r.iid, "pid": r.pid,
            "gender": r.gender, "face_size": r.face_size,
            "complexity": r.complexity, "length": r.length, "seed": r.seed,
            "id_loss_antelope": loss_a, "id_loss_buffalo": loss_b,
            "face_detected": det_a,
            "clipscore_l14": clip_l.clipscore(img, r.prompt),
            "pickscore_raw": picker.score_one(img, r.prompt),
            "prompt_truncates": clip_l.truncates(r.prompt),
        }
        rec["pickscore_paper"] = rec["pickscore_raw"] / PAPER_SCALE
        if clip_b is not None:
            rec["clipscore_b32"] = clip_b.clipscore(img, r.prompt)

        if is_style:
            rec["style"] = r.style
            rec["base_pid"] = r.base_pid
            rec["style_score"] = clip_l.clipscore(img, r.style_phrase)
            # reuse the detection that produced the ID embedding
            import cv2
            _, box = id_a.embed_with_box(cv2.imread(path))
            if box is None:
                rec.update(fmi=None, style_face=None, style_bg=None,
                           fmi_reason="no_face")
            else:
                fmi, sf, sb, why = face_masking_index(
                    img, r.style_phrase, box, clip_l)
                rec.update(fmi=fmi, style_face=sf, style_bg=sb, fmi_reason=why)

        rows.append(rec)

        if n % args.flush_every == 0:
            frames = [prev] if prev is not None else []
            pd.concat(frames + [pd.DataFrame(rows)], ignore_index=True).to_parquet(out, index=False)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo)-n)/rate/60:.1f} min",
                  flush=True)

    frames = [prev] if prev is not None else []
    final = pd.concat(frames + [pd.DataFrame(rows)], ignore_index=True)
    final.to_parquet(out, index=False)

    print(f"\nwrote {out}  ({len(final)} rows)")
    d = final[final.method == args.method]
    print(f"  face detection rate  {d.face_detected.mean():.3f}")
    print(f"  ID Loss (antelopev2) {d.id_loss_antelope.mean():.3f}")
    print(f"  ID Loss (buffalo_l)  {d.id_loss_buffalo.mean():.3f}")
    print(f"  CLIPScore  ViT-L/14  {d.clipscore_l14.mean():.3f}")
    print(f"  PickScore  raw       {d.pickscore_raw.mean():.2f}"
          f"   /100 = {d.pickscore_raw.mean()/PAPER_SCALE:.3f}")
    if "fmi" in d:
        print(f"  FMI                  {d.fmi.mean():.4f}"
              f"  ({d.fmi.isna().mean():.1%} undefined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

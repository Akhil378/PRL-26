#!/usr/bin/env python
"""Score a directory of generated images into one table.

Every table and figure in the report is a groupby over the Parquet file this
writes. Resumable: already-scored cells are skipped, so scoring can run on a
cheap GPU alongside generation on the A100 and be re-run as images land.

Backends are selectable so the whole path can be exercised on a laptop:
  --id-backend stub      OpenCV stand-in; test plumbing only, never for results
  --clip-backbones B32   skip the 1.7 GB ViT-L/14 download during a smoke test
  --skip-pickscore       skip the 4 GB CLIP-H download during a smoke test

Cluster usage:
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
from src.metrics.pickscore import PAPER_SCALE             # noqa: E402


def cell_to_file(cell: str) -> str:
    return cell.replace("|", "__") + ".png"


def make_id_scorer(backend: str, pack: str, root):
    if backend == "stub":
        from src.metrics.id_stub import StubIDScorer
        return StubIDScorer(pack=pack)
    from src.metrics.id_loss import IDScorer      # imported lazily: insightface
    return IDScorer(pack=pack, root=root)         # does not build on Windows


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-backend", default="insightface",
                    choices=["insightface", "stub"])
    ap.add_argument("--id-root", default=None, help="INSIGHTFACE_HOME/models")
    ap.add_argument("--clip-backbones", default="L14,B32",
                    help="Comma-separated; the first is primary and drives FMI.")
    ap.add_argument("--skip-pickscore", action="store_true")
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_parquet(resolve(args.manifest))
    img_dir = resolve(args.images)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    prev = pd.read_parquet(out) if out.exists() else None
    done = set(prev.cell) if prev is not None else set()
    if done:
        print(f"resuming: {len(done)} cells already scored")

    man["file"] = man.cell.map(cell_to_file)
    todo = man[~man.cell.isin(done)].copy()
    if todo.empty:
        # Every cell already scored. This is the normal state on a re-run, so it
        # must exit cleanly rather than reduce over an empty frame: on an empty
        # object-dtype column .sum() returns '' and int('') raises.
        print("0 to score - all cells already scored")
        return 0
    todo["exists"] = todo.file.map(lambda f: (img_dir / f).exists()).astype(bool)
    missing = int((~todo["exists"]).sum())
    todo = todo[todo["exists"]]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} to score, {missing} not yet generated")
    if todo.empty:
        print("nothing new to score")
        return 0

    is_style = "style_phrase" in todo.columns
    print(f"loading models (id={args.id_backend}, clip={args.clip_backbones})...")
    packs = ["antelopev2", "buffalo_l"]
    id_scorers = {p: make_id_scorer(args.id_backend, p, args.id_root) for p in packs}
    backbones = [b.strip() for b in args.clip_backbones.split(",") if b.strip()]
    clips = {b: CLIPBackend(b) for b in backbones}
    primary = clips[backbones[0]]
    picker = None
    if not args.skip_pickscore:
        from src.metrics.pickscore import PickScorer
        picker = PickScorer()

    import cv2
    rows, t0 = [], time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        path = str(img_dir / r.file)
        ref = str(resolve(r.id_path))
        img = Image.open(path).convert("RGB")

        loss_a, det = id_scorers["antelopev2"].id_loss(path, ref)
        loss_b, _ = id_scorers["buffalo_l"].id_loss(path, ref)

        rec = {
            "cell": r.cell, "method": args.method, "iid": r.iid, "pid": r.pid,
            "gender": r.gender, "face_size": r.face_size,
            "complexity": r.complexity, "length": r.length, "seed": r.seed,
            "id_backend": args.id_backend,
            "id_loss_antelope": loss_a, "id_loss_buffalo": loss_b,
            "face_detected": det,
            "prompt_truncates": primary.truncates(r.prompt),
        }
        for b in backbones:
            rec[f"clipscore_{b.lower()}"] = clips[b].clipscore(img, r.prompt)
        if picker is not None:
            rec["pickscore_raw"] = picker.score_one(img, r.prompt)
            rec["pickscore_paper"] = rec["pickscore_raw"] / PAPER_SCALE

        if is_style:
            rec["style"] = r.style
            rec["base_pid"] = r.base_pid
            rec["style_score"] = primary.clipscore(img, r.style_phrase)
            _, box = id_scorers["antelopev2"].embed_with_box(cv2.imread(path))
            if box is None:
                rec.update(fmi_style=None, fmi_photo=None, style_face=None,
                           style_bg=None, photo_face=None, photo_bg=None,
                           fmi_reason="no_face")
            else:
                rec.update(face_masking_index(img, r.style_phrase, box, primary))

        rows.append(rec)
        if n % args.flush_every == 0:
            pd.concat([d for d in (prev, pd.DataFrame(rows)) if d is not None],
                      ignore_index=True).to_parquet(out, index=False)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)}  {rate:.2f} img/s  "
                  f"eta {(len(todo)-n)/rate/60:.1f} min", flush=True)

    final = pd.concat([d for d in (prev, pd.DataFrame(rows)) if d is not None],
                      ignore_index=True)
    final.to_parquet(out, index=False)

    d = final[final.method == args.method]
    print(f"\nwrote {out}  ({len(final)} rows)")
    print(f"  face detection rate  {d.face_detected.mean():.3f}")
    print(f"  ID Loss (antelopev2) {d.id_loss_antelope.mean():.4f}"
          f"   [{d.id_loss_antelope.notna().sum()} defined]")
    print(f"  ID Loss (buffalo_l)  {d.id_loss_buffalo.mean():.4f}")
    for b in backbones:
        print(f"  CLIPScore {b:<4}         {d[f'clipscore_{b.lower()}'].mean():.4f}")
    if "pickscore_raw" in d:
        print(f"  PickScore raw        {d.pickscore_raw.mean():.2f}"
              f"   /100 = {d.pickscore_raw.mean()/PAPER_SCALE:.4f}")
    if "fmi_photo" in d:
        print(f"  FMI (photo form)     {d.fmi_photo.mean():+.4f}"
              f"   ({d.fmi_photo.isna().mean():.0%} undefined)")
        print(f"  FMI (style form)     {d.fmi_style.mean():+.4f}")
        print("  NOTE: absolute FMI is crop-content biased; report the paired")
        print("        difference against the photoreal control, never this mean.")
        print("  fmi_reason:", dict(d.fmi_reason.value_counts()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

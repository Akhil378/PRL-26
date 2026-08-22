#!/usr/bin/env python
"""Stage every weight PuLID-FLUX fetches at runtime, on the frontend.

Compute nodes have no outbound internet, so anything PuLID would lazily download
on first use has to be on disk before the job starts. Unlike InfU, PuLID does not
take model paths as arguments: `flux/util.py` hardcodes `models/flux1-dev.safetensors`
and `pulid/pipeline_flux.py` calls `hf_hub_download(..., local_dir='models')` and
`FaceAnalysis(root='.')`. Every one of those is RELATIVE TO THE CWD, and the jobs
cd to the repo root, so the repo root needs a `models/` directory.

It must not actually live in $HOME -- flux1-dev.safetensors alone is 23.8 GB and
$HOME is a 100 GB quota. So repo-root `models` is a symlink to $WORK.

Two facts make this cheap and safe, both verified against the installed
huggingface_hub 0.36.2 rather than assumed:

1. `load_flow_model`/`load_ae` test `os.path.exists(ckpt_path)` BEFORE any hub
   call, so the FLUX single-file weights only need to be *present*. They are
   already in the HF cache from the FLUX.1-dev repo download, so we link rather
   than re-fetch: 24 GB of download avoided.
2. Offline, `hf_hub_download(..., local_dir=X)` fails its HEAD call and then
   returns the existing local file, and `snapshot_download(..., local_dir=X)`
   returns X if it is non-empty. So staging the bare files into `models/` is
   sufficient; no upstream patching and no cache-metadata surgery is needed.

Run on the frontend (has internet), from the repo root:

    conda run -p $WORK/envs/pulid --no-capture-output python tools/stage_pulid.py

Safe to re-run; every step is idempotent. Exits non-zero if anything is missing
at the end, so a partial stage cannot masquerade as success.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The PuLID model release to use. app_flux.py defaults to v0.9.1, which the
# upstream changelog reports as ~5 percentage points better on facial similarity
# than v0.9.0 -- using the weaker one would understate the baseline.
PULID_VERSION = "v0.9.1"

# EVA-CLIP: pulid/pipeline_flux.py builds 'EVA02-CLIP-L-14-336' with the
# 'eva_clip' pretrained tag, which eva_clip/pretrained.py maps to this file.
EVA_REPO, EVA_FILE = "QuanSun/EVA-CLIP", "EVA02_CLIP_L_336_psz14_s6B.pt"


def log(msg: str) -> None:
    print(msg, flush=True)


def retry(what: str, fn, attempts: int = 5):
    """Downloads here are 1-24 GB over a link that has dropped mid-transfer
    before. hf_hub_download resumes from .incomplete blobs, so a retry costs
    only the bytes actually lost."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:                                    # noqa: BLE001
            log(f"    attempt {i}/{attempts} failed for {what}: "
                f"{type(e).__name__}: {str(e)[:120]}")
            if i < attempts:
                time.sleep(20)
    log(f"    GIVING UP on {what}")
    return None


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n/1:.1f} {unit}"
        n /= 1024.0
    return str(n)


def size_of(p: Path) -> str:
    try:
        return human(p.stat().st_size)          # follows symlinks, which is what we want
    except OSError:
        return "?"


def link(src: Path, dst: Path) -> bool:
    """Point dst at src, resolving through the HF cache's blob symlink.

    The cache keeps the real bytes in blobs/ and puts a symlink in snapshots/,
    so linking to the snapshot entry would be a symlink to a symlink. Resolve to
    the blob so there is exactly one hop and one thing that can break.
    """
    real = src.resolve()
    if not real.is_file():
        log(f"  MISSING source: {src}")
        return False
    if dst.is_symlink():
        if dst.resolve() == real:
            log(f"  ok (linked): {dst.name}  {size_of(dst)}")
            return True
        dst.unlink()
    elif dst.exists():
        log(f"  {dst.name} exists as a real file; leaving it alone")
        return True
    dst.symlink_to(real)
    log(f"  linked: {dst.name} -> {real}  ({size_of(dst)})")
    return True


def main() -> int:
    work = os.environ.get("WORK")
    if not work:
        log("WORK is unset -- run this on the cluster, not locally.")
        return 2
    prl = Path(os.environ.get("PRL", f"{work}/prl26"))
    models = prl / "models"
    models.mkdir(parents=True, exist_ok=True)

    # ---- repo-root models symlink ------------------------------------------
    # Everything below is addressed by PuLID as a path relative to the cwd.
    log("=== 6a. repo-root models/ symlink ===")
    repo_models = REPO / "models"
    if repo_models.is_symlink():
        if repo_models.resolve() != models.resolve():
            repo_models.unlink()
            repo_models.symlink_to(models)
        log(f"  {repo_models} -> {models}")
    elif repo_models.exists():
        # A real directory here would silently put 24 GB on the $HOME quota.
        log(f"  ERROR: {repo_models} exists and is not a symlink. Move it aside.")
        return 2
    else:
        repo_models.symlink_to(models)
        log(f"  created {repo_models} -> {models}")

    from huggingface_hub import hf_hub_download, snapshot_download

    missing: list[str] = []

    # ---- FLUX single-file weights, linked from the existing cache -----------
    log("=== 6b. FLUX.1-dev single-file weights (link, no download) ===")
    for fname in ("flux1-dev.safetensors", "ae.safetensors"):
        cached = retry(fname, lambda f=fname: hf_hub_download(
            "black-forest-labs/FLUX.1-dev", f, local_files_only=True))
        if cached is None:
            log(f"  {fname} is not in the HF cache; run step 3 of setup_cluster.sh first")
            missing.append(fname)
            continue
        if not link(Path(cached), models / fname):
            missing.append(fname)

    # ---- PuLID model weights ------------------------------------------------
    log("=== 6c. PuLID weights ===")
    # load_pretrain() calls this unconditionally, even when a path is passed, so
    # the file has to sit at exactly models/pulid_flux_<version>.safetensors.
    wname = f"pulid_flux_{PULID_VERSION}.safetensors"
    if (models / wname).exists():
        log(f"  ok: {wname}  {size_of(models / wname)}")
    else:
        got = retry(wname, lambda: hf_hub_download(
            "guozinan/PuLID", wname, local_dir=str(models)))
        if got is None:
            missing.append(wname)
        else:
            log(f"  fetched: {wname}  {size_of(models / wname)}")

    # ---- antelopev2 for PuLID's own ID encoder ------------------------------
    log("=== 6d. antelopev2 (PuLID's copy) ===")
    # PuLID resolves this as ./models/antelopev2 via FaceAnalysis(root='.') and
    # a literal 'models/antelopev2/glintr100.onnx'. It is a separate copy from
    # the one InfU conditions on; 6f checks whether the bytes actually differ.
    ante = models / "antelopev2"
    need = ["glintr100.onnx", "scrfd_10g_bnkps.onnx", "1k3d68.onnx",
            "2d106det.onnx", "genderage.onnx"]
    if all((ante / f).exists() for f in need):
        log(f"  ok: {len(need)} onnx files present")
    else:
        retry("DIAMONIK7777/antelopev2", lambda: snapshot_download(
            "DIAMONIK7777/antelopev2", local_dir=str(ante)))
        for f in need:
            if not (ante / f).exists():
                log(f"  MISSING {f}")
                missing.append(f"antelopev2/{f}")

    # ---- text encoders ------------------------------------------------------
    log("=== 6e. text encoders ===")
    # PuLID does NOT reuse FLUX's diffusers text encoders: flux/util.py loads T5
    # from xlabs-ai's standalone repo. This is what killed job 1789452.
    # T5Tokenizer is sentencepiece-based, so spiece.model must be present, not
    # just tokenizer.json.
    got = retry("xlabs-ai/xflux_text_encoders", lambda: snapshot_download(
        "xlabs-ai/xflux_text_encoders"))
    if got is None:
        missing.append("xlabs-ai/xflux_text_encoders")
    else:
        have = sorted(p.name for p in Path(got).iterdir() if p.is_file())
        log(f"  xflux_text_encoders: {len(have)} files")
        for req in ("spiece.model", "config.json"):
            if req not in have:
                log(f"  MISSING {req} (T5Tokenizer needs it offline)")
                missing.append(f"xflux_text_encoders/{req}")

    # openai/clip-vit-large-patch14 is already cached for the metrics env and
    # HF_HOME is shared, so load_clip() resolves offline with nothing to do.

    log("=== 6f. EVA-CLIP vision tower ===")
    got = retry(EVA_FILE, lambda: hf_hub_download(EVA_REPO, EVA_FILE))
    if got is None:
        missing.append(EVA_FILE)
    else:
        log(f"  ok: {EVA_FILE}  {size_of(Path(got))}")

    # ---- confound check -----------------------------------------------------
    log("=== 6g. antelopev2 provenance check (InfU vs PuLID) ===")
    # Both methods condition on an ArcFace embedding, but each ships its own copy
    # of the weights. If the bytes differ, "InfU vs PuLID" is partly a
    # recogniser-version comparison, which would be a confound worth reporting.
    import hashlib

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    infu_ante = Path(os.environ.get("INSIGHTFACE_HOME", f"{work}/insightface")) / "models" / "antelopev2"
    for f in ("glintr100.onnx", "scrfd_10g_bnkps.onnx"):
        a, b = infu_ante / f, ante / f
        if a.exists() and b.exists():
            ha, hb = sha(a), sha(b)
            verdict = "IDENTICAL" if ha == hb else "*** DIFFERENT -- confound, report this ***"
            log(f"  {f}: infu={ha} pulid={hb}  {verdict}")
        else:
            log(f"  {f}: cannot compare (missing on one side)")

    # ---- summary ------------------------------------------------------------
    log("")
    log(f"=== models/ contents ({models}) ===")
    for p in sorted(models.rglob("*")):
        if p.is_file() or p.is_symlink():
            rel = p.relative_to(models)
            if not str(rel).startswith(".cache"):
                log(f"  {size_of(p):>10}  {rel}")
    log("")
    if missing:
        log(f"!!! {len(missing)} artefact(s) still missing: {', '.join(missing)}")
        log("!!! re-run this script; downloads resume from where they stopped.")
        return 1
    log("all PuLID artefacts staged.")
    log("Next: sbatch.tinygpu slurm/probe_pulid.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())

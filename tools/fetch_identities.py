#!/usr/bin/env python
"""Curate the evaluation identity set from the Face Research Lab London Set.

Why FRLL (figshare 10.6084/m9.figshare.5047666, CC BY 4.0):

  - Participants gave signed consent for their images to be used in research
    "in their original or altered forms and to illustrate research". A photo
    licence covers the photograph; it says nothing about the depicted person.
    This project generates and publishes altered images of these faces, so the
    consent matters more than the licence does.
  - Self-reported age, gender and ethnicity ship with the set, so the strata in
    identities.json come from the participants rather than from someone
    eyeballing photographs.
  - 1350x1350 neutral frontal portraits, far above the 512 px face-crop floor.
  - Not among InfU's stage-1 training corpora (FFHQ, CelebA, CelebV-HQ,
    CelebV-Text, VGGFace2, MillionCelebs, VFHQ, EasyPortrait, CosmicManHQ), so
    the test identities are unseen by InfU and unseen-symmetric with PuLID.

Selection is deterministic: no RNG, so the same set comes back every run.

Usage:
  python tools/fetch_identities.py --inspect          # demographics only
  python tools/fetch_identities.py                    # download + select 15
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "benchmark" / "identities"
CACHE = ROOT / "benchmark" / ".frll_cache"

INFO_URL = "https://ndownloader.figshare.com/files/27397184"
ZIP_URL = "https://ndownloader.figshare.com/files/8541961"   # neutral_front.zip
DOI = "10.6084/m9.figshare.5047666"
CITATION = ("DeBruine, L., & Jones, B. (2017). Face Research Lab London Set. "
            "figshare. https://doi.org/10.6084/m9.figshare.5047666. CC BY 4.0.")

UA = {"User-Agent": "Mozilla/5.0 (prl26-infiniteyou benchmark builder)"}


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {dest.name} ...")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return dest


def band_of(age: int, edges) -> str:
    for lo, hi, label in edges:
        if lo <= age <= hi:
            return label
    return edges[-1][2]


def parse_bands(spec: str):
    """'18-24,25-34,35+' -> [(18,24,'18-24'), (25,34,'25-34'), (35,200,'35+')]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part.endswith("+"):
            out.append((int(part[:-1]), 200, part))
        else:
            lo, hi = part.split("-")
            out.append((int(lo), int(hi), part))
    return out


def load_rows(edges):
    info = fetch(INFO_URL, CACHE / "london_faces_info.csv")
    rows = []
    with open(info, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            age = r.get("face_age", "").strip()
            if not age.isdigit():
                continue                       # a few records have no age
            rows.append({
                "face_id": r["face_id"].strip(),
                "age": int(age),
                "gender": r["face_gender"].strip().lower(),
                "eth": r["face_eth"].strip().lower(),
                "band": band_of(int(age), edges),
            })
    return rows


def report(rows, edges):
    print(f"\nFRLL: {len(rows)} usable records (age recorded)")
    ages = sorted(r["age"] for r in rows)
    print(f"  age   min={ages[0]}  median={ages[len(ages)//2]}  max={ages[-1]}")
    for field in ("gender", "eth", "band"):
        print(f"  {field:<7}", dict(sorted(Counter(r[field] for r in rows).items(),
                                           key=lambda kv: -kv[1])))
    print("\n  gender x ethnicity")
    c = Counter((r["gender"], r["eth"]) for r in rows)
    for k in sorted(c):
        print(f"    {k[0]:<8} {k[1]:<14} {c[k]}")
    print("\n  gender x age band")
    c = Counter((r["gender"], r["band"]) for r in rows)
    for k in sorted(c):
        print(f"    {k[0]:<8} {k[1]:<10} {c[k]}")


def select(rows, gender: str, n: int, rotate: int):
    """Round-robin across (ethnicity, age band), rotating the start.

    Rotation matters: taking the first n strata from a sorted list silently
    collapses the selection onto whichever category sorts first, which is how
    an earlier stratified sampler in this repo produced 15F/9M when asked for
    12/12.
    """
    pool = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["face_id"]):
        if r["gender"] == gender:
            pool[(r["eth"], r["band"])].append(r)
    keys = sorted(pool)
    if not keys:
        raise SystemExit(f"no candidates for gender={gender}")
    chosen, i = [], 0
    while len(chosen) < n:
        if i > len(keys) * 50:
            raise SystemExit(f"only found {len(chosen)} of {n} for {gender}")
        k = keys[(i + rotate) % len(keys)]
        if pool[k]:
            chosen.append(pool[k].pop(0))
        i += 1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="Report demographics and exit; download nothing large.")
    ap.add_argument("--female", type=int, default=8)
    ap.add_argument("--male", type=int, default=7)
    ap.add_argument("--age-bands", default="18-24,25-34,35+",
                    help="FRLL is 18-54 and skews young; 51+ is effectively empty.")
    ap.add_argument("--part2", type=int, default=8,
                    help="Size of the Part 2 subset, gender balanced.")
    a = ap.parse_args()

    edges = parse_bands(a.age_bands)
    rows = load_rows(edges)
    report(rows, edges)
    if a.inspect:
        print("\n(--inspect: nothing downloaded or written)")
        return 0

    picks = select(rows, "female", a.female, 0) + select(rows, "male", a.male, 1)

    print("\ndownloading neutral_front.zip (32 MB) ...")
    zpath = fetch(ZIP_URL, CACHE / "neutral_front.zip")
    DEST.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        by_id = {}
        for n in names:
            stem = Path(n).stem
            by_id.setdefault(stem.split("_")[0], n)

        out = []
        for idx, r in enumerate(picks, 1):
            src = by_id.get(r["face_id"])
            if src is None:
                raise SystemExit(f"no image in zip for face_id {r['face_id']}")
            iid = f"id{idx:02d}"
            dst = DEST / f"{iid}.jpg"
            with z.open(src) as fh, open(dst, "wb") as o:
                shutil.copyfileobj(fh, o)
            out.append({
                "iid": iid, "gender": r["gender"], "age_band": r["band"],
                "age": r["age"], "ethnicity": r["eth"],
                "path": f"benchmark/identities/{iid}.jpg",
                "source": f"FRLL face_id {r['face_id']} (neutral_front), {DOI}",
                "licence": "CC BY 4.0",
                "status": "pending",     # validate_identities.py promotes to ready
                "part2": False,
            })

    # Part 2 subset: gender balanced, spread across the chosen strata
    half = a.part2 // 2
    for g in ("female", "male"):
        picked = [o for o in out if o["gender"] == g]
        step = max(1, len(picked) // half)
        for o in picked[::step][:half]:
            o["part2"] = True

    (ROOT / "benchmark" / "identities.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")

    lic = DEST / "LICENSES.md"
    lines = ["# Identity image provenance", "",
             f"All images from the Face Research Lab London Set, {DOI}, CC BY 4.0.",
             "", f"> {CITATION}", "",
             "Participants gave signed consent for their images to be used in",
             "lab-based and web-based studies in their original or altered forms",
             "and to illustrate research.", "",
             "| file | FRLL id | gender | age | ethnicity |", "|---|---|---|---|---|"]
    for o in out:
        fid = o["source"].split()[2]
        lines.append(f"| {o['iid']}.jpg | {fid} | {o['gender']} | {o['age']} | {o['ethnicity']} |")
    lic.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {len(out)} identities to {DEST.relative_to(ROOT)}")
    print("  gender   ", dict(Counter(o["gender"] for o in out)))
    print("  age band ", dict(Counter(o["age_band"] for o in out)))
    print("  ethnicity", dict(Counter(o["ethnicity"] for o in out)))
    print("  part2    ", dict(Counter(o["gender"] for o in out if o["part2"])))
    print("\nNext: python tools/validate_identities.py --mark-ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the local end-to-end test fixture.

Pulls the two example portraits shipped with the InfiniteYou repository and
builds a tiny manifest from them. These are test fixtures only -- they are not
the evaluation identity set, which must be curated separately under the
contamination and licensing rules in benchmark/GENERATION.md.

Usage:  python tools/make_fixtures.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
IDDIR = FIX / "identities"

RAW = "https://raw.githubusercontent.com/bytedance/InfiniteYou/main/assets/examples"
SOURCES = [
    ("id_t01", "woman.jpg", "female"),
    ("id_t02", "man.jpg", "male"),
]


def fetch():
    IDDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid, fname, gender in SOURCES:
        dest = IDDIR / f"{iid}.jpg"
        if not dest.exists():
            url = f"{RAW}/{fname}"
            print(f"  fetching {url}")
            urllib.request.urlretrieve(url, dest)
        rows.append({
            "iid": iid, "gender": gender, "age_band": "unknown",
            "skin_tone": "unknown",
            "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "source": "bytedance/InfiniteYou assets/examples (test fixture only)",
            "licence": "see upstream repository",
            "status": "ready", "part2": True,
        })
        print(f"  {dest.name}  {dest.stat().st_size/1024:.0f} KB")
    (FIX / "identities_test.json").write_text(json.dumps(rows, indent=1))
    return rows


def build_manifests():
    """Tiny Part 1 and Part 2 manifests: a handful of cells covering every branch."""
    sys.path.insert(0, str(ROOT))
    from src.manifest import derive_seed

    prompts = json.load(open(ROOT / "benchmark" / "prompts_200.json", encoding="utf-8"))
    style = json.load(open(ROOT / "benchmark" / "prompts_style.json", encoding="utf-8"))
    ids = json.load(open(FIX / "identities_test.json", encoding="utf-8"))

    # 3 prompts per gender, spanning face sizes so FMI hits both the normal and
    # the face-dominates-frame branch
    def take(rows, gender, n, key="pid"):
        want = ["closeup", "waist", "full"]
        out = []
        for fs in want:
            for r in rows:
                if r["gender"] == gender and r["face_size"] == fs and r not in out:
                    out.append(r)
                    break
        return out[:n]

    import pandas as pd

    def make(rows, key, path, styles=False):
        recs = []
        for p in rows:
            for i in ids:
                if p["gender"] != i["gender"]:
                    continue
                cell = f'{i["iid"]}|{p[key]}'
                sk = f'{i["iid"]}|{p.get("base_pid", p[key])}'
                rec = {"cell": cell, "iid": i["iid"], "pid": p[key],
                       "prompt": p["text"], "id_path": i["path"],
                       "seed": derive_seed(sk), "seed_key": sk,
                       "gender": p["gender"], "face_size": p["face_size"],
                       "complexity": p["complexity"], "length": p["length"]}
                if styles:
                    rec.update(style=p["style"], style_phrase=p["style_phrase"],
                               base_pid=p["base_pid"])
                recs.append(rec)
        pd.DataFrame(recs).to_parquet(FIX / path, index=False)
        print(f"  {path}: {len(recs)} cells")
        return recs

    p1 = take(prompts, "female", 3) + take(prompts, "male", 3)
    make(p1, "pid", "manifest_test.parquet")

    # two base prompts x all four styles, so the paired delta can be computed
    bases = sorted({s["base_pid"] for s in style})[:2]
    s1 = [s for s in style if s["base_pid"] in bases]
    make(s1, "spid", "manifest_test_style.parquet", styles=True)


if __name__ == "__main__":
    print("fixture identities:")
    fetch()
    print("fixture manifests:")
    build_manifests()
    print(f"\nfixtures in {FIX.relative_to(ROOT)}")

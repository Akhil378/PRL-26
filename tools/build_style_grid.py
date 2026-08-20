#!/usr/bin/env python
"""Build the Part 2 stylization grid from the Part 1 benchmark.

Selects a stratified subset of base prompts and crosses it with four style
conditions. Every condition appends a style clause of comparable length --
including the photoreal control -- so that prompt length is not confounded
with style. The bare style phrase is stored separately because the Face
Masking Index scores image crops against the style alone.

Usage:  python tools/build_style_grid.py [--n-base 24]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark" / "prompts_200.json"
OUT = ROOT / "benchmark" / "prompts_style.json"

# style key -> (clause appended to the base prompt, bare phrase for FMI/style scoring)
STYLES = {
    "photoreal":   ("as a natural, high-resolution colour photograph",
                    "a natural colour photograph"),
    "oil_painting": ("in the style of a 19th-century oil painting",
                     "a 19th-century oil painting"),
    "render_3d":   ("as a stylised 3D character render with soft studio lighting",
                    "a stylised 3D character render"),
    "watercolour": ("as a loose watercolour painting on textured paper",
                    "a loose watercolour painting"),
}


GENDERS = ["female", "male"]
LENGTHS = ["short", "medium", "long"]


def select_base(rows, n_base):
    """Deterministic stratified pick.

    Gender and length are balanced exactly by construction (n_base must be
    divisible by 6). Within each gender x length cell, face_size and
    complexity are spread by a rotating round-robin, with the rotation offset
    varying per cell so the same face_size does not lead every time.
    """
    if n_base % (len(GENDERS) * len(LENGTHS)):
        raise ValueError(f"--n-base must be divisible by {len(GENDERS)*len(LENGTHS)}")
    per_cell = n_base // (len(GENDERS) * len(LENGTHS))

    picked = []
    for gi, g in enumerate(GENDERS):
        for li, lg in enumerate(LENGTHS):
            pool = defaultdict(list)
            for r in sorted(rows, key=lambda x: x["pid"]):
                if r["gender"] == g and r["length"] == lg:
                    pool[(r["face_size"], r["complexity"])].append(r)
            keys = sorted(pool)
            if not keys:
                raise RuntimeError(f"no prompts for {g}/{lg}")
            rot = (gi * len(LENGTHS) + li) % len(keys)
            order = keys[rot:] + keys[:rot]

            chosen, i = [], 0
            while len(chosen) < per_cell:
                if i >= len(order) * 20:
                    raise RuntimeError(f"pool exhausted for {g}/{lg}")
                k = order[i % len(order)]
                if pool[k]:
                    chosen.append(pool[k].pop(0))
                i += 1
            picked += chosen
    return sorted(picked, key=lambda x: x["pid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=36)
    args = ap.parse_args()

    rows = json.load(open(SRC, encoding="utf-8"))
    base = select_base(rows, args.n_base)

    grid = []
    for b in base:
        for style, (clause, phrase) in STYLES.items():
            grid.append({
                "spid": f'{b["pid"]}_{style}',
                "base_pid": b["pid"],
                "base_text": b["text"],
                "style": style,
                "style_phrase": phrase,
                "text": f'{b["text"]}, {clause}',
                "gender": b["gender"],
                "face_size": b["face_size"],
                "complexity": b["complexity"],
                "length": b["length"],
            })

    OUT.write_text(json.dumps(grid, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"base prompts: {len(base)}   styles: {len(STYLES)}   grid rows: {len(grid)}\n")
    for f in ["gender", "face_size", "complexity", "length"]:
        print(f"  {f:<12}", dict(Counter(b[f] for b in base)))
    print()
    gc = Counter(b["gender"] for b in base)
    for n_ids, f_ids in [(8, 4)]:
        m_ids = n_ids - f_ids
        per_style = gc["female"] * f_ids + gc["male"] * m_ids
        cells = per_style * len(STYLES)
        print(f"main grid with {n_ids} identities ({f_ids}F/{m_ids}M), gender-matched:")
        print(f"  {per_style} cells/style x {len(STYLES)} styles = {cells} per method")
        print(f"  x2 methods = {cells*2} images  (~{cells*2*30/3600:.1f} GPU-h at 30 s/img)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze the (identity x prompt) job list that every generation run reads.

Pairing follows the paper's scheme: each prompt carries a gender tag and is
paired only with identities of that gender, which is what reduces the full
cross product to roughly half.

The seed is derived from the cell key rather than drawn at random, so every
method generates the same cell from the same initial noise. That makes the
InfU/PuLID comparison paired at the level of the latent, removes a large
variance component, and is exactly reproducible from this file alone.

Usage:
  python src/manifest.py --prompts benchmark/prompts_200.json \\
                         --out benchmark/manifest_repro.parquet
  python src/manifest.py --prompts benchmark/prompts_style.json \\
                         --out benchmark/manifest_style.parquet --part2-only
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def derive_seed(cell: str) -> int:
    """Stable 31-bit seed from the cell key. Never change this function."""
    return int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16) % (2**31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--identities", default="benchmark/identities.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part2-only", action="store_true",
                    help="Restrict to identities flagged part2 (the Part 2 subset).")
    ap.add_argument("--allow-pending", action="store_true",
                    help="Build even though identity images are not yet curated.")
    args = ap.parse_args()

    prompts = json.load(open(ROOT / args.prompts, encoding="utf-8"))
    ids = json.load(open(ROOT / args.identities, encoding="utf-8"))

    if args.part2_only:
        ids = [i for i in ids if i.get("part2")]

    pending = [i["iid"] for i in ids if i.get("status") != "ready"]
    if pending and not args.allow_pending:
        print(f"ERROR: {len(pending)} identities are not marked ready: "
              f"{', '.join(pending[:6])}{'...' if len(pending) > 6 else ''}")
        print("Curate the images, run tools/validate_identities.py, then rerun.")
        print("Pass --allow-pending to build a dry-run manifest anyway.")
        return 1

    # the prompt id field differs between Part 1 (pid) and Part 2 (spid)
    key = "spid" if prompts and "spid" in prompts[0] else "pid"

    rows = []
    for p in prompts:
        for i in ids:
            if p["gender"] != i["gender"]:
                continue
            cell = f'{i["iid"]}|{p[key]}'
            # Seed keys off the BASE prompt, not the style variant, so that every
            # style condition for one (identity, base prompt) starts from the same
            # latent. Without this the photoreal control and its stylised siblings
            # differ by noise as well as by style, and the within-cell delta the
            # paired analysis relies on picks up variance it does not need.
            seed_key = f'{i["iid"]}|{p.get("base_pid", p[key])}'
            row = {
                "cell": cell,
                "iid": i["iid"],
                "pid": p[key],
                "prompt": p["text"],
                "id_path": i["path"],
                "seed": derive_seed(seed_key),
                "seed_key": seed_key,
                "gender": p["gender"],
                "face_size": p["face_size"],
                "complexity": p["complexity"],
                "length": p["length"],
            }
            for extra in ("style", "style_phrase", "base_pid", "age_hint"):
                if extra in p:
                    row[extra] = p[extra]
            rows.append(row)

    assert len({r["cell"] for r in rows}) == len(rows), "duplicate cell keys"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
    except ImportError:
        out = out.with_suffix(".csv")
        import csv
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("(pandas unavailable - wrote CSV instead)")

    print(f"{len(rows)} cells  |  {len({r['iid'] for r in rows})} identities"
          f"  x  {len({r['pid'] for r in rows})} prompts")
    print("  by gender ", dict(Counter(r["gender"] for r in rows)))
    if "style" in rows[0]:
        print("  by style  ", dict(Counter(r["style"] for r in rows)))
    est = len(rows) * 30 / 3600
    print(f"  ~{est:.1f} GPU-h per method at 30 s/img  ({est*2:.1f} h for two methods)")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

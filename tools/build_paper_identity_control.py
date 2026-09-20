#!/usr/bin/env python
"""Stage the InfiniteYou authors' own example portraits as a control identity set.

Why this control exists
-----------------------
Part 1 measures InfU at roughly twice the published ID Loss (0.418 against
0.209). The report attributes that to benchmark composition, and the
stratification supports part of the claim: full-body prompts score 0.525 and are
34% of the benchmark. But the *close-up* stratum still scores 0.341, which is
63% above the published overall mean. Composition explains the spread; it does
not explain the level.

The remaining candidate is the identity set. This benchmark deliberately draws
identities from FRLL, chosen to sit outside every corpus InfU was pretrained on.
The paper's own 15 identities are never named, and may well sit inside them.

This control separates the two explanations with the only identities whose
provenance is certain: the two portraits the authors ship in their own
repository, which are the images their demo and figures use.

  - ID Loss falls toward the published value -> the gap is the identity set, and
    the published numbers depend on identity provenance. That is a finding.
  - ID Loss stays near 0.42 -> composition and identity set are both ruled out,
    and something in this implementation differs. That is also a finding, and a
    more uncomfortable one, which is exactly why it is worth measuring.

Everything else is held fixed: same 200 prompts, same gender-matched pairing,
same seed derivation, same resolution, steps, guidance and precision as Part 1.

Usage:  python tools/build_paper_identity_control.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "benchmark" / "identities_paper"
IDS_JSON = ROOT / "benchmark" / "identities_paper.json"

# The upstream checkout staged by tools/setup_cluster.sh.
INFU_REPO_CANDIDATES = [
    Path("/home/woody/rlvl/rlvl159v/prl26/InfiniteYou"),
    ROOT.parent / "InfiniteYou",
]

# (iid, upstream filename, gender). Gender is what the pairing needs; the
# authors' filenames assert it.
SOURCES = [
    ("pid_w", "woman.jpg", "female"),
    ("pid_m", "man.jpg", "male"),
]


def find_repo() -> Path:
    for c in INFU_REPO_CANDIDATES:
        if (c / "assets" / "examples").is_dir():
            return c
    raise SystemExit(
        "InfiniteYou checkout not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in INFU_REPO_CANDIDATES))


def main() -> int:
    repo = find_repo()
    src_dir = repo / "assets" / "examples"
    DEST.mkdir(parents=True, exist_ok=True)

    rows = []
    for iid, fname, gender in SOURCES:
        src = src_dir / fname
        if not src.is_file():
            raise SystemExit(f"missing upstream example: {src}")
        dst = DEST / f"{iid}.jpg"
        shutil.copyfile(src, dst)
        rows.append({
            "iid": iid,
            "gender": gender,
            "age_band": "unknown",
            "ethnicity": "unknown",
            "path": f"benchmark/identities_paper/{iid}.jpg",
            "source": f"bytedance/InfiniteYou assets/examples/{fname}",
            "licence": "see upstream repository",
            "status": "ready",
            "part2": False,
        })
        print(f"  {iid}  <- {fname}  ({dst.stat().st_size/1024:.0f} KB)")

    IDS_JSON.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {IDS_JSON.relative_to(ROOT)}")
    print("\nNext:")
    print("  python src/manifest.py --prompts benchmark/prompts_200.json \\")
    print("      --identities benchmark/identities_paper.json \\")
    print("      --out benchmark/manifest_paperid.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())

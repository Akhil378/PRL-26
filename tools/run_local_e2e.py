#!/usr/bin/env python
"""One command that proves the whole pipeline works before any GPU time is spent.

Runs the real code path end to end -- manifest, sharded generation, resume,
scoring, FMI, aggregation -- substituting only what a laptop cannot run: the
diffusion model (src/gen_stub.py) and insightface (--id-backend stub). Every
other line executed here is the same line that runs on the cluster.

Usage:  python tools/run_local_e2e.py [--clean]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "out"
PY = sys.executable

STEPS = [
    ("unit: FMI crop geometry", [PY, "tests/test_crops.py"]),
    ("unit: shard selection and resume", [PY, "tests/test_runner.py"]),
    ("build fixtures", [PY, "tools/make_fixtures.py"]),
    ("generate Part 1 (shard 0 of 2)", [PY, "src/gen_stub.py",
        "--manifest", "tests/fixtures/manifest_test.parquet",
        "--out", "tests/fixtures/out/stub_p1", "--shard", "0", "--num-shards", "2"]),
    ("generate Part 1 (shard 1 of 2)", [PY, "src/gen_stub.py",
        "--manifest", "tests/fixtures/manifest_test.parquet",
        "--out", "tests/fixtures/out/stub_p1", "--shard", "1", "--num-shards", "2"]),
    ("resume must be a no-op", [PY, "src/gen_stub.py",
        "--manifest", "tests/fixtures/manifest_test.parquet",
        "--out", "tests/fixtures/out/stub_p1"]),
    ("generate Part 2 style grid", [PY, "src/gen_stub.py",
        "--manifest", "tests/fixtures/manifest_test_style.parquet",
        "--out", "tests/fixtures/out/stub_p2"]),
    ("score Part 1", [PY, "src/score_all.py",
        "--manifest", "tests/fixtures/manifest_test.parquet",
        "--images", "tests/fixtures/out/stub_p1", "--method", "stub_p1",
        "--out", "tests/fixtures/out/scores_p1.parquet",
        "--id-backend", "stub", "--clip-backbones", "B32", "--skip-pickscore"]),
    ("score Part 2 (with FMI)", [PY, "src/score_all.py",
        "--manifest", "tests/fixtures/manifest_test_style.parquet",
        "--images", "tests/fixtures/out/stub_p2", "--method", "stub_p2",
        "--out", "tests/fixtures/out/scores_p2.parquet",
        "--id-backend", "stub", "--clip-backbones", "B32", "--skip-pickscore"]),
    ("scoring resume must be a no-op", [PY, "src/score_all.py",
        "--manifest", "tests/fixtures/manifest_test.parquet",
        "--images", "tests/fixtures/out/stub_p1", "--method", "stub_p1",
        "--out", "tests/fixtures/out/scores_p1.parquet",
        "--id-backend", "stub", "--clip-backbones", "B32", "--skip-pickscore"]),
    ("FMI positive/negative control", [PY, "tests/test_fmi_control.py"]),
]

NOISE = ("Loading weights", "unauthenticated requests", "[ WARN:", "warnings.warn")


def run(label, cmd):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip() and not any(t in line for t in NOISE):
            print("  " + line)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="Delete generated fixtures first (keeps downloads).")
    a = ap.parse_args()
    if a.clean and OUT.exists():
        shutil.rmtree(OUT)
        print(f"removed {OUT.relative_to(ROOT)}")

    failed = []
    for label, cmd in STEPS:
        if run(label, cmd) != 0:
            failed.append(label)

    print(f"\n{'='*70}")
    if failed:
        print(f"{len(failed)}/{len(STEPS)} STEPS FAILED:")
        for f in failed:
            print("  -", f)
        print("\nDo not submit to the cluster until this is green.")
        return 1
    print(f"all {len(STEPS)} steps passed - the pipeline runs end to end.")
    print("\nStill unexercised locally, and only verifiable on the cluster:")
    print("  - the InfUFluxPipeline and PuLID FluxGenerator calls themselves")
    print("  - insightface antelopev2 / buffalo_l (no Windows wheel)")
    print("  - PickScore (CLIP-H) and CLIP ViT-L/14")
    print("Cover those with a --limit 20 run before launching a full array.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

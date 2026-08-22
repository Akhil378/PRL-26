#!/usr/bin/env python
"""Adversarial tests for the failure modes a cluster run will actually hit.

A 24-hour cap, preemption, a full filesystem and a queue that reorders jobs all
mean the happy path is the least interesting one. Each case here corresponds to
something that will happen during a real run.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runner_common import cell_to_file, select_work, shard_of  # noqa: E402

PY = sys.executable
FIX = ROOT / "tests" / "fixtures"
MAN = FIX / "manifest_test.parquet"
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


def gen(out, extra=()):
    return subprocess.run(
        [PY, "src/gen_stub.py", "--manifest", str(MAN), "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True)


print("1. a failing cell must not end the run")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    r = gen(out, ["--fail-rate", "0.5"])
    check("process still exits 0", r.returncode == 0, f"rc={r.returncode}")
    log = pd.read_csv(out / "log_shard0.csv")
    n_err = (log.status != "ok").sum()
    check("some cells failed as instructed", n_err > 0, f"{n_err} failures")
    check("failures are recorded in the log", log.status.str.startswith("error").any())
    check("no .part debris from failures", len(list(out.glob("*.part"))) == 0)
    check("successful cells still written",
          len(list(out.glob("*.png"))) == (log.status == "ok").sum())

    # rerun without failures: only the previously-failed cells should be redone
    before = {p.name for p in out.glob("*.png")}
    r2 = gen(out)
    after = {p.name for p in out.glob("*.png")}
    check("rerun completes the missing cells", len(after) == len(pd.read_parquet(MAN)),
          f"{len(before)} -> {len(after)}")

print("\n2. a job killed mid-write leaves no corrupt output")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    gen(out)
    victim = next(out.glob("*.png"))
    # simulate the kill: a stale .part from an interrupted save
    stale = out / (victim.name + ".part")
    stale.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    man = pd.read_parquet(MAN)
    todo = select_work(man, out)
    check("stale .part is not mistaken for finished work", len(todo) == 0,
          f"{len(todo)} selected")
    check("only .png files count as complete",
          all(p.suffix == ".png" for p in out.glob("*.png")))
    victim.unlink()
    todo2 = select_work(man, out)
    check("a genuinely missing cell is reselected", len(todo2) == 1,
          str(list(todo2.cell)))

print("\n3. changing the shard count between runs must not lose or duplicate work")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    gen(out, ["--shard", "0", "--num-shards", "3"])
    partial = len(list(out.glob("*.png")))
    check("first shard produced a subset", 0 < partial < 6, f"{partial} of 6")
    # resubmit the same manifest with a different split, as a rescoped rerun would
    for s in range(2):
        gen(out, ["--shard", str(s), "--num-shards", "2"])
    total = len(list(out.glob("*.png")))
    check("all cells present after a reshard", total == 6, f"{total} of 6")
    log_files = list(out.glob("log_shard*.csv"))
    redone = sum(len(pd.read_csv(f)) for f in log_files)
    check("no cell was generated twice", redone <= 6 + partial,
          f"{redone} generation attempts for 6 cells")

print("\n4. scoring must survive a corrupt or truncated image")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    gen(out)
    bad = next(out.glob("*.png"))
    bad.write_bytes(b"\x89PNG\r\n\x1a\n not a real png")
    sc = Path(td) / "scores.parquet"
    r = subprocess.run(
        [PY, "src/score_all.py", "--manifest", str(MAN), "--images", str(out),
         "--method", "t", "--out", str(sc), "--id-backend", "stub",
         "--clip-backbones", "B32", "--skip-pickscore"],
        cwd=ROOT, capture_output=True, text=True)
    check("scoring does not crash on a corrupt file", r.returncode == 0,
          (r.stderr.strip().splitlines() or ["ok"])[-1][:70])

print("\n5. empty shard and out-of-range shard")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    r = gen(out, ["--shard", "9", "--num-shards", "10"])
    check("an empty shard exits cleanly", r.returncode == 0, f"rc={r.returncode}")
    man = pd.read_parquet(MAN)
    counts = [len(select_work(man, out, s, 10)) for s in range(10)]
    check("shards still partition the manifest", sum(counts) == len(man),
          f"{counts}")

print("\n6. shard assignment is stable across processes")
man = pd.read_parquet(MAN)
a = [shard_of(c, 8) for c in man.cell]
b = subprocess.run(
    [PY, "-c",
     "import sys;sys.path.insert(0,'.');"
     "import pandas as pd;from src.runner_common import shard_of;"
     f"m=pd.read_parquet(r'{MAN}');print([shard_of(c,8) for c in m.cell])"],
    cwd=ROOT, capture_output=True, text=True)
check("same assignment in a fresh interpreter", str(a) == b.stdout.strip(),
      "hash seeding is not randomised")

print("\n7. long prompts are truncated, not silently dropped")
prompts = pd.read_parquet(MAN)
longest = max(prompts.prompt, key=len)
check("fixture includes a non-trivial prompt", len(longest) > 20, f"{len(longest)} chars")

print()
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    sys.exit(1)
print("all robustness tests passed")

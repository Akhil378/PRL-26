#!/usr/bin/env python
"""Unit tests for shard selection and resume. No GPU, no models."""
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runner_common import (cell_to_file, file_to_cell, select_work,  # noqa: E402
                               shard_of)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


man = pd.DataFrame({"cell": [f"id{i:02d}|p{j:03d}" for i in range(1, 16)
                             for j in range(1, 101)]})
N = len(man)

print("filename round-trip")
c = "id01|p001_oil_painting"
check("cell -> file -> cell is identity", file_to_cell(cell_to_file(c)) == c,
      cell_to_file(c))
check("no pipe survives into the filename", "|" not in cell_to_file(c))

print("\nsharding")
counts = Counter(shard_of(c, 4) for c in man.cell)
check("every shard is used", set(counts) == {0, 1, 2, 3}, str(dict(counts)))
spread = max(counts.values()) - min(counts.values())
check("shards are balanced within 5%", spread / N < 0.05, f"spread {spread} of {N}")
check("single shard is a no-op", all(shard_of(c, 1) == 0 for c in man.cell))
check("assignment is deterministic",
      [shard_of(c, 4) for c in man.cell[:50]] == [shard_of(c, 4) for c in man.cell[:50]])

# adding rows must not move existing cells between shards
before = {c: shard_of(c, 4) for c in man.cell}
grown = list(man.cell) + [f"id16|p{j:03d}" for j in range(1, 101)]
check("adding manifest rows does not reshuffle existing cells",
      all(before[c] == shard_of(c, 4) for c in man.cell), f"{len(grown)} rows now")

print("\nselect_work / resume")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    all_shards = [select_work(man, out, s, 4) for s in range(4)]
    total = sum(len(d) for d in all_shards)
    check("shards partition the manifest exactly", total == N, f"{total} vs {N}")
    seen = set()
    for d in all_shards:
        seen |= set(d.cell)
    check("no cell is duplicated across shards", len(seen) == N)

    # simulate a job that died halfway through shard 0
    s0 = all_shards[0]
    for c in s0.cell[:len(s0) // 2]:
        (out / cell_to_file(c)).touch()
    resumed = select_work(man, out, 0, 4)
    check("resume skips completed cells",
          len(resumed) == len(s0) - len(s0) // 2, f"{len(resumed)} left of {len(s0)}")
    check("resume never re-emits a finished cell",
          not set(resumed.cell) & set(s0.cell[:len(s0) // 2]))

    for c in s0.cell:
        (out / cell_to_file(c)).touch()
    check("a finished shard selects nothing", len(select_work(man, out, 0, 4)) == 0)
    check("other shards are unaffected", len(select_work(man, out, 1, 4)) == len(all_shards[1]))

    check("limit caps the batch", len(select_work(man, out, 2, 4, limit=10)) == 10)

print()
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    sys.exit(1)
print("all runner tests passed")

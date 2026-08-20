#!/usr/bin/env python
"""Shard selection, resume and logging shared by the generation runners.

TinyGPU caps a job at 24 hours and a full Part 1 run does not fit in one, so
resumability is structural rather than a convenience. Work is split by a stable
hash of the cell key -- not by row position -- so that adding rows to a manifest
never reshuffles which shard an existing cell belongs to.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import pandas as pd


def cell_to_file(cell: str) -> str:
    """Cell key -> output filename. The inverse must stay stable across runs."""
    return cell.replace("|", "__") + ".png"


def file_to_cell(name: str) -> str:
    return Path(name).stem.replace("__", "|")


def shard_of(cell: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    return int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16) % num_shards


def select_work(man: pd.DataFrame, out_dir: Path, shard: int = 0,
                num_shards: int = 1, limit: int = 0) -> pd.DataFrame:
    """Rows this worker should generate: its shard, minus what already exists."""
    df = man.copy()
    if num_shards > 1:
        df = df[df.cell.map(lambda c: shard_of(c, num_shards)) == shard]
    df = df[~df.cell.map(lambda c: (out_dir / cell_to_file(c)).exists())]
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


class RunLog:
    """Append-only per-image log, flushed periodically so a killed job keeps it."""

    def __init__(self, path: Path, flush_every: int = 25):
        self.path = path
        self.flush_every = flush_every
        self.rows = []
        self.t0 = time.time()

    def add(self, cell: str, status: str, seconds: float):
        self.rows.append({"cell": cell, "status": status, "sec": round(seconds, 2)})
        if len(self.rows) % self.flush_every == 0:
            self.flush()

    def flush(self):
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def progress(self, n: int, total: int) -> str:
        el = time.time() - self.t0
        rate = n / el if el else 0
        eta = (total - n) / rate / 60 if rate else float("nan")
        return f"{n}/{total}  {rate*3600:.0f} img/h  eta {eta:.0f} min"

    def summary(self) -> str:
        if not self.rows:
            return "nothing generated"
        df = pd.DataFrame(self.rows)
        ok = df[df.status == "ok"]
        parts = [f"{len(ok)} ok", f"{len(df) - len(ok)} failed"]
        if len(ok):
            parts.append(f"median {ok.sec.median():.1f} s/img")
        return "  |  ".join(parts)


def resolve_model_dir(model_dir: str, subfolder: Optional[str] = None) -> str:
    """Accept either a local path or a HuggingFace repo id."""
    p = Path(model_dir)
    if p.exists():
        return str(p / subfolder) if subfolder else str(p)
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=model_dir)
    return str(Path(local) / subfolder) if subfolder else local

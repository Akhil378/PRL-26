#!/usr/bin/env python
"""Build and validate the reconstructed portrait benchmark.

Reads the hand-authored prompt pool, derives the `length` stratum from the
actual word count (so the tag can never drift from the text), validates the
schema, reports the realised marginals, and writes prompts_200.json.

Usage:  python tools/build_benchmark.py [--check-clip-tokens]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmark" / "prompts_raw.jsonl"
OUT = ROOT / "benchmark" / "prompts_200.json"

# Length bins, in words. Fixed here and reported in the paper's setup section.
SHORT_MAX = 8
MEDIUM_MAX = 20

ALLOWED = {
    "gender": {"female", "male"},
    "face_size": {"closeup", "waist", "full"},
    "complexity": {"plain", "single_object", "busy"},
    "age_hint": {"none", "young", "senior"},
}
REQUIRED = ["pid", "text", *ALLOWED.keys()]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'-]+", text))


def length_bin(n: int) -> str:
    if n <= SHORT_MAX:
        return "short"
    if n <= MEDIUM_MAX:
        return "medium"
    return "long"


def bar(n: int, total: int, width: int = 28) -> str:
    filled = round(width * n / total) if total else 0
    return "#" * filled + "." * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-clip-tokens", action="store_true",
                    help="Report how many prompts exceed CLIP's 77-token limit "
                         "(requires transformers).")
    args = ap.parse_args()

    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []

    seen = set()
    for r in rows:
        for f in REQUIRED:
            if f not in r:
                errors.append(f"{r.get('pid', '?')}: missing field '{f}'")
        for f, allowed in ALLOWED.items():
            if f in r and r[f] not in allowed:
                errors.append(f"{r['pid']}: {f}={r[f]!r} not in {sorted(allowed)}")
        if r.get("pid") in seen:
            errors.append(f"duplicate pid {r['pid']}")
        seen.add(r.get("pid"))
        r["n_words"] = word_count(r["text"])
        r["length"] = length_bin(r["n_words"])

    texts = Counter(r["text"].lower() for r in rows)
    for t, c in texts.items():
        if c > 1:
            errors.append(f"duplicate prompt text ({c}x): {t[:60]}")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print("  -", e)
        return 1

    n = len(rows)
    print(f"benchmark: {n} prompts\n")

    for field in ["gender", "length", "face_size", "complexity", "age_hint"]:
        counts = Counter(r[field] for r in rows)
        print(f"{field}")
        for k in sorted(counts, key=lambda x: -counts[x]):
            v = counts[k]
            print(f"  {k:<14} {v:>4}  {100*v/n:5.1f}%  {bar(v, n)}")
        print()

    print("gender x length")
    print(f"  {'':<10}{'short':>8}{'medium':>8}{'long':>8}")
    for g in sorted(ALLOWED["gender"]):
        row = Counter(r["length"] for r in rows if r["gender"] == g)
        print(f"  {g:<10}{row['short']:>8}{row['medium']:>8}{row['long']:>8}")
    print()

    print("gender x face_size")
    print(f"  {'':<10}{'closeup':>9}{'waist':>8}{'full':>8}")
    for g in sorted(ALLOWED["gender"]):
        row = Counter(r["face_size"] for r in rows if r["gender"] == g)
        print(f"  {g:<10}{row['closeup']:>9}{row['waist']:>8}{row['full']:>8}")
    print()

    w = [r["n_words"] for r in rows]
    w.sort()
    print(f"word count  min={w[0]}  median={w[n//2]}  max={w[-1]}  mean={sum(w)/n:.1f}")

    # Gender-matched pairing yields roughly half the full cross product.
    for n_ids, f_ids in [(15, 8)]:
        m_ids = n_ids - f_ids
        gc = Counter(r["gender"] for r in rows)
        cells = gc["female"] * f_ids + gc["male"] * m_ids
        print(f"\nprojected cells with {n_ids} identities ({f_ids}F/{m_ids}M): "
              f"{cells}   (paper reports 1,497)")

    if args.check_clip_tokens:
        try:
            from transformers import CLIPTokenizerFast
            tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
            over = [(r["pid"], len(tok(r["text"])["input_ids"])) for r in rows
                    if len(tok(r["text"])["input_ids"]) > 77]
            print(f"\nCLIP 77-token limit: {len(over)} prompt(s) will truncate")
            for pid, ln in over[:10]:
                print(f"  {pid}: {ln} tokens")
        except ImportError:
            print("\n(transformers not installed - skipping CLIP token check)")

    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

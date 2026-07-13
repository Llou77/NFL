#!/usr/bin/env python3
"""
merge_backtests.py — combine per-season backtest fragments.

The backtest CI runs one job per test season in parallel
(`pipeline.py --mode backtest --backtest-season S`), each writing
data/predictions/backtest_fragments/backtest_S.json. This script merges the
fragments into the single backtest_results.json the frontend reads, and
copies it to docs/assets/.

Merging starts from the EXISTING backtest_results.json, so if one matrix job
fails, the previous result for that season is kept instead of being wiped.

Stdlib-only. Usage: python3 scripts/merge_backtests.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "data" / "predictions" / "backtest_fragments"
OUT  = ROOT / "data" / "predictions" / "backtest_results.json"
DOCS = ROOT / "docs" / "assets"


def main() -> None:
    merged = {}
    if OUT.exists():
        try:
            merged = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            merged = {}

    fragments = sorted(FRAG.glob("backtest_*.json")) if FRAG.exists() else []
    if not fragments:
        print("no backtest fragments found — nothing to merge", file=sys.stderr)
        sys.exit(1)

    n_new = 0
    for p in fragments:
        try:
            frag = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"skipping corrupt fragment {p.name}: {e}", file=sys.stderr)
            continue
        merged.update(frag)
        n_new += len(frag)
        print(f"  merged {p.name}: {list(frag.keys())}")

    OUT.write_text(json.dumps(merged, indent=2, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, DOCS / "backtest_results.json")
    print(f"backtest_results.json: {len(merged)} season entries "
          f"({n_new} updated from {len(fragments)} fragment file(s))")


if __name__ == "__main__":
    main()

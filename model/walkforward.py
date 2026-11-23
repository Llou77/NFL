#!/usr/bin/env python3
"""
walkforward.py — one walk-forward fold per invocation.

Fold (S, W): build features for seasons S-W .. S from the committed
data/frozen/ layer only (no network), train on S-W .. S-1 with the
production code, predict every game of S and score it against the actual
results and the Vegas line (evaluate.compute_metrics).

Never tune anything on these results: aggregating walk-forward folds and
optimising on them turns the test seasons into training data.

Usage:  python model/walkforward.py --test-season 2016 --window 4
Output: data/predictions/walkforward/wf_w{W}_{S}.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("walkforward")

FROZEN = ROOT / "data" / "frozen"
RAW    = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "predictions" / "walkforward"


def prepare_tables(seasons: list[int]) -> None:
    """Materialise data/raw/ for this fold purely from data/frozen/."""
    import pandas as pd
    from data_loader import concat_seasons

    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("schedules", "ngs_passing", "ngs_rushing", "ngs_receiving"):
        shutil.copy2(FROZEN / f"{name}.parquet", RAW / f"{name}.parquet")
    for table, years in (("player_stats_weekly", seasons),
                         ("rosters_weekly", [seasons[0] - 1] + seasons)):
        frames = [pd.read_parquet(FROZEN / f"{table}_{s}.parquet")
                  for s in years if (FROZEN / f"{table}_{s}.parquet").exists()]
        concat_seasons(frames).to_parquet(RAW / f"{table}.parquet", index=False)
    missing = [s for s in seasons if not (FROZEN / f"pbp_agg_{s}.parquet").exists()]
    if missing:
        sys.exit(f"Missing frozen PBP aggregates for {missing} — run the freeze workflow.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--window", type=int, default=4, help="training seasons before the test season")
    args = ap.parse_args()
    S, W = args.test_season, args.window
    seasons = list(range(S - W, S + 1))
    log.info("FOLD: test=%d window=%d seasons=%s", S, W, seasons)

    prepare_tables(seasons)
    from evaluate import run_backtest
    from feature_engineering import build_all_features
    from feature_h2h import build_h2h_features

    game_df = build_h2h_features(build_all_features(seasons=seasons))
    res = run_backtest(game_df, S, n_train_seasons=W)
    res.update({"window": W, "covid_affected": 2020 in seasons})

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"wf_w{W}_{S}.json"
    out.write_text(json.dumps({f"{S}_w{W}": res}, indent=2, default=str))
    log.info("FOLD DONE %d (w=%d) → %s", S, W, out)


if __name__ == "__main__":
    main()

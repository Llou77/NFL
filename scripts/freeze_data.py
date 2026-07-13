#!/usr/bin/env python3
"""
freeze_data.py — build the committed data/frozen/ layer.

WHY: between seasons (and for every already-finished season) the historical
nflverse data can never change again. Re-downloading it on every pipeline run
wastes time and — worse — breaks the pipeline whenever nflverse renames a
release file (this actually happened: player_stats → stats_player in 2025).
This script downloads everything for CLOSED seasons exactly once and commits
it to the repo, so normal pipeline runs read closed-season data from disk
with ZERO network access.

WHAT IT STORES (data/frozen/):
  - per-season tables for every closed season in the model window
    (player stats, rosters, snap counts, injuries, depth charts, FTN,
    PFR advanced stats) under the same keys data_loader uses;
  - full-history "static" tables (NGS, draft, combine, officials, lines,
    team descriptors, contracts, players, id map, schedule snapshot);
  - pbp_agg_{season}.parquet — the per-season TEAM-GAME AGGREGATE of raw
    play-by-play, built with feature_engineering._aggregate_pbp_core (the
    exact function the live pipeline uses). Raw PBP itself is ~25 MB/season
    and is deliberately NOT committed; the aggregate is ~0.2 MB/season.

WHEN TO RUN: once now (offseason), then once a year after the Super Bowl
(the freeze_data workflow has a yearly cron). A season is considered CLOSED
when it is strictly before data_loader.CURRENT_SEASON (Jan/Feb still count
as the previous season, so a just-finished season freezes correctly).

Usage:  python scripts/freeze_data.py [--force]
        --force  re-download and overwrite existing frozen files
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("freeze")

from data_loader import (  # noqa: E402
    BASE, RAW_GH, CURRENT_SEASON, ALL_SEASONS, _fetch,
)

FROZEN = ROOT / "data" / "frozen"
FROZEN.mkdir(parents=True, exist_ok=True)

CLOSED_SEASONS = [s for s in ALL_SEASONS if s < CURRENT_SEASON]

# Hard safety limits for a git repo
MAX_FILE_MB  = 90     # GitHub rejects files >100 MB — stay well below
WARN_TOTAL_MB = 300

# ── what to freeze ─────────────────────────────────────────────────────────────

# (key template, [url templates in fallback order], format)
PER_SEASON = [
    ("player_stats_weekly_{}",
     [f"{BASE}/stats_player/stats_player_week_{{0}}.parquet",
      f"{BASE}/player_stats/player_stats_{{0}}.parquet"], "parquet"),
    ("player_stats_season_{}",
     [f"{BASE}/stats_player/stats_player_reg_{{0}}.parquet",
      f"{BASE}/player_stats/player_stats_season_{{0}}.parquet"], "parquet"),
    ("rosters_{}",        [f"{BASE}/rosters/roster_{{0}}.parquet"], "parquet"),
    ("rosters_weekly_{}", [f"{BASE}/weekly_rosters/roster_weekly_{{0}}.parquet"], "parquet"),
    ("snap_counts_{}",    [f"{BASE}/snap_counts/snap_counts_{{0}}.parquet"], "parquet"),
    ("injuries_{}",       [f"{BASE}/injuries/injuries_{{0}}.parquet"], "parquet"),
    ("depth_charts_{}",   [f"{BASE}/depth_charts/depth_charts_{{0}}.parquet"], "parquet"),
    ("ftn_charting_{}",   [f"{BASE}/ftn_charting/ftn_charting_{{0}}.parquet"], "parquet"),
    ("pfr_pass_{}",       [f"{BASE}/pfr_advstats/advstats_week_pass_{{0}}.parquet"], "parquet"),
    ("pfr_rush_{}",       [f"{BASE}/pfr_advstats/advstats_week_rush_{{0}}.parquet"], "parquet"),
    ("pfr_rec_{}",        [f"{BASE}/pfr_advstats/advstats_week_rec_{{0}}.parquet"], "parquet"),
    ("pfr_def_{}",        [f"{BASE}/pfr_advstats/advstats_week_def_{{0}}.parquet"], "parquet"),
]

# (key, url, format) — full-history files, frozen as offline fallbacks
STATIC = [
    ("ngs_passing",   f"{BASE}/nextgen_stats/ngs_passing.parquet",  "parquet"),
    ("ngs_rushing",   f"{BASE}/nextgen_stats/ngs_rushing.parquet",  "parquet"),
    ("ngs_receiving", f"{BASE}/nextgen_stats/ngs_receiving.parquet","parquet"),
    ("draft_picks",   f"{BASE}/draft_picks/draft_picks.parquet",    "parquet"),
    ("draft_values",  f"{RAW_GH}/nflverse/nfldata/master/data/draft_values.csv", "csv"),
    ("combine",       f"{BASE}/combine/combine.parquet",            "parquet"),
    ("win_totals",    f"{RAW_GH}/nflverse/nfldata/master/data/win_totals.csv", "csv"),
    ("game_lines",    "https://raw.githubusercontent.com/mrcaseb/nfl-data/master/data/nfl_lines_odds.csv.gz", "csv"),
    ("scoring_lines", f"{RAW_GH}/nflverse/nfldata/master/data/sc_lines.csv", "csv"),
    ("officials",     f"{RAW_GH}/nflverse/nfldata/master/data/officials.csv", "csv"),
    ("team_desc",     f"{RAW_GH}/nflverse/nflfastR-data/master/teams_colors_logos.csv", "csv"),
    ("id_map",        f"{RAW_GH}/dynastyprocess/data/master/files/db_playerids.csv", "csv"),
    ("players",       f"{BASE}/players/players.parquet",            "parquet"),
    ("contracts",     f"{BASE}/contracts/historical_contracts.parquet", "parquet"),
    ("schedules",     "http://www.habitatring.com/games.csv",       "csv"),
]


def _freeze_df(df, name: str) -> Path:
    path = FROZEN / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    mb = path.stat().st_size / 1e6
    log.info("  frozen %-38s %8d rows  %6.2f MB", name, len(df), mb)
    if mb > MAX_FILE_MB:
        log.error("FILE TOO LARGE for git: %s (%.1f MB > %d MB) — aborting "
                  "before anything is committed.", name, mb, MAX_FILE_MB)
        sys.exit(1)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing frozen files")
    args = ap.parse_args()

    log.info("Current season: %d — freezing closed seasons: %s",
             CURRENT_SEASON, CLOSED_SEASONS)

    manifest_files = {}
    failures = []

    # ── 1. per-season tables ────────────────────────────────────────────────
    for key_tpl, url_tpls, fmt in PER_SEASON:
        for s in CLOSED_SEASONS:
            key = key_tpl.format(s)
            if not args.force and (FROZEN / f"{key}.parquet").exists():
                log.info("  keep   %-38s (already frozen)", key)
                continue
            df = _fetch([t.format(s) for t in url_tpls], fmt)
            if df is None:
                failures.append(key)
                continue
            _freeze_df(df, key)
            manifest_files[key] = len(df)

    # ── 2. static / full-history tables ─────────────────────────────────────
    for key, url, fmt in STATIC:
        if not args.force and (FROZEN / f"{key}.parquet").exists():
            log.info("  keep   %-38s (already frozen)", key)
            continue
        df = _fetch(url, fmt)
        if df is None:
            failures.append(key)
            continue
        _freeze_df(df, key)
        manifest_files[key] = len(df)

    # ── 3. per-season PBP aggregates ────────────────────────────────────────
    # Raw PBP (~25 MB/season) is downloaded transiently, aggregated with the
    # SAME function the live pipeline uses, and only the small aggregate is
    # frozen. This is what lets normal runs skip PBP downloads entirely.
    from feature_engineering import _aggregate_pbp_core
    for s in CLOSED_SEASONS:
        key = f"pbp_agg_{s}"
        if not args.force and (FROZEN / f"{key}.parquet").exists():
            log.info("  keep   %-38s (already frozen)", key)
            continue
        log.info("  downloading raw PBP %d (transient, not committed) …", s)
        pbp = _fetch(f"{BASE}/pbp/play_by_play_{s}.parquet")
        if pbp is None:
            failures.append(key)
            continue
        agg = _aggregate_pbp_core(pbp)
        del pbp
        _freeze_df(agg, key)
        manifest_files[key] = len(agg)

    # ── 4. manifest + size report ───────────────────────────────────────────
    total_mb = sum(p.stat().st_size for p in FROZEN.glob("*.parquet")) / 1e6
    manifest = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "current_season": CURRENT_SEASON,
        "frozen_seasons": CLOSED_SEASONS,
        "total_mb":       round(total_mb, 2),
        "updated_files":  manifest_files,
        "failures":       failures,
    }
    with open(FROZEN / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Frozen layer total: %.1f MB (%d files)",
             total_mb, len(list(FROZEN.glob('*.parquet'))))
    if total_mb > WARN_TOTAL_MB:
        log.warning("Frozen layer exceeds %d MB — consider moving the largest "
                    "tables to GitHub Release assets instead.", WARN_TOTAL_MB)
    if failures:
        # Missing per-season files for CLOSED seasons are a real problem —
        # fail loudly so the workflow shows red instead of silently shipping
        # an incomplete frozen layer.
        log.error("FAILED to freeze %d table(s): %s", len(failures), failures)
        sys.exit(1)
    log.info("Freeze complete.")


if __name__ == "__main__":
    main()

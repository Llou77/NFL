"""
data_loader.py
==============
Downloads and caches every available nflverse / nfl_data_py table.
All data is stored as parquet files in data/raw/ for fast reloads.
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── seasons we care about ──────────────────────────────────────────────────────
WINDOW_SEASONS = [2023, 2024, 2025]          # 3-season training window
CURRENT_SEASON = 2026                         # prediction target
ALL_SEASONS = WINDOW_SEASONS + [CURRENT_SEASON]
EXTENDED_SEASONS = list(range(2014, 2027))   # for H2H lookback (10 yrs)
BACKTEST_SEASONS = list(range(2020, 2027))   # for backtesting framework


def _save(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame as parquet and return its path."""
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    logger.info(f"  Saved {name}.parquet  ({len(df):,} rows)")
    return path


def _load(name: str) -> Optional[pd.DataFrame]:
    """Load a parquet file if it exists."""
    path = RAW_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _fresh_enough(name: str, max_hours: float = 24.0) -> bool:
    """Return True if the parquet file is younger than max_hours."""
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < (max_hours * 3600)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download / reload every available data table.
    Returns a dict of {table_name: DataFrame}.
    Set force_refresh=True to re-download even if cache is fresh.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError(
            "nfl_data_py not installed. Run: pip install nfl_data_py"
        )

    tables: dict[str, pd.DataFrame] = {}

    # ── 1. Schedules (game-level metadata + results) ───────────────────────
    key = "schedules"
    if force_refresh or not _fresh_enough(key, max_hours=6):
        logger.info("Fetching schedules …")
        df = nfl.import_schedules(ALL_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # Extended schedules for H2H (back to 2014)
    key = "schedules_extended"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching extended schedules (H2H history) …")
        df = nfl.import_schedules(EXTENDED_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # ── 2. Play-by-play (the richest table) ───────────────────────────────
    # Contains EPA, WPA, CPOE, air_yards, success, etc.
    for season in ALL_SEASONS:
        key = f"pbp_{season}"
        if force_refresh or not _fresh_enough(key, max_hours=12):
            logger.info(f"Fetching PBP {season} …")
            df = nfl.import_pbp_data([season], downcast=True)
            _save(df, key)
        tables[key] = _load(key)

    # Extended PBP for H2H lookback
    for season in EXTENDED_SEASONS:
        if season in ALL_SEASONS:
            continue
        key = f"pbp_{season}"
        if force_refresh or not _fresh_enough(key, max_hours=72):
            logger.info(f"Fetching PBP {season} (H2H) …")
            try:
                df = nfl.import_pbp_data([season], downcast=True)
                _save(df, key)
            except Exception as e:
                logger.warning(f"  Could not fetch PBP {season}: {e}")
        if (RAW_DIR / f"{key}.parquet").exists():
            tables[key] = _load(key)

    # ── 3. Weekly player stats ──────────────────────────────────────────────
    key = "player_stats_weekly"
    if force_refresh or not _fresh_enough(key, max_hours=12):
        logger.info("Fetching weekly player stats …")
        df = nfl.import_weekly_data(ALL_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # ── 4. Seasonal player stats ────────────────────────────────────────────
    key = "player_stats_seasonal"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching seasonal player stats …")
        df = nfl.import_seasonal_data(ALL_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # ── 5. Rosters (weekly depth chart + snap counts) ──────────────────────
    key = "rosters"
    if force_refresh or not _fresh_enough(key, max_hours=12):
        logger.info("Fetching rosters …")
        df = nfl.import_rosters(ALL_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # ── 6. Snap counts ──────────────────────────────────────────────────────
    key = "snap_counts"
    if force_refresh or not _fresh_enough(key, max_hours=12):
        logger.info("Fetching snap counts …")
        df = nfl.import_snap_counts(ALL_SEASONS)
        _save(df, key)
    tables[key] = _load(key)

    # ── 7. Draft picks ──────────────────────────────────────────────────────
    key = "draft_picks"
    if force_refresh or not _fresh_enough(key, max_hours=168):   # weekly
        logger.info("Fetching draft picks …")
        df = nfl.import_draft_picks()
        _save(df, key)
    tables[key] = _load(key)

    # ── 8. Draft pick values (historical trade chart) ──────────────────────
    key = "draft_values"
    if force_refresh or not _fresh_enough(key, max_hours=168):
        logger.info("Fetching draft pick values …")
        df = nfl.import_draft_values()
        _save(df, key)
    tables[key] = _load(key)

    # ── 9. Combine results ──────────────────────────────────────────────────
    key = "combine"
    if force_refresh or not _fresh_enough(key, max_hours=168):
        logger.info("Fetching combine results …")
        df = nfl.import_combine_data()
        _save(df, key)
    tables[key] = _load(key)

    # ── 10. Pre-season Vegas win totals ─────────────────────────────────────
    key = "win_totals"
    if force_refresh or not _fresh_enough(key, max_hours=168):
        logger.info("Fetching Vegas win totals …")
        df = nfl.import_win_totals()
        _save(df, key)
    tables[key] = _load(key)

    # ── 11. Scoring lines / spreads ─────────────────────────────────────────
    key = "scoring_lines"
    if force_refresh or not _fresh_enough(key, max_hours=6):
        logger.info("Fetching scoring lines …")
        df = nfl.import_sc_lines()
        _save(df, key)
    tables[key] = _load(key)

    # ── 12. Officials (referee assignments) ─────────────────────────────────
    key = "officials"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching officials …")
        df = nfl.import_officials()
        _save(df, key)
    tables[key] = _load(key)

    # ── 13. Team descriptors (colors, logos, stadium info) ──────────────────
    key = "team_desc"
    if force_refresh or not _fresh_enough(key, max_hours=168):
        logger.info("Fetching team descriptors …")
        df = nfl.import_team_desc()
        _save(df, key)
    tables[key] = _load(key)

    # ── 14. ID mappings (cross-reference player IDs across platforms) ────────
    key = "id_map"
    if force_refresh or not _fresh_enough(key, max_hours=168):
        logger.info("Fetching player ID map …")
        df = nfl.import_ids()
        _save(df, key)
    tables[key] = _load(key)

    # ── 15. NGS passing stats (Next Gen Stats) ───────────────────────────────
    key = "ngs_passing"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching NGS passing …")
        try:
            df = nfl.import_ngs_data("passing", ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  NGS passing unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 16. NGS rushing stats ────────────────────────────────────────────────
    key = "ngs_rushing"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching NGS rushing …")
        try:
            df = nfl.import_ngs_data("rushing", ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  NGS rushing unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 17. NGS receiving stats ──────────────────────────────────────────────
    key = "ngs_receiving"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching NGS receiving …")
        try:
            df = nfl.import_ngs_data("receiving", ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  NGS receiving unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 18. PFR passing weekly ───────────────────────────────────────────────
    key = "pfr_passing"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching PFR weekly passing …")
        try:
            df = nfl.import_pfr_advstats(ALL_SEASONS, stat_type="pass", s_type="week")
            _save(df, key)
        except Exception as e:
            logger.warning(f"  PFR passing unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 19. PFR rushing weekly ───────────────────────────────────────────────
    key = "pfr_rushing"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching PFR weekly rushing …")
        try:
            df = nfl.import_pfr_advstats(ALL_SEASONS, stat_type="rush", s_type="week")
            _save(df, key)
        except Exception as e:
            logger.warning(f"  PFR rushing unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 20. PFR receiving weekly ─────────────────────────────────────────────
    key = "pfr_receiving"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching PFR weekly receiving …")
        try:
            df = nfl.import_pfr_advstats(ALL_SEASONS, stat_type="rec", s_type="week")
            _save(df, key)
        except Exception as e:
            logger.warning(f"  PFR receiving unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 21. PFR defensive weekly ─────────────────────────────────────────────
    key = "pfr_defense"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching PFR weekly defense …")
        try:
            df = nfl.import_pfr_advstats(ALL_SEASONS, stat_type="def", s_type="week")
            _save(df, key)
        except Exception as e:
            logger.warning(f"  PFR defense unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 22. FTN charting data (blocking, pressure, route running) ────────────
    key = "ftn_charting"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching FTN charting data …")
        try:
            df = nfl.import_ftn_data(ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  FTN charting unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 23. Participation / tracking data ────────────────────────────────────
    key = "participation"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching participation data …")
        try:
            df = nfl.import_participation(ALL_SEASONS, include_pbp=False)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  Participation data unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 24. Injuries ──────────────────────────────────────────────────────────
    key = "injuries"
    if force_refresh or not _fresh_enough(key, max_hours=6):   # very fresh
        logger.info("Fetching injury reports …")
        try:
            df = nfl.import_injuries(ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  Injuries unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 25. Contracts / depth charts (nflverse supplemental) ─────────────────
    key = "depth_charts"
    if force_refresh or not _fresh_enough(key, max_hours=12):
        logger.info("Fetching depth charts …")
        try:
            df = nfl.import_depth_charts(ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  Depth charts unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    # ── 26. Expected points + win probability seasonal totals ─────────────────
    key = "ep_seasonal"
    if force_refresh or not _fresh_enough(key, max_hours=24):
        logger.info("Fetching EP/WP seasonal aggregates …")
        try:
            df = nfl.import_seasonal_pfr(ALL_SEASONS)
            _save(df, key)
        except Exception as e:
            logger.warning(f"  EP seasonal unavailable: {e}")
    if (RAW_DIR / f"{key}.parquet").exists():
        tables[key] = _load(key)

    logger.info(f"\nData loading complete. {len(tables)} tables loaded.")
    _save_manifest(tables)
    return tables


def _save_manifest(tables: dict) -> None:
    """Write a JSON manifest of loaded tables with row counts."""
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "tables": {k: len(v) for k, v in tables.items() if v is not None},
    }
    with open(RAW_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE ACCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def get_schedules(extended: bool = False) -> pd.DataFrame:
    key = "schedules_extended" if extended else "schedules"
    df = _load(key)
    if df is None:
        raise FileNotFoundError(f"Run load_all() first — {key}.parquet not found.")
    return df


def get_pbp(seasons: list[int] | None = None) -> pd.DataFrame:
    if seasons is None:
        seasons = ALL_SEASONS
    frames = []
    for s in seasons:
        df = _load(f"pbp_{s}")
        if df is not None:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No PBP data found. Run load_all() first.")
    return pd.concat(frames, ignore_index=True)


def get_table(name: str) -> Optional[pd.DataFrame]:
    return _load(name)


def get_game_type_flag(game_type: str) -> float:
    """
    Return the training sample weight for a given game_type string.
    Used by feature_engineering and train modules.
    """
    weights = {
        "REG": 1.0,
        "WC":  0.70,
        "DIV": 0.75,
        "CON": 0.75,
        "SB":  0.60,
        "POST": 0.70,   # generic fallback for any other playoff label
    }
    return weights.get(game_type.upper(), 1.0)


def list_available_tables() -> list[str]:
    """List all parquet files currently saved in data/raw/."""
    return sorted([p.stem for p in RAW_DIR.glob("*.parquet")])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== NFL Data Loader ===")
    print(f"Seasons: {ALL_SEASONS}")
    print(f"Extended H2H seasons: {EXTENDED_SEASONS[0]}–{EXTENDED_SEASONS[-1]}")
    print()
    tables = load_all(force_refresh=False)
    print()
    print("Available tables:")
    for name in sorted(tables.keys()):
        print(f"  {name:40s} {len(tables[name]):>8,} rows")

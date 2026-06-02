"""
data_loader.py
==============
Downloads and caches every available nflverse / nfldata table.
Calls nflverse GitHub release URLs directly — no nfl-data-py dependency,
which means no pandas version conflicts.

All data is stored as parquet files in data/raw/ for fast reloads.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── seasons ────────────────────────────────────────────────────────────────────
WINDOW_SEASONS   = [2023, 2024, 2025]
CURRENT_SEASON   = 2026
ALL_SEASONS      = WINDOW_SEASONS + [CURRENT_SEASON]
EXTENDED_SEASONS = list(range(2014, 2027))   # for H2H 10-year lookback
BACKTEST_SEASONS = list(range(2020, 2027))

# ── nflverse base URL ──────────────────────────────────────────────────────────
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
RAW_GH = "https://raw.githubusercontent.com"


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(df: pd.DataFrame, name: str) -> Path:
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    logger.info("  Saved %-40s %8d rows", name + ".parquet", len(df))
    return path


def _load(name: str) -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _fresh(name: str, max_hours: float = 24.0) -> bool:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_hours * 3600


def _fetch(url: str, fmt: str = "parquet") -> Optional[pd.DataFrame]:
    """Download a parquet or CSV from a URL. Returns None on failure."""
    try:
        if fmt == "parquet":
            return pd.read_parquet(url, engine="pyarrow")
        else:
            return pd.read_csv(url)
    except Exception as e:
        logger.warning("  Could not fetch %s — %s", url, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download / reload every available data table.
    Returns dict {table_name: DataFrame}.
    """
    tables: dict[str, pd.DataFrame] = {}

    def _get(key, url, fmt="parquet", max_hours=24.0):
        """Fetch-and-cache helper."""
        if not force_refresh and _fresh(key, max_hours):
            df = _load(key)
            if df is not None:
                tables[key] = df
                return
        logger.info("Fetching %s …", key)
        df = _fetch(url, fmt)
        if df is not None:
            _save(df, key)
            tables[key] = df
        else:
            existing = _load(key)
            if existing is not None:
                tables[key] = existing

    def _get_yearly(key_tpl, url_tpl, seasons, fmt="parquet", max_hours=24.0):
        """Fetch per-season files and cache individually."""
        frames = []
        for s in seasons:
            key = key_tpl.format(s)
            if not force_refresh and _fresh(key, max_hours):
                df = _load(key)
            else:
                logger.info("Fetching %s …", key)
                df = _fetch(url_tpl.format(s), fmt)
                if df is not None:
                    _save(df, key)
            if df is not None:
                frames.append(df)
        return frames

    # ── 1. Schedules ──────────────────────────────────────────────────────────
    _get("schedules",
         "http://www.habitatring.com/games.csv",
         fmt="csv", max_hours=6)

    # ── 2. Play-by-play (current + window seasons) ────────────────────────────
    for s in ALL_SEASONS:
        key = f"pbp_{s}"
        url = f"{BASE}/pbp/play_by_play_{s}.parquet"
        if not force_refresh and _fresh(key, max_hours=12):
            df = _load(key)
        else:
            logger.info("Fetching PBP %d …", s)
            df = _fetch(url)
            if df is not None:
                _save(df, key)
        if df is not None:
            tables[key] = df

    # Extended PBP for H2H (older seasons, refresh rarely)
    for s in EXTENDED_SEASONS:
        if s in ALL_SEASONS:
            continue
        key = f"pbp_{s}"
        if not force_refresh and _fresh(key, max_hours=168):  # 1 week
            df = _load(key)
            if df is not None:
                tables[key] = df
            continue
        logger.info("Fetching PBP %d (H2H history) …", s)
        df = _fetch(f"{BASE}/pbp/play_by_play_{s}.parquet")
        if df is not None:
            _save(df, key)
            tables[key] = df

    # ── 3. Weekly player stats ────────────────────────────────────────────────
    frames = _get_yearly(
        "player_stats_weekly_{}", 
        f"{BASE}/player_stats/player_stats_{{0}}.parquet",
        ALL_SEASONS, max_hours=12
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "player_stats_weekly")
        tables["player_stats_weekly"] = combined

    # ── 4. Seasonal player stats ──────────────────────────────────────────────
    frames = _get_yearly(
        "player_stats_season_{}",
        f"{BASE}/player_stats/player_stats_season_{{0}}.parquet",
        ALL_SEASONS, max_hours=24
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "player_stats_seasonal")
        tables["player_stats_seasonal"] = combined

    # ── 5. Rosters (seasonal) ─────────────────────────────────────────────────
    frames = _get_yearly(
        "rosters_{}",
        f"{BASE}/rosters/roster_{{0}}.parquet",
        ALL_SEASONS, max_hours=12
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "rosters")
        tables["rosters"] = combined

    # ── 6. Weekly rosters ─────────────────────────────────────────────────────
    frames = _get_yearly(
        "rosters_weekly_{}",
        f"{BASE}/weekly_rosters/roster_weekly_{{0}}.parquet",
        ALL_SEASONS, max_hours=12
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "rosters_weekly")
        tables["rosters_weekly"] = combined

    # ── 7. Snap counts ────────────────────────────────────────────────────────
    frames = _get_yearly(
        "snap_counts_{}",
        f"{BASE}/snap_counts/snap_counts_{{0}}.parquet",
        ALL_SEASONS, max_hours=12
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "snap_counts")
        tables["snap_counts"] = combined

    # ── 8. Injuries ───────────────────────────────────────────────────────────
    frames = _get_yearly(
        "injuries_{}",
        f"{BASE}/injuries/injuries_{{0}}.parquet",
        ALL_SEASONS, max_hours=6  # very fresh — injury reports update daily
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "injuries")
        tables["injuries"] = combined

    # ── 9. Depth charts ───────────────────────────────────────────────────────
    frames = _get_yearly(
        "depth_charts_{}",
        f"{BASE}/depth_charts/depth_charts_{{0}}.parquet",
        ALL_SEASONS, max_hours=12
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "depth_charts")
        tables["depth_charts"] = combined

    # ── 10. FTN charting data ─────────────────────────────────────────────────
    frames = _get_yearly(
        "ftn_charting_{}",
        f"{BASE}/ftn_charting/ftn_charting_{{0}}.parquet",
        ALL_SEASONS, max_hours=24
    )
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        _save(combined, "ftn_charting")
        tables["ftn_charting"] = combined

    # ── 11. Next Gen Stats ────────────────────────────────────────────────────
    for stat_type in ["passing", "rushing", "receiving"]:
        key = f"ngs_{stat_type}"
        url = f"{BASE}/nextgen_stats/ngs_{stat_type}.parquet"
        if not force_refresh and _fresh(key, max_hours=24):
            df = _load(key)
            if df is not None:
                tables[key] = df
            continue
        logger.info("Fetching NGS %s …", stat_type)
        df = _fetch(url)
        if df is not None:
            _save(df, key)
            tables[key] = df

    # ── 12. PFR advanced stats (weekly) ───────────────────────────────────────
    for stat_type in ["pass", "rush", "rec", "def"]:
        frames = []
        for s in ALL_SEASONS:
            key = f"pfr_{stat_type}_{s}"
            url = f"{BASE}/pfr_advstats/advstats_week_{stat_type}_{s}.parquet"
            if not force_refresh and _fresh(key, max_hours=24):
                df = _load(key)
            else:
                logger.info("Fetching PFR %s %d …", stat_type, s)
                df = _fetch(url)
                if df is not None:
                    _save(df, key)
            if df is not None:
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            _save(combined, f"pfr_{stat_type}")
            tables[f"pfr_{stat_type}"] = combined

    # ── 13. Draft picks ───────────────────────────────────────────────────────
    _get("draft_picks",
         f"{BASE}/draft_picks/draft_picks.parquet",
         max_hours=168)

    # ── 14. Draft values ──────────────────────────────────────────────────────
    _get("draft_values",
         f"{RAW_GH}/nflverse/nfldata/master/data/draft_values.csv",
         fmt="csv", max_hours=168)

    # ── 15. Combine data ──────────────────────────────────────────────────────
    _get("combine",
         f"{BASE}/combine/combine.parquet",
         max_hours=168)

    # ── 16. Vegas preseason win totals ───────────────────────────────────────
    # Correct source: nflverse/nfldata (season-level preseason O/U lines)
    _get("win_totals",
         "https://raw.githubusercontent.com/nflverse/nfldata/master/data/win_totals.csv",
         fmt="csv", max_hours=168)

    # ── 16b. Historical game lines (spread/total per game) ────────────────────
    # This is the mrcaseb source — game-level lines, not preseason win totals
    _get("game_lines",
         "https://raw.githubusercontent.com/mrcaseb/nfl-data/master/data/nfl_lines_odds.csv.gz",
         fmt="csv", max_hours=24)

    # ── 17. Scoring lines / spreads ───────────────────────────────────────────
    _get("scoring_lines",
         f"{RAW_GH}/nflverse/nfldata/master/data/sc_lines.csv",
         fmt="csv", max_hours=6)

    # ── 18. Officials / referees ──────────────────────────────────────────────
    _get("officials",
         f"{RAW_GH}/nflverse/nfldata/master/data/officials.csv",
         fmt="csv", max_hours=168)

    # ── 19. Team descriptors (colors, logos, stadium) ─────────────────────────
    _get("team_desc",
         f"{RAW_GH}/nflverse/nflfastR-data/master/teams_colors_logos.csv",
         fmt="csv", max_hours=168)

    # ── 20. Player ID mapping ─────────────────────────────────────────────────
    _get("id_map",
         f"{RAW_GH}/dynastyprocess/data/master/files/db_playerids.csv",
         fmt="csv", max_hours=168)

    # ── 21. Players (static descriptors) ─────────────────────────────────────
    _get("players",
         f"{BASE}/players/players.parquet",
         max_hours=168)

    # ── 22. Historical contracts ──────────────────────────────────────────────
    _get("contracts",
         f"{BASE}/contracts/historical_contracts.parquet",
         max_hours=168)

    logger.info("\nData loading complete — %d tables loaded.", len(tables))
    _save_manifest(tables)
    return tables


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE ACCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def get_schedules(extended: bool = False) -> pd.DataFrame:
    """Return the schedules DataFrame (loaded from cache)."""
    df = _load("schedules")
    if df is None:
        raise FileNotFoundError("Run load_all() first — schedules.parquet not found.")
    if not extended:
        return df
    # For extended H2H lookback, schedules CSV already covers all years
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
    """Return default training sample weight for a game_type string."""
    return {
        "REG": 1.00, "WC": 0.70, "DIV": 0.75,
        "CON": 0.75, "SB": 0.60, "POST": 0.70,
    }.get(str(game_type).upper(), 1.0)


def list_available_tables() -> list[str]:
    return sorted(p.stem for p in RAW_DIR.glob("*.parquet"))


def _save_manifest(tables: dict) -> None:
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "tables": {k: len(v) for k, v in tables.items() if v is not None},
    }
    with open(RAW_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = load_all(force_refresh=False)
    print(f"\nLoaded {len(tables)} tables:")
    for name in sorted(tables):
        print(f"  {name:45s} {len(tables[name]):>8,} rows")

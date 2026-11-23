"""
data_loader.py — downloads and caches the nflverse tables the model uses.

Closed seasons are read from the committed data/frozen/ layer (see
scripts/freeze_data.py) with no network access; only the schedule, the
season in progress and the full-history Next Gen Stats files are fetched
live. Every table is cached as parquet in data/raw/.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
FROZEN_DIR = ROOT / "data" / "frozen"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_current_season(today: Optional[datetime] = None) -> int:
    """NFL season year from the clock: March-December belong to the calendar
    year, January-February (playoffs, Super Bowl) to the previous season."""
    t = today or datetime.now(timezone.utc)
    return t.year if t.month >= 3 else t.year - 1


CURRENT_SEASON = get_current_season()
# Training window: the last four completed seasons plus the season in progress.
ALL_SEASONS = list(range(CURRENT_SEASON - 4, CURRENT_SEASON + 1))

BASE   = "https://github.com/nflverse/nflverse-data/releases/download"
RAW_GH = "https://raw.githubusercontent.com"

# nflverse's canonical schedule (scores, lines, roof, starting QBs, coaches).
# habitatring.com is Lee Sharpe's older mirror of the same file.
SCHEDULE_URLS = [
    f"{RAW_GH}/nflverse/nfldata/master/data/games.csv",
    "http://www.habitatring.com/games.csv",
]


# ── cache helpers ──────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(RAW_DIR / f"{name}.parquet", index=False, engine="pyarrow",
                  compression="snappy")
    logger.info("  Saved %-40s %8d rows", name + ".parquet", len(df))


def _load(name: str) -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _load_frozen(name: str) -> Optional[pd.DataFrame]:
    path = FROZEN_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def has_frozen(name: str) -> bool:
    return (FROZEN_DIR / f"{name}.parquet").exists()


def _fresh(name: str, max_hours: float) -> bool:
    path = RAW_DIR / f"{name}.parquet"
    return path.exists() and (time.time() - path.stat().st_mtime) < max_hours * 3600


def _fetch(url, fmt: str = "parquet") -> Optional[pd.DataFrame]:
    """Download a parquet or CSV file. `url` may be a list of candidates tried
    in order (nflverse renames releases now and then). None on failure."""
    for u in (url if isinstance(url, (list, tuple)) else [url]):
        try:
            return pd.read_parquet(u, engine="pyarrow") if fmt == "parquet" else pd.read_csv(u)
        except Exception as e:  # noqa: BLE001 — try the next candidate
            logger.warning("  Could not fetch %s — %s", u, e)
    return None


def concat_seasons(frames: list) -> pd.DataFrame:
    """Concatenate per-season frames whose column dtypes drift across nflverse
    eras (e.g. jersey_number is a string in some seasons and a float in
    others), which pyarrow refuses to write as one column. Mixed text/number
    columns become nullable strings; mixed numeric columns become float64."""
    dtypes: dict = {}
    for f in frames:
        for c in f.columns:
            dtypes.setdefault(c, set()).add(str(f[c].dtype))
    mixed = {c: ds for c, ds in dtypes.items() if len(ds) > 1}
    out = []
    for f in frames:
        f = f.copy()
        for c, ds in mixed.items():
            if c in f.columns:
                if any(d in ("object", "string") for d in ds):
                    f[c] = f[c].astype("string")
                else:
                    f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
        out.append(f)
    return pd.concat(out, ignore_index=True)


def load_frozen_pbp_agg(season: int) -> Optional[pd.DataFrame]:
    """Per-season team-game aggregate of play-by-play, built by
    scripts/freeze_data.py with feature_engineering._aggregate_pbp_core."""
    return _load_frozen(f"pbp_agg_{season}")


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_all(force_refresh: bool = False, seasons: Optional[list] = None) -> dict:
    """Fetch (or reload from cache) every table the feature pipeline reads.
    Returns {table_name: DataFrame}."""
    seasons = seasons or ALL_SEASONS
    tables: dict[str, pd.DataFrame] = {}

    def _get(key, url, fmt="parquet", max_hours=24.0):
        """Single file: fresh cache → download → stale cache → frozen copy."""
        if not force_refresh and _fresh(key, max_hours):
            tables[key] = _load(key)
            return
        logger.info("Fetching %s …", key)
        df = _fetch(url, fmt)
        if df is not None:
            _save(df, key)
        else:
            df = _load(key)
            if df is None:
                df = _load_frozen(key)
        if df is not None:
            tables[key] = df

    def _get_yearly(key_tpl, url_tpls, years, max_hours=12.0):
        """Per-season files: closed seasons from the frozen layer, the season
        in progress from the network (with stale-cache fallback)."""
        frames = []
        for s in years:
            key = key_tpl.format(s)
            df = _load_frozen(key) if s < CURRENT_SEASON else None
            if df is None and not force_refresh and _fresh(key, max_hours):
                df = _load(key)
            if df is None:
                logger.info("Fetching %s …", key)
                df = _fetch([t.format(s) for t in url_tpls])
                if df is not None:
                    _save(df, key)
                else:
                    df = _load(key)
                    if df is None:
                        df = _load_frozen(key)
            if df is not None:
                frames.append(df)
        return frames

    # 1. Schedule: one row per game, all seasons since 1999.
    _get("schedules", SCHEDULE_URLS, fmt="csv", max_hours=6)

    # 2. Play-by-play: closed seasons are represented by their frozen
    #    aggregate, so raw PBP is only downloaded for unfrozen seasons.
    for s in seasons:
        if s < CURRENT_SEASON and has_frozen(f"pbp_agg_{s}"):
            continue
        key = f"pbp_{s}"
        df = _load(key) if not force_refresh and _fresh(key, 12) else None
        if df is None:
            logger.info("Fetching PBP %d …", s)
            df = _fetch(f"{BASE}/pbp/play_by_play_{s}.parquet")
            if df is not None:
                _save(df, key)
            else:
                df = _load(key)
        if df is not None:
            tables[key] = df

    # 3. Weekly player stats (QB form). nflverse renamed the release in 2025;
    #    the old name stays as a fallback.
    frames = _get_yearly(
        "player_stats_weekly_{}",
        [f"{BASE}/stats_player/stats_player_week_{{0}}.parquet",
         f"{BASE}/player_stats/player_stats_{{0}}.parquet"],
        seasons,
    )
    if frames:
        tables["player_stats_weekly"] = concat_seasons(frames)
        _save(tables["player_stats_weekly"], "player_stats_weekly")

    # 4. Weekly rosters (roster continuity). One extra, older season so the
    #    first season in the window also has a previous roster to compare to.
    prev = seasons[0] - 1
    roster_years = ([prev] if has_frozen(f"rosters_weekly_{prev}") else []) + list(seasons)
    frames = _get_yearly("rosters_weekly_{}",
                         [f"{BASE}/weekly_rosters/roster_weekly_{{0}}.parquet"],
                         roster_years)
    if frames:
        tables["rosters_weekly"] = concat_seasons(frames)
        _save(tables["rosters_weekly"], "rosters_weekly")

    # 5. Next Gen Stats (single full-history files, frozen copy as fallback).
    for stat_type in ("passing", "rushing", "receiving"):
        _get(f"ngs_{stat_type}", f"{BASE}/nextgen_stats/ngs_{stat_type}.parquet")

    logger.info("Data loading complete — %d tables loaded.", len(tables))
    with open(RAW_DIR / "manifest.json", "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "tables": {k: len(v) for k, v in tables.items()}}, f, indent=2)
    return tables


# ══════════════════════════════════════════════════════════════════════════════
#  ACCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def get_schedules() -> pd.DataFrame:
    df = _load("schedules")
    if df is None:
        raise FileNotFoundError("schedules.parquet not found — run load_all() first.")
    return df


def get_pbp(seasons: list[int]) -> pd.DataFrame:
    frames = [df for df in (_load(f"pbp_{s}") for s in seasons) if df is not None]
    if not frames:
        raise FileNotFoundError("No PBP data found — run load_all() first.")
    return pd.concat(frames, ignore_index=True)


def get_table(name: str) -> Optional[pd.DataFrame]:
    return _load(name)


def regular_season_weeks(season: int) -> int:
    """Final regular-season week: 17 through 2020, 18 since the 2021 expansion."""
    return 18 if int(season) >= 2021 else 17


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, df in sorted(load_all().items()):
        print(f"  {name:30s} {len(df):>9,} rows")

"""
feature_h2h.py
==============
Builds head-to-head history features for every game matchup.

Tiered lookback system:
  - Division rivals    (2 games/season): 3-season window  (~6 games)
  - Same-conf non-div  (0-1 games/season): 6-season window (~6 games)
  - Cross-conference   (rare):            10-season window (~4-6 games)

Features are zeroed when fewer than MIN_H2H_GAMES exist in the lookback.
H2H signals are modulated by roster continuity to avoid over-indexing
on matchups involving largely different personnel.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import get_schedules, ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_H2H_GAMES = 3     # minimum games needed to use H2H signal

# Division lookups
_DIVISIONS = {
    "NFC East":  {"DAL", "NYG", "PHI", "WAS"},
    "NFC North": {"CHI", "DET", "GB",  "MIN"},
    "NFC South": {"ATL", "CAR", "NO",  "TB"},
    "NFC West":  {"ARI", "LAR", "SF",  "SEA"},
    "AFC East":  {"BUF", "MIA", "NE",  "NYJ"},
    "AFC North": {"BAL", "CIN", "CLE", "PIT"},
    "AFC South": {"HOU", "IND", "JAX", "TEN"},
    "AFC West":  {"DEN", "KC",  "LV",  "LAC"},
}
_CONFERENCES = {
    "NFC": {"DAL","NYG","PHI","WAS","CHI","DET","GB","MIN",
            "ATL","CAR","NO","TB","ARI","LAR","SF","SEA"},
    "AFC": {"BUF","MIA","NE","NYJ","BAL","CIN","CLE","PIT",
            "HOU","IND","JAX","TEN","DEN","KC","LV","LAC"},
}


def _matchup_type(team_a: str, team_b: str) -> str:
    """Return 'division', 'conference', or 'cross' for two team abbreviations."""
    for div_teams in _DIVISIONS.values():
        if team_a in div_teams and team_b in div_teams:
            return "division"
    for conf_teams in _CONFERENCES.values():
        if team_a in conf_teams and team_b in conf_teams:
            return "conference"
    return "cross"


def _lookback_seasons(matchup_type: str, current_season: int) -> list[int]:
    if matchup_type == "division":
        return list(range(current_season - 3, current_season))   # 3 seasons
    elif matchup_type == "conference":
        return list(range(current_season - 6, current_season))   # 6 seasons
    else:
        return list(range(current_season - 10, current_season))  # 10 seasons


def build_h2h_features(
    game_features: pd.DataFrame,
    roster_continuity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For every row in game_features (one row per game), compute H2H features
    using all historical data prior to that game's date.

    Parameters
    ----------
    game_features : DataFrame with columns home_team, away_team, game_date, season
    roster_continuity : optional DataFrame with (team, season, roster_continuity)

    Returns
    -------
    game_features with H2H columns appended
    """
    logger.info("Building H2H features …")

    # Load all historical schedules (extended 10 years)
    hist = get_schedules(extended=True)
    if "home_score" not in hist.columns or "away_score" not in hist.columns:
        logger.warning("  Historical scores not available — H2H features skipped")
        return _add_zero_h2h(game_features)

    # Parse dates
    hist["game_date"] = pd.to_datetime(hist["gameday"], errors="coerce")
    hist = hist[hist["game_date"].notna()].copy()

    # Normalize spread_line (positive = home favored in some datasets; adjust)
    if "spread_line" in hist.columns:
        hist["actual_spread"] = hist["home_score"] - hist["away_score"]
        hist["home_covered"] = (
            hist["actual_spread"] > -hist["spread_line"].fillna(0)
        ).astype(float)
    else:
        hist["actual_spread"] = hist["home_score"] - hist["away_score"]
        hist["home_covered"] = np.nan

    hist["total_score"] = hist["home_score"] + hist["away_score"]

    # Build a lookup: canonical matchup key (sorted teams) → list of historical games
    # We store (date, home, away, home_score, away_score, total, actual_spread, home_covered)
    matchup_history: dict[frozenset, list[dict]] = {}

    for _, row in hist.iterrows():
        key = frozenset([row["home_team"], row["away_team"]])
        entry = {
            "date":          row["game_date"],
            "home":          row["home_team"],
            "away":          row["away_team"],
            "home_score":    row.get("home_score", np.nan),
            "away_score":    row.get("away_score", np.nan),
            "total":         row.get("total_score", np.nan),
            "actual_spread": row.get("actual_spread", np.nan),
            "home_covered":  row.get("home_covered", np.nan),
            "game_type":     row.get("game_type", "REG"),
            "season":        row.get("season", 0),
        }
        matchup_history.setdefault(key, []).append(entry)

    # Pre-sort each matchup's history by date
    for key in matchup_history:
        matchup_history[key].sort(key=lambda x: x["date"])

    # Roster continuity lookup for modulation
    cont_lookup: dict[tuple, float] = {}
    if roster_continuity is not None:
        for _, r in roster_continuity.iterrows():
            cont_lookup[(r["team"], r["season"])] = float(r.get("roster_continuity", 0.5))

    # Compute H2H features row by row
    h2h_rows = []
    for _, game in game_features.iterrows():
        h = _compute_h2h_row(
            game, matchup_history, cont_lookup
        )
        h2h_rows.append(h)

    h2h_df = pd.DataFrame(h2h_rows)

    # Merge back
    result = pd.concat(
        [game_features.reset_index(drop=True), h2h_df.reset_index(drop=True)],
        axis=1,
    )

    # Save
    path = PROCESSED_DIR / "h2h_features.parquet"
    h2h_df.to_parquet(path, index=False)
    logger.info(f"  H2H features saved ({len(h2h_df)} rows, {len(h2h_df.columns)} cols)")

    return result


def _compute_h2h_row(
    game: pd.Series,
    matchup_history: dict,
    cont_lookup: dict,
) -> dict:
    """Compute all H2H features for a single game row."""
    home = game.get("home_team") or game.get("home_team_x")
    away = game.get("away_team") or game.get("away_team_x")
    game_date = game.get("game_date")
    season    = game.get("season", 2026)

    # Default / null feature set
    null_features = _null_h2h()

    if not home or not away or not game_date:
        return null_features

    mtype    = _matchup_type(home, away)
    seasons  = _lookback_seasons(mtype, int(season))
    key      = frozenset([home, away])
    history  = matchup_history.get(key, [])

    # Filter: only games BEFORE this game's date AND within lookback seasons
    past = [
        g for g in history
        if g["date"] < game_date and int(g["season"]) in seasons
    ]

    if len(past) < MIN_H2H_GAMES:
        # Not enough data — return neutral with low confidence
        feat = null_features.copy()
        feat["h2h_data_confidence"] = max(0.2, len(past) * 0.1)
        feat["h2h_matchup_type"]    = _mtype_code(mtype)
        feat["h2h_n_games"]         = len(past)
        return feat

    # Recency weights (exponential decay, more recent = higher weight)
    weights = np.array([
        np.exp(-0.15 * (len(past) - 1 - i))
        for i in range(len(past))
    ])
    weights /= weights.sum()

    # ── Compute H2H statistics ─────────────────────────────────────────────
    # From home team's perspective
    home_wins = []
    margins   = []
    totals    = []
    ats_cover = []

    for i, g in enumerate(past):
        if g["home"] == home:
            win = 1 if g["home_score"] > g["away_score"] else (
                  0.5 if g["home_score"] == g["away_score"] else 0
            )
            margin = g["home_score"] - g["away_score"]
            covered = g["home_covered"]
        else:  # home played as away in this historical game
            win = 1 if g["away_score"] > g["home_score"] else (
                  0.5 if g["home_score"] == g["away_score"] else 0
            )
            margin = g["away_score"] - g["home_score"]
            covered = 1 - g["home_covered"] if not np.isnan(g["home_covered"]) else np.nan

        home_wins.append(win)
        margins.append(margin)
        if not np.isnan(g["total"]):
            totals.append(g["total"])
        if not np.isnan(covered):
            ats_cover.append(covered)

    home_wins = np.array(home_wins)
    margins   = np.array(margins)

    h2h_win_rate     = float(np.dot(weights, home_wins))
    h2h_avg_margin   = float(np.dot(weights, margins))
    h2h_avg_total    = float(np.average(totals, weights=weights[:len(totals)])) if totals else np.nan
    h2h_ats_cover    = float(np.mean(ats_cover)) if ats_cover else np.nan
    h2h_n_games      = len(past)
    h2h_score_var    = float(np.std(margins))  # how volatile is this matchup

    # Home win rate in this H2H (not reusing home/away roles)
    home_in_hist = [g for g in past if g["home"] == home]
    if len(home_in_hist) >= 2:
        h2h_home_win_rate = float(np.mean(
            [1 if g["home_score"] > g["away_score"] else 0
             for g in home_in_hist]
        ))
    else:
        h2h_home_win_rate = np.nan

    # ── H2H data confidence ────────────────────────────────────────────────
    conf_table = {6: 1.0, 5: 0.9, 4: 0.8, 3: 0.7}
    h2h_data_confidence = conf_table.get(min(h2h_n_games, 6), 0.7)

    # ── Roster continuity modulation ──────────────────────────────────────
    # If teams have changed a lot, H2H is less relevant
    home_cont = cont_lookup.get((home, season), 0.5)
    away_cont = cont_lookup.get((away, season), 0.5)
    avg_cont  = (home_cont + away_cont) / 2.0
    # Low continuity (<0.6) reduces H2H signal weight
    continuity_modifier = min(1.0, avg_cont / 0.6)

    h2h_win_rate   *= continuity_modifier
    h2h_avg_margin *= continuity_modifier
    if not np.isnan(h2h_avg_total):
        # Total is less affected by roster changes
        pass
    h2h_data_confidence *= continuity_modifier

    return {
        "h2h_win_rate":          h2h_win_rate,
        "h2h_avg_margin":        h2h_avg_margin,
        "h2h_avg_total":         h2h_avg_total,
        "h2h_ats_cover_rate":    h2h_ats_cover,
        "h2h_home_win_rate":     h2h_home_win_rate,
        "h2h_score_variance":    h2h_score_var,
        "h2h_n_games":           h2h_n_games,
        "h2h_data_confidence":   h2h_data_confidence,
        "h2h_matchup_type":      _mtype_code(mtype),
        "h2h_continuity_mod":    continuity_modifier,
    }


def _null_h2h() -> dict:
    return {
        "h2h_win_rate":         0.5,     # neutral
        "h2h_avg_margin":       0.0,
        "h2h_avg_total":        np.nan,
        "h2h_ats_cover_rate":   np.nan,
        "h2h_home_win_rate":    np.nan,
        "h2h_score_variance":   np.nan,
        "h2h_n_games":          0,
        "h2h_data_confidence":  0.2,
        "h2h_matchup_type":     0,
        "h2h_continuity_mod":   1.0,
    }


def _mtype_code(mtype: str) -> int:
    return {"division": 2, "conference": 1, "cross": 0}[mtype]


def _add_zero_h2h(game_features: pd.DataFrame) -> pd.DataFrame:
    """Add null H2H columns when history is unavailable."""
    for col, val in _null_h2h().items():
        game_features[col] = val
    return game_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("H2H module loaded. Call build_h2h_features(game_features_df) to compute.")

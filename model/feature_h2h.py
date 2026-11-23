"""
feature_h2h.py — head-to-head features from earlier meetings.

Uses final scores from the schedule (all completed games since 1999,
relocated franchises merged), only meetings strictly before the game's date,
with a lookback that depends on how often the two teams meet:
  division rivals  — current + previous 3 seasons
  same conference  — current + previous 6 seasons
  cross-conference — current + previous 10 seasons
Fewer than MIN_H2H_GAMES meetings → neutral values. The win-rate and margin
signals are damped when either roster has turned over a lot.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_H2H_GAMES = 3

_DIVISIONS = [
    {"DAL", "NYG", "PHI", "WAS"}, {"CHI", "DET", "GB", "MIN"},
    {"ATL", "CAR", "NO", "TB"},   {"ARI", "LA", "SF", "SEA"},
    {"BUF", "MIA", "NE", "NYJ"},  {"BAL", "CIN", "CLE", "PIT"},
    {"HOU", "IND", "JAX", "TEN"}, {"DEN", "KC", "LV", "LAC"},
]
_NFC = set().union(*_DIVISIONS[:4])
_LOOKBACK = {"division": 3, "conference": 6, "cross": 10}
_MTYPE_CODE = {"division": 2, "conference": 1, "cross": 0}

_NULL = {
    "h2h_win_rate": 0.5, "h2h_avg_margin": 0.0, "h2h_avg_total": np.nan,
    "h2h_ats_cover_rate": np.nan, "h2h_home_win_rate": np.nan,
    "h2h_score_variance": np.nan, "h2h_n_games": 0, "h2h_data_confidence": 0.2,
    "h2h_matchup_type": 0, "h2h_continuity_mod": 1.0,
}


def _matchup_type(a: str, b: str) -> str:
    if any(a in d and b in d for d in _DIVISIONS):
        return "division"
    return "conference" if (a in _NFC) == (b in _NFC) else "cross"


def build_h2h_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Append h2h_* columns to the game-level matrix (one row per game)."""
    from data_loader import get_schedules
    from feature_engineering import FRANCHISE

    logger.info("Building H2H features …")
    s = get_schedules().copy()
    for c in ("home_score", "away_score", "spread_line", "season"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s[s["home_score"].notna() & s["away_score"].notna()].copy()
    s["date"] = pd.to_datetime(s["gameday"], errors="coerce")
    s["home"] = s["home_team"].replace(FRANCHISE)
    s["away"] = s["away_team"].replace(FRANCHISE)
    margin = s["home_score"] - s["away_score"]
    # spread_line is home-oriented, positive = home favoured: home covers iff
    # margin > spread_line. Pushes and games without a line count as unknown.
    s["home_covered"] = np.where(s["spread_line"].isna() | (margin == s["spread_line"]),
                                 np.nan, (margin > s["spread_line"]).astype(float))

    history: dict[frozenset, pd.DataFrame] = {
        k: g.sort_values("date") for k, g in s.groupby(
            s[["home", "away"]].apply(lambda r: frozenset(r), axis=1))
    }

    rows = [_h2h_row(g, history) for g in game_df.itertuples(index=False)]
    h2h = pd.DataFrame(rows, index=game_df.index)
    logger.info("  H2H features: %d games, %d with ≥%d prior meetings",
                len(h2h), int((h2h["h2h_n_games"] >= MIN_H2H_GAMES).sum()), MIN_H2H_GAMES)
    return pd.concat([game_df, h2h], axis=1)


def _h2h_row(game, history: dict) -> dict:
    home, away = game.home_franchise, game.away_franchise
    season, date = int(game.season), game.game_date
    mtype = _matchup_type(home, away)
    past = history.get(frozenset((home, away)))
    if past is not None:
        past = past[(past["date"] < date) & (past["season"] >= season - _LOOKBACK[mtype])]
    n = 0 if past is None else len(past)
    if n < MIN_H2H_GAMES:
        return {**_NULL, "h2h_n_games": n, "h2h_matchup_type": _MTYPE_CODE[mtype],
                "h2h_data_confidence": max(0.2, n * 0.1)}

    as_home = (past["home"] == home).to_numpy()
    hs, as_ = past["home_score"].to_numpy(), past["away_score"].to_numpy()
    margin = np.where(as_home, hs - as_, as_ - hs)            # from today's home team's view
    win = np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))
    covered = np.where(as_home, past["home_covered"], 1 - past["home_covered"])
    w = np.exp(-0.15 * np.arange(n - 1, -1, -1))              # recent meetings weigh more
    w /= w.sum()

    conts = [c for c in (getattr(game, "home_roster_continuity", np.nan),
                         getattr(game, "away_roster_continuity", np.nan)) if pd.notna(c)]
    cont = float(np.mean(conts)) if conts else 0.5
    mod = min(1.0, cont / 0.6)                                # low continuity → weaker signal

    home_games = as_home.sum()
    return {
        "h2h_win_rate":        float(np.dot(w, win)) * mod,
        "h2h_avg_margin":      float(np.dot(w, margin)) * mod,
        "h2h_avg_total":       float(np.dot(w, hs + as_)),
        "h2h_ats_cover_rate":  float(np.nanmean(covered)) if np.isfinite(covered).any() else np.nan,
        "h2h_home_win_rate":   float((margin[as_home] > 0).mean()) if home_games >= 2 else np.nan,
        "h2h_score_variance":  float(np.std(margin)),
        "h2h_n_games":         n,
        "h2h_data_confidence": {6: 1.0, 5: 0.9, 4: 0.8}.get(min(n, 6), 0.7) * mod,
        "h2h_matchup_type":    _MTYPE_CODE[mtype],
        "h2h_continuity_mod":  mod,
    }

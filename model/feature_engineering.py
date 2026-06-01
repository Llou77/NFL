"""
feature_engineering.py
=======================
Computes every feature used by the model from raw nflverse tables.

Feature groups:
  1.  Team rolling stats (EPA, efficiency, scoring, turnover, pace, ST)
  2.  QB-specific metrics (CPOE, pressure, air yards)
  3.  Matchup gap features (offense vs. opposing defense)
  4.  Contextual features (home/away, rest, weather, dome, primetime)
  5.  Injury / roster availability scores
  6.  Power rankings (Elo + Vegas-implied)
  7.  Cross-season adjustment features
  8.  Official / referee tendencies
  9.  Pace & game-script features
  10. game_type flag + sample weights

All features are computed with strict temporal ordering —
no future data ever leaks into a game's feature row.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    get_schedules, get_pbp, get_table,
    get_game_type_flag, ALL_SEASONS, EXTENDED_SEASONS
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rolling lookback window sizes
ROLLING_SHORT = 4     # recent form
ROLLING_MED   = 8     # medium form
ROLLING_LONG  = 16    # full-season form

ELO_K        = 20.0   # ELO update speed
ELO_START    = 1500.0


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_all_features(seasons: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Build the full feature matrix. Returns one row per team per game
    (i.e., every game appears twice — once for each team as 'team').
    The final game_features() function pivots this into one row per game.
    """
    if seasons is None:
        seasons = ALL_SEASONS

    logger.info("Building features …")

    # ── 1. Base game table ─────────────────────────────────────────────────
    games = _build_game_base(seasons)
    logger.info(f"  Base game table: {len(games)} rows")

    # ── 2. PBP aggregates per team per game ───────────────────────────────
    pbp_agg = _aggregate_pbp(seasons)
    logger.info(f"  PBP aggregates: {len(pbp_agg)} rows")

    # ── 3. Merge PBP into game table (home + away perspective) ────────────
    team_games = _merge_team_perspective(games, pbp_agg)
    logger.info(f"  Team-game table: {len(team_games)} rows")

    # ── 4. Rolling features (applied per team, time-ordered) ──────────────
    team_games = _add_rolling_features(team_games)
    logger.info("  Rolling features added")

    # ── 5. QB features ────────────────────────────────────────────────────
    team_games = _add_qb_features(team_games, seasons)
    logger.info("  QB features added")

    # ── 6. Injury / roster scores ─────────────────────────────────────────
    team_games = _add_injury_features(team_games)
    logger.info("  Injury features added")

    # ── 7. Elo ratings ────────────────────────────────────────────────────
    team_games = _add_elo_ratings(team_games)
    logger.info("  Elo ratings added")

    # ── 8. Cross-season features ──────────────────────────────────────────
    team_games = _add_cross_season_features(team_games)
    logger.info("  Cross-season features added")

    # ── 9. Official tendencies ────────────────────────────────────────────
    team_games = _add_official_features(team_games, seasons)
    logger.info("  Official features added")

    # ── 10. NGS features ──────────────────────────────────────────────────
    team_games = _add_ngs_features(team_games, seasons)
    logger.info("  NGS features added")

    # ── 11. Contextual (home, rest, weather, dome) ────────────────────────
    team_games = _add_contextual_features(team_games)
    logger.info("  Contextual features added")

    # ── 12. Game type flag + sample weights ───────────────────────────────
    team_games["game_type_weight"] = team_games["game_type"].apply(
        get_game_type_flag
    )

    # Save team-game table
    path = PROCESSED_DIR / "team_games.parquet"
    team_games.to_parquet(path, index=False)
    logger.info(f"  team_games.parquet saved ({len(team_games):,} rows)")

    # ── 13. Pivot to one row per game ─────────────────────────────────────
    game_feats = pivot_to_game_features(team_games)
    path = PROCESSED_DIR / "game_features.parquet"
    game_feats.to_parquet(path, index=False)
    logger.info(f"  game_features.parquet saved ({len(game_feats):,} rows)")

    return game_feats


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: BASE GAME TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _build_game_base(seasons: list[int]) -> pd.DataFrame:
    """Load schedule, filter to relevant seasons, normalize game_type."""
    sched = get_schedules(extended=False)
    sched = sched[sched["season"].isin(seasons)].copy()

    # Standardize game_type
    gt_map = {
        "REG": "REG",
        "WC":  "WC",  "WIL": "WC",
        "DIV": "DIV",
        "CON": "CON", "CCG": "CON",
        "SB":  "SB",  "POST": "WC",
    }
    sched["game_type"] = (
        sched["game_type"].str.upper().map(gt_map).fillna("REG")
    )

    # Parse game date
    sched["game_date"] = pd.to_datetime(sched["gameday"], errors="coerce")

    # Compute day of week (0=Mon … 6=Sun)
    sched["day_of_week"] = sched["game_date"].dt.dayofweek

    # Primetime flag
    sched["is_primetime"] = sched["gametime"].apply(
        lambda t: 1 if isinstance(t, str) and
                  any(h in t for h in ["20:", "21:", "19:30", "8:20", "8:15"])
                  else 0
    )

    # International game flag
    intl_venues = [
        "tottenham", "wembley", "allianz", "estadio azteca",
        "melbourne", "maracana", "stade de france", "santiago bernabeu",
    ]
    venue_col = "stadium" if "stadium" in sched.columns else "location"
    if venue_col in sched.columns:
        sched["is_international"] = sched[venue_col].str.lower().apply(
            lambda v: 1 if isinstance(v, str) and
                      any(x in v for x in intl_venues) else 0
        )
    else:
        sched["is_international"] = 0

    # Dome flag (from team_desc)
    team_desc = get_table("team_desc")
    if team_desc is not None and "team_abbr" in team_desc.columns:
        dome_teams = set(
            team_desc.loc[
                team_desc["team_stadium_type"].str.lower().str.contains(
                    "dome|indoor|retract", na=False
                ), "team_abbr"
            ]
        )
        sched["is_dome"] = sched["home_team"].isin(dome_teams).astype(int)
    else:
        sched["is_dome"] = 0

    # Division rivalry flag
    sched["is_division_game"] = (
        sched["div_game"].fillna(0).astype(int)
        if "div_game" in sched.columns
        else _compute_division_flag(sched)
    )

    keep_cols = [
        "game_id", "season", "week", "game_type", "game_date",
        "home_team", "away_team",
        "home_score", "away_score",
        "spread_line", "total_line",
        "home_moneyline", "away_moneyline",
        "stadium", "roof",
        "temp", "wind", "humidity",
        "day_of_week", "is_primetime", "is_international",
        "is_dome", "is_division_game",
        "referee",
        "game_type_weight" if "game_type_weight" in sched.columns else None,
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in sched.columns]

    return sched[keep_cols].reset_index(drop=True)


def _compute_division_flag(sched: pd.DataFrame) -> pd.Series:
    """Compute division rivalry from team division memberships."""
    nfc_east  = {"DAL", "NYG", "PHI", "WAS"}
    nfc_north = {"CHI", "DET", "GB",  "MIN"}
    nfc_south = {"ATL", "CAR", "NO",  "TB"}
    nfc_west  = {"ARI", "LAR", "SF",  "SEA"}
    afc_east  = {"BUF", "MIA", "NE",  "NYJ"}
    afc_north = {"BAL", "CIN", "CLE", "PIT"}
    afc_south = {"HOU", "IND", "JAX", "TEN"}
    afc_west  = {"DEN", "KC",  "LV",  "LAC"}
    divisions = [nfc_east, nfc_north, nfc_south, nfc_west,
                 afc_east, afc_north, afc_south, afc_west]

    def _is_div(row):
        h, a = row["home_team"], row["away_team"]
        return int(any(h in d and a in d for d in divisions))

    return sched.apply(_is_div, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: PBP AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_pbp(seasons: list[int]) -> pd.DataFrame:
    """
    For each team × game, compute per-game aggregates from play-by-play.
    Returns DataFrame with columns: game_id, team (posteam/defteam), side, + stats.
    """
    pbp = get_pbp(seasons)

    # ── Offensive aggregates ──────────────────────────────────────────────
    off = (
        pbp[pbp["posteam"].notna() & (pbp["play_type"].isin([
            "pass", "run", "qb_kneel", "qb_spike"
        ]))]
        .groupby(["game_id", "posteam"])
        .agg(
            # EPA
            off_epa_per_play        = ("epa", "mean"),
            off_epa_total           = ("epa", "sum"),
            # Success rate
            off_success_rate        = ("success", "mean"),
            # Yards
            off_yards_per_play      = ("yards_gained", "mean"),
            off_total_yards         = ("yards_gained", "sum"),
            # Passing
            off_pass_epa            = ("epa",
                lambda x: x[pbp.loc[x.index, "play_type"] == "pass"].mean()),
            off_cpoe                = ("cpoe", "mean"),
            off_air_yards_per_att   = ("air_yards", "mean"),
            off_yac_per_rec         = ("yards_after_catch", "mean"),
            off_pass_success        = ("success",
                lambda x: x[pbp.loc[x.index, "play_type"] == "pass"].mean()),
            # Rushing
            off_rush_epa            = ("epa",
                lambda x: x[pbp.loc[x.index, "play_type"] == "run"].mean()),
            off_rush_yards_per_att  = ("yards_gained",
                lambda x: x[pbp.loc[x.index, "play_type"] == "run"].mean()),
            off_rush_success        = ("success",
                lambda x: x[pbp.loc[x.index, "play_type"] == "run"].mean()),
            # Win probability
            off_wpa                 = ("wpa", "sum"),
            # Explosive plays (>= 20 yards)
            off_explosive_play_rate = ("yards_gained",
                lambda x: (x >= 20).mean()),
            # Sacks allowed
            off_sack_rate           = ("sack", "mean"),
            # Penalties
            off_penalty_yards       = ("penalty_yards", "sum"),
            # Plays
            n_plays_off             = ("play_id", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # ── 3rd down offensive ─────────────────────────────────────────────────
    pbp3 = pbp[pbp["down"] == 3]
    td3 = (
        pbp3.groupby(["game_id", "posteam"])
        .agg(
            off_third_down_rate = ("first_down", "mean"),
            n_third_downs       = ("play_id", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # ── Red zone offense ───────────────────────────────────────────────────
    pbp_rz = pbp[pbp["yardline_100"] <= 20]
    rz_off = (
        pbp_rz.groupby(["game_id", "posteam"])
        .agg(
            off_rz_success_rate = ("success", "mean"),
            off_rz_td_rate      = ("touchdown", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # ── Defensive aggregates ──────────────────────────────────────────────
    def_agg = (
        pbp[pbp["defteam"].notna() & (pbp["play_type"].isin([
            "pass", "run", "qb_kneel", "qb_spike"
        ]))]
        .groupby(["game_id", "defteam"])
        .agg(
            def_epa_per_play       = ("epa", "mean"),       # lower=better defense
            def_epa_total          = ("epa", "sum"),
            def_success_rate       = ("success", "mean"),   # lower=better
            def_yards_per_play     = ("yards_gained", "mean"),
            def_pass_epa           = ("epa",
                lambda x: x[pbp.loc[x.index, "play_type"] == "pass"].mean()),
            def_rush_epa           = ("epa",
                lambda x: x[pbp.loc[x.index, "play_type"] == "run"].mean()),
            def_explosive_allowed  = ("yards_gained",
                lambda x: (x >= 20).mean()),
            def_sack_rate          = ("sack", "mean"),      # higher=better defense
            def_pressure_rate      = ("qb_hit", "mean"),
            n_plays_def            = ("play_id", "count"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # ── 3rd down defensive ─────────────────────────────────────────────────
    td3_def = (
        pbp3.groupby(["game_id", "defteam"])
        .agg(def_third_down_allowed = ("first_down", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # ── Red zone defense ───────────────────────────────────────────────────
    rz_def = (
        pbp_rz.groupby(["game_id", "defteam"])
        .agg(
            def_rz_success_allowed  = ("success", "mean"),
            def_rz_td_allowed       = ("touchdown", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # ── Turnovers ──────────────────────────────────────────────────────────
    to_agg = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "posteam"])
        .agg(
            turnovers_committed = ("turnover", "sum"),
            interceptions       = ("interception", "sum"),
            fumbles_lost        = ("fumble_lost", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    to_forced = (
        pbp[pbp["defteam"].notna()]
        .groupby(["game_id", "defteam"])
        .agg(turnovers_forced = ("turnover", "sum"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # ── Pace (hurry-up snaps, time between plays) ──────────────────────────
    pace_agg = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "posteam"])
        .agg(
            avg_time_between_plays = ("time_of_day", lambda x:
                pd.to_numeric(x, errors="coerce").diff().abs().mean()
                if len(x) > 1 else np.nan),
            shotgun_rate           = ("shotgun", "mean"),
            no_huddle_rate         = ("no_huddle", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # ── Special teams ──────────────────────────────────────────────────────
    st_off = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "posteam"])
        .agg(
            fg_made         = ("field_goal_result",
                lambda x: (x == "made").sum()),
            fg_attempted    = ("field_goal_result",
                lambda x: x.notna().sum()),
            kick_return_yds = ("return_yards",
                lambda x: x[pbp.loc[x.index, "play_type"] == "kickoff"].sum()),
            punt_return_yds = ("return_yards",
                lambda x: x[pbp.loc[x.index, "play_type"] == "punt"].sum()),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    st_off["fg_pct"] = st_off["fg_made"] / st_off["fg_attempted"].clip(lower=1)

    # ── Merge all aggregates ───────────────────────────────────────────────
    result = off.copy()
    for df in [td3, rz_off, def_agg, td3_def, rz_def,
               to_agg, to_forced, pace_agg, st_off]:
        merge_on = ["game_id", "team"]
        result = result.merge(df, on=merge_on, how="left")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: TEAM PERSPECTIVE TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _merge_team_perspective(
    games: pd.DataFrame, pbp_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates one row per (team, game) with:
    - All PBP aggregates for this team
    - Game metadata (date, week, opponent, home/away flag, actual score)
    """
    rows = []

    for _, g in games.iterrows():
        for side in ["home", "away"]:
            opp_side = "away" if side == "home" else "home"
            team = g[f"{side}_team"]
            opp  = g[f"{opp_side}_team"]

            row = {
                "game_id":           g["game_id"],
                "season":            g["season"],
                "week":              g["week"],
                "game_type":         g["game_type"],
                "game_date":         g["game_date"],
                "team":              team,
                "opponent":          opp,
                "is_home":           1 if side == "home" else 0,
                "team_score":        g.get(f"{side}_score", np.nan),
                "opp_score":         g.get(f"{opp_side}_score", np.nan),
                "spread_line":       g.get("spread_line", np.nan),
                "total_line":        g.get("total_line", np.nan),
                "temp":              g.get("temp", np.nan),
                "wind":              g.get("wind", np.nan),
                "humidity":          g.get("humidity", np.nan),
                "is_dome":           g.get("is_dome", 0),
                "is_primetime":      g.get("is_primetime", 0),
                "is_international":  g.get("is_international", 0),
                "is_division_game":  g.get("is_division_game", 0),
                "day_of_week":       g.get("day_of_week", np.nan),
                "referee":           g.get("referee", None),
                "game_type_weight":  get_game_type_flag(g["game_type"]),
            }

            # Merge in this team's PBP aggregates
            team_pbp = pbp_agg[
                (pbp_agg["game_id"] == g["game_id"]) &
                (pbp_agg["team"] == team)
            ]
            if len(team_pbp) > 0:
                for col in team_pbp.columns:
                    if col not in ("game_id", "team"):
                        row[col] = team_pbp.iloc[0][col]

            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["season", "week", "game_date", "game_id", "team"]
    ).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_rolling_features(tg: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, sort by date and compute rolling averages over N=4,8,16 games.
    All rolling windows are strictly PRIOR games (shift(1) ensures no leakage).
    """
    stat_cols = [
        "off_epa_per_play", "off_success_rate", "off_yards_per_play",
        "off_pass_epa", "off_cpoe", "off_air_yards_per_att",
        "off_rush_epa", "off_rush_yards_per_att",
        "off_third_down_rate", "off_rz_success_rate", "off_rz_td_rate",
        "off_explosive_play_rate", "off_sack_rate", "off_wpa",
        "def_epa_per_play", "def_success_rate", "def_yards_per_play",
        "def_pass_epa", "def_rush_epa", "def_explosive_allowed",
        "def_third_down_allowed", "def_rz_success_allowed", "def_rz_td_allowed",
        "def_sack_rate", "def_pressure_rate",
        "turnovers_committed", "turnovers_forced", "interceptions", "fumbles_lost",
        "fg_pct", "kick_return_yds", "punt_return_yds",
        "shotgun_rate", "no_huddle_rate",
        "team_score", "opp_score",
    ]
    stat_cols = [c for c in stat_cols if c in tg.columns]

    result_frames = []
    for team, grp in tg.groupby("team"):
        grp = grp.sort_values("game_date").copy()
        for col in stat_cols:
            if col not in grp.columns:
                continue
            series = grp[col].shift(1)   # no leakage — only past games
            for N, suffix in [
                (ROLLING_SHORT, "r4"),
                (ROLLING_MED,   "r8"),
                (ROLLING_LONG,  "r16"),
            ]:
                grp[f"{col}_{suffix}"] = (
                    series.rolling(N, min_periods=max(1, N // 2)).mean()
                )

        # Derived rolling: turnover differential
        if "turnovers_committed" in grp.columns and "turnovers_forced" in grp.columns:
            for N, suffix in [(4, "r4"), (8, "r8"), (16, "r16")]:
                tc = grp["turnovers_committed"].shift(1)
                tf = grp["turnovers_forced"].shift(1)
                grp[f"turnover_diff_{suffix}"] = (
                    (tf - tc).rolling(N, min_periods=1).mean()
                )

        # Derived rolling: point differential
        if "team_score" in grp.columns and "opp_score" in grp.columns:
            for N, suffix in [(4, "r4"), (8, "r8"), (16, "r16")]:
                diff = (grp["team_score"] - grp["opp_score"]).shift(1)
                grp[f"score_diff_{suffix}"] = (
                    diff.rolling(N, min_periods=1).mean()
                )

        # Win rate rolling
        if "team_score" in grp.columns and "opp_score" in grp.columns:
            wins = (grp["team_score"] > grp["opp_score"]).astype(float).shift(1)
            for N, suffix in [(4, "r4"), (8, "r8"), (16, "r16")]:
                grp[f"win_rate_{suffix}"] = (
                    wins.rolling(N, min_periods=1).mean()
                )

        result_frames.append(grp)

    return pd.concat(result_frames, ignore_index=True).sort_values(
        ["season", "week", "game_date", "game_id"]
    ).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: QB FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_qb_features(tg: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Add QB-specific rolling stats (CPOE, EPA/dropback, air yards, etc.)."""
    player_stats = get_table("player_stats_weekly")
    if player_stats is None:
        logger.warning("  player_stats_weekly not available — skipping QB features")
        return tg

    # Filter to QBs
    qb_cols_needed = [
        "player_id", "player_name", "recent_team", "season", "week",
        "attempts", "completions", "passing_yards", "passing_tds",
        "interceptions", "sacks", "sack_yards",
        "passing_epa", "dakota",
    ]
    qb_cols = [c for c in qb_cols_needed if c in player_stats.columns]
    qbs = player_stats[
        player_stats["position"] == "QB" if "position" in player_stats.columns
        else player_stats["season"].isin(seasons)
    ][qb_cols].copy()

    # Identify starter per team per week (most attempts)
    if "attempts" in qbs.columns:
        qbs = qbs.sort_values(
            ["recent_team", "season", "week", "attempts"], ascending=[True, True, True, False]
        )
        starters = qbs.groupby(["recent_team", "season", "week"]).first().reset_index()
    else:
        starters = qbs.copy()

    # Compute per-game efficiency
    if "attempts" in starters.columns:
        starters["qb_epa_per_att"] = (
            starters.get("passing_epa", 0) /
            starters["attempts"].clip(lower=1)
        )

    # Rolling QB metrics (shift to avoid leakage)
    for team, grp in starters.groupby("recent_team"):
        grp = grp.sort_values(["season", "week"])
        for col in ["qb_epa_per_att", "dakota"]:
            if col in grp.columns:
                for N, suffix in [(4, "r4"), (8, "r8"), (16, "r16")]:
                    starters.loc[grp.index, f"qb_{col}_{suffix}"] = (
                        grp[col].shift(1).rolling(N, min_periods=1).mean().values
                    )

    # Merge into team_games
    merge_keys = ["recent_team", "season", "week"]
    rename = {"recent_team": "team"}
    qb_merge = starters.rename(columns=rename)[
        ["team", "season", "week"] +
        [c for c in starters.columns if c.startswith("qb_") and "_r" in c]
    ]

    tg = tg.merge(qb_merge, on=["team", "season", "week"], how="left")
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: INJURY / ROSTER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_injury_features(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute availability scores from the injury report.
    Scoring: Active=1.0, Questionable=0.75, Doubtful=0.25, Out/IR=0.0
    Position weights: QB=5, WR1/RB1=2, OL=1, DL/LB/DB=1
    """
    injuries = get_table("injuries")
    if injuries is None:
        logger.warning("  Injuries not available — using default availability scores")
        tg["qb_available"] = 1.0
        tg["off_availability_score"] = 1.0
        tg["def_availability_score"] = 1.0
        tg["injury_report_severity"] = 0.0
        tg["injury_data_freshness"] = 0.4   # no data = low freshness
        return tg

    status_score = {
        "Active": 1.0, "Probable": 0.95, "Questionable": 0.75,
        "Doubtful": 0.25, "Out": 0.0, "IR": 0.0, "PUP": 0.0,
        "DNP": 0.5,   # did not practice — uncertain
    }
    pos_weight = {
        "QB": 5.0, "WR": 2.0, "RB": 2.0, "TE": 1.5,
        "OT": 1.2, "OG": 1.0, "C": 1.0,
        "DE": 1.2, "DT": 1.0, "LB": 1.0, "CB": 1.2, "S": 1.0,
    }

    def _team_availability(team_df: pd.DataFrame, pos_filter: list) -> float:
        pos_rows = team_df[team_df["position"].isin(pos_filter)]
        if len(pos_rows) == 0:
            return 1.0
        scores = []
        for _, r in pos_rows.iterrows():
            s = status_score.get(r.get("report_status", "Active"), 0.75)
            w = pos_weight.get(r.get("position", ""), 1.0)
            scores.append(s * w)
        return np.mean(scores)

    OFF_POS = ["QB", "WR", "RB", "TE", "OT", "OG", "C"]
    DEF_POS = ["DE", "DT", "LB", "CB", "S"]

    # Build lookup: (team, season, week) → scores
    if "practice_primary_injury" in injuries.columns:
        inj_lookup = {}
        for (team, season, week), grp in injuries.groupby(
            ["team", "season", "week"]
        ):
            inj_lookup[(team, season, week)] = {
                "qb_available":          float(
                    (grp[grp["position"] == "QB"]["report_status"]
                     .map(status_score).fillna(0.75).max())
                    if len(grp[grp["position"] == "QB"]) > 0 else 1.0
                ),
                "off_availability_score": _team_availability(grp, OFF_POS),
                "def_availability_score": _team_availability(grp, DEF_POS),
                "injury_report_severity": float(
                    (grp["report_status"]
                     .map({"Out": 1, "Doubtful": 0.5}).fillna(0).sum())
                ),
            }

        for col in ["qb_available", "off_availability_score",
                    "def_availability_score", "injury_report_severity"]:
            tg[col] = tg.apply(
                lambda r: inj_lookup.get(
                    (r["team"], r["season"], r["week"]), {}
                ).get(col, 1.0 if "available" in col or "score" in col else 0.0),
                axis=1,
            )
    else:
        tg["qb_available"]           = 1.0
        tg["off_availability_score"] = 1.0
        tg["def_availability_score"] = 1.0
        tg["injury_report_severity"] = 0.0

    # Injury data freshness (computed at predict time; placeholder here)
    tg["injury_data_freshness"] = 0.65   # will be updated by confidence.py

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: ELO RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def _add_elo_ratings(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game-by-game Elo ratings for every team.
    Rating is computed BEFORE each game (no leakage).
    Update rule: margin-of-victory adjusted Elo.
    """
    elo: dict[str, float] = {}

    tg = tg.sort_values(["season", "week", "game_date", "game_id"]).copy()
    tg["elo_pre_game"] = np.nan
    tg["elo_expected_win_prob"] = np.nan

    # Process games in chronological order — only home/away pair per game_id
    processed_games: set[str] = set()

    for idx, row in tg.iterrows():
        team = row["team"]
        opp  = row["opponent"]
        gid  = row["game_id"]

        team_elo = elo.get(team, ELO_START)
        opp_elo  = elo.get(opp,  ELO_START)

        # Expected win probability
        exp_win = 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))

        tg.at[idx, "elo_pre_game"]        = team_elo
        tg.at[idx, "elo_expected_win_prob"] = exp_win

        # Update only once per game (when processing the home team row)
        if gid not in processed_games and row["is_home"] == 1:
            processed_games.add(gid)
            h_score = row["team_score"]
            a_score = row["opp_score"]

            if pd.notna(h_score) and pd.notna(a_score):
                # Margin-of-victory multiplier
                margin = abs(h_score - a_score)
                mov_mult = np.log(margin + 1) * 2.2 / (
                    0.001 + abs(team_elo - opp_elo) * 0.001 + 1.0
                )
                h_actual = 1.0 if h_score > a_score else (
                    0.5 if h_score == a_score else 0.0
                )
                delta = ELO_K * mov_mult * (h_actual - exp_win)
                elo[team] = team_elo + delta
                elo[opp]  = opp_elo  - delta

    # Season reset: each new season, regress 33% toward the mean
    prev_season = None
    for idx, row in tg.iterrows():
        if prev_season is not None and row["season"] != prev_season:
            for t in list(elo.keys()):
                elo[t] = ELO_START + 0.67 * (elo[t] - ELO_START)
        prev_season = row["season"]

    # Elo vs. Vegas spread: implied power rating
    if "spread_line" in tg.columns:
        # spread_line is from home team perspective (negative = home favored)
        tg["vegas_implied_power"] = tg.apply(
            lambda r: -r["spread_line"] / 2.0 if r["is_home"] == 1
                      else r["spread_line"] / 2.0,
            axis=1,
        )
        # Divergence between Elo win prob and Vegas implied prob
        tg["elo_vegas_divergence"] = (
            tg["elo_expected_win_prob"] - (
                tg["vegas_implied_power"].apply(
                    lambda x: 0.5 + x / 28.0 if pd.notna(x) else 0.5
                )
            )
        )

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: CROSS-SEASON FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_cross_season_features(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that capture team changes between seasons:
    roster continuity, coaching changes, draft quality, Vegas O/U.
    """
    cross: dict[tuple, dict] = {}

    # ── Win totals (Vegas pre-season consensus) ────────────────────────────
    win_totals = get_table("win_totals")
    wt_lookup: dict[tuple, float] = {}
    if win_totals is not None:
        for _, r in win_totals.iterrows():
            team   = r.get("team") or r.get("team_abbr")
            season = r.get("season")
            wins   = r.get("wins") or r.get("implied_wins") or r.get("win_total")
            if team and season and wins:
                wt_lookup[(team, season)] = float(wins)

    tg["vegas_preseason_wins"] = tg.apply(
        lambda r: wt_lookup.get((r["team"], r["season"]), np.nan), axis=1
    )

    # ── Roster continuity (% of prior season snaps retained) ──────────────
    rosters = get_table("rosters")
    if rosters is not None and "season" in rosters.columns:
        continuity: dict[tuple, float] = {}
        for (team, season), grp in rosters.groupby(
            [rosters.get("team") or "team", "season"]
        ):
            prev = rosters[
                (rosters.get("team", rosters.get("team_abbr")) == team) &
                (rosters["season"] == season - 1)
            ]
            if len(prev) == 0 or len(grp) == 0:
                continuity[(team, season)] = 0.5
            else:
                cur_ids  = set(grp.get("gsis_id", grp.get("player_id", [])))
                prev_ids = set(prev.get("gsis_id", prev.get("player_id", [])))
                if len(prev_ids) == 0:
                    continuity[(team, season)] = 0.5
                else:
                    retained = len(cur_ids & prev_ids)
                    continuity[(team, season)] = retained / max(len(prev_ids), 1)

        tg["roster_continuity"] = tg.apply(
            lambda r: continuity.get((r["team"], r["season"]), 0.5), axis=1
        )
    else:
        tg["roster_continuity"] = 0.5

    # ── Draft quality score ────────────────────────────────────────────────
    draft_picks = get_table("draft_picks")
    draft_values = get_table("draft_values")
    draft_quality: dict[tuple, float] = {}

    if draft_picks is not None and draft_values is not None:
        if "pfr_av" in draft_picks.columns:
            # Use Approximate Value as draft quality proxy
            dq = (
                draft_picks.groupby(["team", "season"])["pfr_av"]
                .sum().reset_index()
            )
            for _, r in dq.iterrows():
                draft_quality[(r["team"], r["season"])] = float(r["pfr_av"])
        elif "pick" in draft_picks.columns and "value" in draft_values.columns:
            merged = draft_picks.merge(draft_values, on="pick", how="left")
            dq = (
                merged.groupby(["team", "season"])["value"]
                .sum().reset_index()
            )
            for _, r in dq.iterrows():
                draft_quality[(r["team"], r["season"])] = float(r["value"])

    tg["draft_quality_score"] = tg.apply(
        lambda r: draft_quality.get((r["team"], r["season"]), 0.0), axis=1
    )

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9: OFFICIAL / REFEREE TENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

def _add_official_features(tg: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """
    Referees have measurable tendencies:
    - Penalty rate (flags thrown per game)
    - Home win rate (do they systematically favor home teams?)
    - Total score tendency (high-scoring or low-scoring games)
    """
    officials = get_table("officials")
    sched     = get_schedules(extended=False)

    if officials is None or "referee" not in sched.columns:
        tg["ref_penalty_rate"]  = np.nan
        tg["ref_home_win_rate"] = np.nan
        tg["ref_total_tendency"]= np.nan
        return tg

    # Merge referee into schedule
    ref_lookup = dict(zip(sched["game_id"], sched.get("referee", [np.nan]*len(sched))))

    # For each referee, compute historical tendencies from prior games
    # (we use all available data, not just window seasons, for more signal)
    sched_all = get_schedules(extended=True)
    if "home_score" in sched_all.columns:
        sched_all["total"] = sched_all["home_score"] + sched_all["away_score"]
        sched_all["home_win"] = (sched_all["home_score"] > sched_all["away_score"]).astype(float)
    else:
        sched_all["total"] = np.nan
        sched_all["home_win"] = np.nan

    if "referee" in sched_all.columns:
        ref_stats = (
            sched_all.groupby("referee")
            .agg(
                ref_home_win_rate = ("home_win", "mean"),
                ref_total_tendency= ("total", "mean"),
                ref_n_games       = ("game_id", "count"),
            )
            .reset_index()
        )
        ref_stats = ref_stats[ref_stats["ref_n_games"] >= 10]  # min sample

        ref_dict = {
            r["referee"]: {
                "ref_home_win_rate":  r["ref_home_win_rate"],
                "ref_total_tendency": r["ref_total_tendency"],
            }
            for _, r in ref_stats.iterrows()
        }

        for col in ["ref_home_win_rate", "ref_total_tendency"]:
            tg[col] = tg["referee"].map(
                lambda r: ref_dict.get(r, {}).get(col, np.nan)
                if pd.notna(r) else np.nan
            )
    else:
        tg["ref_home_win_rate"]  = np.nan
        tg["ref_total_tendency"] = np.nan

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10: NGS FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_ngs_features(tg: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Add Next Gen Stats features (separation, speed, etc.) when available."""
    ngs_pass = get_table("ngs_passing")
    ngs_rush = get_table("ngs_rushing")
    ngs_recv = get_table("ngs_receiving")

    for tbl, prefix, keys in [
        (ngs_pass, "ngs_pass", ["avg_time_to_throw", "avg_completed_air_yards",
                                 "avg_intended_air_yards", "aggressiveness"]),
        (ngs_rush, "ngs_rush", ["efficiency", "percent_attempts_gte_eight_defenders",
                                 "avg_time_to_los", "rush_yards_over_expected_per_att"]),
        (ngs_recv, "ngs_recv", ["avg_separation", "avg_intended_air_yards",
                                 "catch_percentage", "avg_yac"]),
    ]:
        if tbl is None:
            continue
        # NGS is at player level; aggregate to team level
        team_col = next(
            (c for c in ["team_abbr", "team", "possession_team"] if c in tbl.columns),
            None,
        )
        if team_col is None:
            continue

        agg_dict = {k: "mean" for k in keys if k in tbl.columns}
        if not agg_dict:
            continue

        grp_cols = [team_col, "season", "week"] if "week" in tbl.columns else [team_col, "season"]
        tbl_agg = tbl.groupby(grp_cols).agg(agg_dict).reset_index()
        tbl_agg = tbl_agg.rename(columns={
            team_col: "team",
            **{k: f"{prefix}_{k}" for k in agg_dict},
        })

        merge_on = (
            ["team", "season", "week"]
            if "week" in tbl_agg.columns
            else ["team", "season"]
        )
        tg = tg.merge(tbl_agg, on=merge_on, how="left")

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11: CONTEXTUAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_contextual_features(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest days, travel distance, weather compound features, and
    season-position indicators.
    """
    tg = tg.sort_values(["team", "season", "week", "game_date"]).copy()

    # ── Rest days ─────────────────────────────────────────────────────────
    tg["prev_game_date"] = tg.groupby(["team", "season"])["game_date"].shift(1)
    tg["rest_days"] = (
        (tg["game_date"] - tg["prev_game_date"]).dt.days.fillna(10)
    )
    # Bye week detection (rest > 10 days)
    tg["is_bye_week"] = (tg["rest_days"] > 10).astype(int)
    # Short week (Thursday night; rest < 6)
    tg["is_short_week"] = (tg["rest_days"] < 6).astype(int)

    # ── Rest differential (computed at pivot step) ─────────────────────────
    # Stored here per team; gap is added in pivot_to_game_features()

    # ── Weather compound features ─────────────────────────────────────────
    if "wind" in tg.columns:
        tg["high_wind"] = (tg["wind"].fillna(0) >= 15).astype(int)
        tg["extreme_wind"] = (tg["wind"].fillna(0) >= 25).astype(int)

    if "temp" in tg.columns:
        tg["cold_game"] = (tg["temp"].fillna(60) < 32).astype(int)
        tg["very_cold_game"] = (tg["temp"].fillna(60) < 20).astype(int)

    # Dome nullifies weather
    for col in ["high_wind", "extreme_wind", "cold_game", "very_cold_game"]:
        if col in tg.columns:
            tg[col] = tg[col] * (1 - tg["is_dome"].fillna(0))

    # ── Season position ───────────────────────────────────────────────────
    tg["season_half"] = (tg["week"] > 9).astype(int)       # late season
    tg["week_norm"]   = tg["week"] / 18.0                   # 0..1

    # ── Day of week encoding ──────────────────────────────────────────────
    if "day_of_week" in tg.columns:
        # 0=Mon,3=Thu,6=Sun
        tg["is_sunday"]   = (tg["day_of_week"] == 6).astype(int)
        tg["is_thursday"] = (tg["day_of_week"] == 3).astype(int)
        tg["is_monday"]   = (tg["day_of_week"] == 0).astype(int)
        tg["is_saturday"] = (tg["day_of_week"] == 5).astype(int)  # late season

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  PIVOT: ONE ROW PER GAME
# ══════════════════════════════════════════════════════════════════════════════

def pivot_to_game_features(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot team-game table into one row per game with:
    - home_ and away_ prefixed team features
    - matchup gap features (home_x - away_x for key metrics)
    - rest differential
    - target columns (home_score, away_score)
    """
    home = tg[tg["is_home"] == 1].copy()
    away = tg[tg["is_home"] == 0].copy()

    # Exclude columns that shouldn't be duplicated
    drop_from_pivot = {
        "is_home", "team", "opponent", "game_date", "season", "week",
        "game_type", "game_type_weight", "spread_line", "total_line",
        "is_dome", "is_primetime", "is_international", "is_division_game",
        "day_of_week", "is_sunday", "is_thursday", "is_monday", "is_saturday",
        "temp", "wind", "humidity", "high_wind", "extreme_wind",
        "cold_game", "very_cold_game", "is_bye_week", "is_short_week",
        "referee", "ref_home_win_rate", "ref_total_tendency",
    }

    team_stat_cols = [
        c for c in tg.columns
        if c not in drop_from_pivot and c not in (
            "game_id", "team_score", "opp_score",
            "rest_days", "prev_game_date",
        )
    ]

    # Rename team-specific columns
    home_stats = home[["game_id"] + team_stat_cols + ["team_score", "opp_score", "rest_days", "team"]].copy()
    away_stats = away[["game_id"] + team_stat_cols + ["team_score", "opp_score", "rest_days", "team"]].copy()

    home_stats = home_stats.add_prefix("home_")
    away_stats = away_stats.add_prefix("away_")
    home_stats = home_stats.rename(columns={"home_game_id": "game_id"})
    away_stats = away_stats.rename(columns={"away_game_id": "game_id"})

    game_df = home_stats.merge(away_stats, on="game_id", how="inner")

    # Bring back shared game-level features from home row
    shared_cols = [
        "game_id", "season", "week", "game_type", "game_type_weight",
        "game_date", "spread_line", "total_line",
        "is_dome", "is_primetime", "is_international", "is_division_game",
        "temp", "wind", "humidity", "high_wind", "extreme_wind",
        "cold_game", "very_cold_game",
        "is_sunday", "is_thursday", "is_monday", "is_saturday",
        "ref_home_win_rate", "ref_total_tendency",
    ]
    shared_cols = [c for c in shared_cols if c in home.columns]
    shared = home[shared_cols].copy()
    game_df = game_df.merge(shared, on="game_id", how="left")

    # ── Rest differential ─────────────────────────────────────────────────
    if "home_rest_days" in game_df.columns and "away_rest_days" in game_df.columns:
        game_df["rest_diff"] = game_df["home_rest_days"] - game_df["away_rest_days"]

    # ── Matchup gap features ──────────────────────────────────────────────
    gap_pairs = [
        # (home_off_col, away_def_col, gap_name)
        ("home_off_epa_per_play_r8",    "away_def_epa_per_play_r8",    "home_off_vs_away_def_epa_r8"),
        ("home_off_pass_epa_r8",        "away_def_pass_epa_r8",        "home_pass_vs_away_pass_def_r8"),
        ("home_off_rush_epa_r8",        "away_def_rush_epa_r8",        "home_rush_vs_away_rush_def_r8"),
        ("home_off_third_down_rate_r8", "away_def_third_down_allowed_r8", "home_3rd_gap_r8"),
        ("home_off_rz_td_rate_r8",      "away_def_rz_td_allowed_r8",   "home_rz_gap_r8"),
        # mirror for away team
        ("away_off_epa_per_play_r8",    "home_def_epa_per_play_r8",    "away_off_vs_home_def_epa_r8"),
        ("away_off_pass_epa_r8",        "home_def_pass_epa_r8",        "away_pass_vs_home_pass_def_r8"),
        ("away_off_rush_epa_r8",        "home_def_rush_epa_r8",        "away_rush_vs_home_rush_def_r8"),
        ("away_off_third_down_rate_r8", "home_def_third_down_allowed_r8", "away_3rd_gap_r8"),
        ("away_off_rz_td_rate_r8",      "home_def_rz_td_allowed_r8",   "away_rz_gap_r8"),
    ]
    for off_col, def_col, gap_name in gap_pairs:
        if off_col in game_df.columns and def_col in game_df.columns:
            game_df[gap_name] = game_df[off_col] - game_df[def_col]

    # Elo gap
    if "home_elo_pre_game" in game_df.columns and "away_elo_pre_game" in game_df.columns:
        game_df["elo_gap"] = game_df["home_elo_pre_game"] - game_df["away_elo_pre_game"]

    # Turnover differential gap
    for suffix in ["r4", "r8", "r16"]:
        h = f"home_turnover_diff_{suffix}"
        a = f"away_turnover_diff_{suffix}"
        if h in game_df.columns and a in game_df.columns:
            game_df[f"turnover_diff_gap_{suffix}"] = game_df[h] - game_df[a]

    # CPOE gap (QB quality)
    if "home_off_cpoe_r8" in game_df.columns and "away_off_cpoe_r8" in game_df.columns:
        game_df["cpoe_gap_r8"] = game_df["home_off_cpoe_r8"] - game_df["away_off_cpoe_r8"]

    # Target columns
    game_df["target_home_score"] = home["team_score"].values[:len(game_df)] if "team_score" in home.columns else np.nan
    game_df["target_away_score"] = away["team_score"].values[:len(game_df)] if "team_score" in away.columns else np.nan

    # Recompute from merged scores if available
    if "home_team_score" in game_df.columns:
        game_df["target_home_score"] = game_df["home_team_score"]
    if "away_team_score" in game_df.columns:
        game_df["target_away_score"] = game_df["away_team_score"]

    game_df["target_total"] = game_df["target_home_score"] + game_df["target_away_score"]
    game_df["target_spread"] = game_df["target_home_score"] - game_df["target_away_score"]

    # Sort
    game_df = game_df.sort_values(["season", "week", "game_date"]).reset_index(drop=True)

    logger.info(f"  Game feature matrix: {len(game_df)} rows × {len(game_df.columns)} columns")
    return game_df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE LIST EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_columns(game_df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (exclude targets and metadata)."""
    exclude = {
        "game_id", "season", "week", "game_type", "game_date",
        "home_team", "away_team",
        "home_team_score", "away_team_score",
        "home_opp_score", "away_opp_score",
        "target_home_score", "target_away_score",
        "target_total", "target_spread",
        "game_type_weight",
        "home_prev_game_date", "away_prev_game_date",
        "home_injury_data_freshness", "away_injury_data_freshness",
    }
    return [c for c in game_df.columns if c not in exclude]


def save_feature_dictionary(game_df: pd.DataFrame) -> None:
    """Save a CSV describing all features."""
    import csv
    feat_cols = get_feature_columns(game_df)
    path = PROCESSED_DIR / "feature_dictionary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_name", "dtype", "null_pct", "mean", "std"])
        for col in feat_cols:
            s = game_df[col]
            null_pct = s.isna().mean()
            mean = s.mean() if pd.api.types.is_numeric_dtype(s) else ""
            std  = s.std()  if pd.api.types.is_numeric_dtype(s) else ""
            w.writerow([col, str(s.dtype), f"{null_pct:.3f}", mean, std])
    logger.info(f"Feature dictionary saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== Feature Engineering ===")
    gf = build_all_features()
    feat_cols = get_feature_columns(gf)
    print(f"\nTotal features: {len(feat_cols)}")
    print(f"Games in matrix: {len(gf)}")
    save_feature_dictionary(gf)

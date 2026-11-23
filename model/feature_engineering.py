"""
feature_engineering.py — builds the game-level feature matrix.

Two rules decide whether a column may become a model feature:
  1. it is knowable before kickoff of the game it describes (no same-game
     statistics, no post-game ratings), and
  2. it is also AVAILABLE for genuinely upcoming games when the weekly job
     runs. A column that exists for every historical game but is always
     missing for the next game (game-day weather, referee crews, final injury
     statuses) teaches the model a pattern it can never use live.

Rolling statistics are computed on each team's full game timeline (played
and upcoming games): the value for a game is the mean over the team's last N
PLAYED games before it, so the next upcoming game sees exactly the window a
historical game saw in training.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import settings

logger = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOWS = ((4, "r4"), (8, "r8"), (16, "r16"))

# Feature families that only exist from a given season on (lets the tree
# models tell "not measured in this era" apart from a real value).
COVERAGE_COMPLEXITY_MIN_SEASON = 2019
WEPA_MIN_SEASON = 2014
NGS_MIN_SEASON  = 2016

# Relocated franchises: older schedule rows use the old abbreviation.
# Team-history features (Elo, head-to-head, QB/coach changes, HFA) follow
# the franchise, not the city.
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

# Fallback only, for games whose `roof` is not filled in yet (retractable
# roofs are mostly decided on game day). nflverse uses "LA" for the Rams.
_INDOOR_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LA", "LAC", "LV", "MIN", "NO"}


def _franchise(s: pd.Series) -> pd.Series:
    return s.replace(FRANCHISE)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_all_features(seasons: Optional[list] = None) -> pd.DataFrame:
    """Build the full game-level feature matrix. Returns one row per game."""
    from data_loader import ALL_SEASONS
    seasons = seasons or ALL_SEASONS
    logger.info("Building game feature matrix for seasons %s …", seasons)

    games = _build_game_base(seasons)
    logger.info("  Base games: %d", len(games))
    pbp_agg = _aggregate_pbp(seasons)
    logger.info("  PBP agg: %d team-game rows", len(pbp_agg))

    tg = _build_team_games(games, pbp_agg)
    tg = _add_rolling(tg)
    tg = _add_qb_form(tg)
    tg = _add_ngs_features(tg)

    team_history = _schedule_team_games()     # every game since 1999, long format
    tg = _add_elo(tg, team_history)
    tg = _add_qb_and_coach_changes(tg, team_history)
    tg = _add_roster_continuity(tg)
    tg = _add_contextual(tg, team_history)

    game_df = _pivot_to_game(tg)
    game_df.to_parquet(PROCESSED_DIR / "game_features.parquet", index=False)
    logger.info("  Final game matrix: %d rows × %d cols", len(game_df), len(game_df.columns))
    return game_df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ALLOWLIST — anything not allowed here never reaches the model
# ══════════════════════════════════════════════════════════════════════════════

# Lagged rolling statistics (see _lagged_rolling).
_ROLLING_SUFFIXES = ("_r4", "_r8", "_r16")

# Per-team columns; they appear as home_<base> / away_<base> after the pivot.
_PREGAME_SIDE_BASES = {
    "elo_pre_game", "elo_expected_win_prob", "vegas_implied_power", "elo_vegas_divergence",
    "rest_days", "is_bye_week", "is_short_week",
    "roster_continuity", "qb_changed", "qb_start_share", "coach_changed",
    "team_instability", "team_specific_hfa",
}

# Game-level columns.
_PREGAME_SHARED = {
    "spread_line", "total_line",
    "is_dome", "is_primetime", "is_international", "is_division_game",
    "is_sunday", "is_thursday", "is_monday", "is_saturday",
    "season_avg_total", "scoring_era_adj", "rolling_hfa",
    "rest_diff", "elo_gap", "elo_vs_spread", "coverage_complexity_gap",
    "wepa_available", "coverage_available", "ngs_available",
    "week_norm", "season_half",
}


def get_feature_columns(game_df: pd.DataFrame) -> list:
    """Model features = numeric columns allowed by the allowlist above, plus
    h2h_* (computed from strictly earlier games). Logs what it drops."""
    allowed, dropped = [], []
    for c in game_df.columns:
        if not pd.api.types.is_numeric_dtype(game_df[c]) or c.startswith("target_"):
            continue
        if (c.endswith(_ROLLING_SUFFIXES) or c.startswith("h2h_") or c in _PREGAME_SHARED
                or (c.startswith(("home_", "away_")) and c.split("_", 1)[1] in _PREGAME_SIDE_BASES)):
            allowed.append(c)
        else:
            dropped.append(c)
    logger.info("get_feature_columns: allowlist kept %d, dropped %d columns "
                "(examples of dropped: %s)", len(allowed), len(dropped), ", ".join(dropped[:8]))
    return allowed


# ══════════════════════════════════════════════════════════════════════════════
#  BASE TABLES FROM THE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

_GAME_TYPES = {"REG": "REG", "WC": "WC", "DIV": "DIV", "CON": "CON", "CCG": "CON",
               "SB": "SB", "POST": "WC"}


def _prepare_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw schedule (shared by the game base and the history)."""
    s = sched.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["game_type"] = s["game_type"].astype(str).str.upper().map(_GAME_TYPES).fillna("REG")
    s["game_date"] = pd.to_datetime(s["gameday"], errors="coerce")
    for c in ("home_score", "away_score", "spread_line", "total_line"):
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
    return s


def _build_game_base(seasons: list) -> pd.DataFrame:
    from data_loader import get_schedules
    sched = _prepare_schedule(get_schedules())
    sched = sched[sched["season"].isin(seasons)].copy()
    if sched.empty:
        raise ValueError(f"No schedule data for seasons {seasons} — run load_all() first.")

    sched["day_of_week"] = sched["game_date"].dt.dayofweek.fillna(6).astype(int)

    # gametime is US Eastern on a 24h clock: 19:00 or later = night game
    # (TNF/SNF/MNF incl. the 19:15 Monday doubleheaders).
    hour = pd.to_numeric(sched["gametime"].astype(str).str.split(":").str[0], errors="coerce")
    sched["is_primetime"] = (hour >= 19).astype(int)

    # Roof per GAME (handles neutral sites and retractable roofs); the team
    # list is only a fallback for games whose roof is not published yet.
    roof = sched["roof"].astype(str).str.lower()
    known = roof.isin(["dome", "closed", "outdoors", "open"])
    sched["is_dome"] = np.where(known, roof.isin(["dome", "closed"]),
                                sched["home_team"].isin(_INDOOR_TEAMS)).astype(int)

    # Neutral-site regular-season games are the international series (the
    # stadium names are unreliable: 2025 lists the nominal home stadium).
    sched["is_international"] = ((sched["location"] == "Neutral")
                                 & (sched["game_type"] == "REG")).astype(int)
    sched["is_division_game"] = pd.to_numeric(sched["div_game"], errors="coerce").fillna(0).astype(int)

    keep = ["game_id", "season", "week", "game_type", "game_date",
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line", "day_of_week",
            "is_primetime", "is_international", "is_dome", "is_division_game"]
    return sched[keep].reset_index(drop=True)


def _schedule_team_games() -> pd.DataFrame:
    """Every scheduled game since 1999 in long format (one row per team per
    game) with franchise-normalised team codes. Used for features that need
    history from before the training window: Elo, QB/coach changes, HFA."""
    from data_loader import get_schedules
    s = _prepare_schedule(get_schedules())
    s["neutral"] = s.get("location", pd.Series("Home", index=s.index)).eq("Neutral")
    cols = ["game_id", "season", "week", "game_type", "game_date", "neutral"]
    home = s[cols].assign(team=_franchise(s["home_team"]), opponent=_franchise(s["away_team"]),
                          team_score=s["home_score"], opp_score=s["away_score"],
                          qb_id=s.get("home_qb_id"), coach=s.get("home_coach"), is_home=1)
    away = s[cols].assign(team=_franchise(s["away_team"]), opponent=_franchise(s["home_team"]),
                          team_score=s["away_score"], opp_score=s["home_score"],
                          qb_id=s.get("away_qb_id"), coach=s.get("away_coach"), is_home=0)
    out = pd.concat([home, away], ignore_index=True)
    return out.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHTED EPA
# ══════════════════════════════════════════════════════════════════════════════

def _compute_wepa_weights(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Weighted EPA (WEPA) for each play.

    Based on nfelo research (2022): weighting plays by their predictive value
    for future team performance improves predictive power by ~14% over raw EPA.

    Weight logic:
      - Garbage time (wp < 0.05 or wp > 0.95, AND < 15 min remaining): 0.20
        → opponent eases up, not reflective of true team quality
      - QB spike / kneel / two-point attempt: 0.10
        → nearly zero football signal
      - Penalty plays: 0.40
        → partially random, reduces true performance signal
      - Explosive plays (air_yards >= 20 OR yards_gained >= 15): 0.65
        → high variance, luck component (Pythagorean luck)
      - Normal pass (not above): 1.20
        → most predictive play type per nfelo study
      - Normal run: 0.80
        → less predictive than passing
    """
    pbp = pbp.copy()

    # Classify each play
    is_garbage = (
        ((pbp["wp"] < 0.05) | (pbp["wp"] > 0.95)) &
        (pbp["game_seconds_remaining"] < 900)   # last 15 mins
    )
    is_spike_kneel = pbp["play_type"].isin(["qb_spike", "qb_kneel"])
    is_two_point   = pbp.get("two_point_attempt", pd.Series(0, index=pbp.index)).astype(bool)
    is_penalty_only = (pbp["penalty"] == 1) & ~pbp["play_type"].isin(["pass","run"])
    is_explosive   = (
        (pbp["air_yards"] >= 20) | (pbp["yards_gained"] >= 15)
    ) & ~is_garbage

    # Build weight vector (vectorized)
    weight = pd.Series(1.0, index=pbp.index)   # default

    # Apply in priority order (most restrictive first)
    is_pass = pbp["play_type"] == "pass"
    is_run  = pbp["play_type"] == "run"

    weight = np.where(is_spike_kneel | is_two_point, 0.10, weight)
    weight = np.where(is_garbage,                    0.20, weight)
    weight = np.where(is_penalty_only,               0.40, weight)
    weight = np.where(is_explosive & ~is_garbage,    0.65, weight)
    weight = np.where(is_pass & ~is_explosive & ~is_garbage & ~is_penalty_only, 1.20, weight)
    weight = np.where(is_run  & ~is_explosive & ~is_garbage & ~is_penalty_only, 0.80, weight)

    pbp["_wepa_weight"] = weight
    pbp["_wepa"]        = pbp["epa"] * pbp["_wepa_weight"]

    return pbp



# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: PBP AGGREGATION (vectorized, no lambdas)
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_pbp(seasons: list) -> pd.DataFrame:
    """Team-game aggregates, built season by season.

    Closed seasons come from the committed frozen layer
    (data/frozen/pbp_agg_{s}.parquet). Those files are produced by
    scripts/freeze_data.py using the SAME _aggregate_pbp_core below, and the
    aggregation is strictly per-game / per-season (groupbys on game_id and
    season), so a per-season frozen result is identical to aggregating a
    multi-season concat. Only seasons without a frozen file (the season in
    progress) are aggregated from freshly fetched raw PBP.
    """
    from data_loader import get_pbp, load_frozen_pbp_agg

    frames = []
    for s in seasons:
        agg = load_frozen_pbp_agg(s)
        if agg is not None:
            frames.append(agg)
            continue
        try:
            pbp_s = get_pbp([s])
        except FileNotFoundError:
            logger.warning("  PBP %d: no raw data — skipped.", s)
            continue
        if len(pbp_s) == 0:
            continue
        frames.append(_aggregate_pbp_core(pbp_s))

    if not frames:
        logger.warning("No PBP data available — skipping PBP features.")
        return pd.DataFrame(columns=["game_id", "team"])
    return pd.concat(frames, ignore_index=True)


def _aggregate_pbp_core(pbp: pd.DataFrame) -> pd.DataFrame:
    """The actual aggregation math for ONE pbp DataFrame (any season span).
    Shared by the live path (_aggregate_pbp) and scripts/freeze_data.py —
    a single source of truth guarantees frozen == live results."""
    if len(pbp) == 0:
        return pd.DataFrame(columns=["game_id","team"])

    # Ensure all required columns exist with safe defaults
    numeric_cols_zero = [
        "epa","wpa","cpoe","air_yards","yards_gained","yards_after_catch",
        "success","sack","qb_hit","penalty_yards","first_down","touchdown",
        "interception","fumble_lost","fumble","shotgun","no_huddle",
        "return_yards",
        # WEPA columns
        "wp","game_seconds_remaining","penalty",
        # Coverage complexity columns
        "defenders_in_box","number_of_pass_rushers",
        # Turnover luck columns
        "pass_attempt","rush_attempt",
    ]
    for col in numeric_cols_zero:
        if col not in pbp.columns:
            pbp[col] = 0.0
        else:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)

    # Also ensure yardline_100 for red zone detection
    if "yardline_100" not in pbp.columns:
        pbp["yardline_100"] = 50.0
    else:
        pbp["yardline_100"] = pd.to_numeric(pbp["yardline_100"], errors="coerce").fillna(50)

    scrimmage_mask = pbp["play_type"].isin(["pass","run","qb_kneel","qb_spike"])
    pass_mask = pbp["play_type"] == "pass"
    run_mask  = pbp["play_type"] == "run"

    # ══ WEIGHTED EPA (WEPA) ══════════════════════════════════════════════════
    # nfelo research: weight plays by their predictive value for future performance
    # Garbage time, explosive plays, penalties, spikes downweighted
    # Normal pass plays most predictive (1.2×), runs less so (0.8×)
    pbp = _compute_wepa_weights(pbp)

    # ── Offensive aggregates — split pass/run first, then merge ──────────

    # All scrimmage plays offense
    off_base = (
        pbp[pbp["posteam"].notna() & scrimmage_mask]
        .groupby(["game_id","posteam"])
        .agg(
            off_epa_per_play       =("epa",         "mean"),
            off_epa_total          =("epa",         "sum"),
            # WEPA: weighted EPA (nfelo research — more predictive than raw EPA)
            off_wepa_per_play      =("_wepa",       "mean"),
            off_wepa_total         =("_wepa",       "sum"),
            off_success_rate       =("success",     "mean"),
            off_yards_per_play     =("yards_gained","mean"),
            off_total_yards        =("yards_gained","sum"),
            off_cpoe               =("cpoe",        "mean"),
            off_air_yards_per_att  =("air_yards",   "mean"),
            off_yac_per_rec        =("yards_after_catch","mean"),
            off_wpa                =("wpa",         "sum"),
            off_sack_rate          =("sack",        "mean"),
            off_penalty_yards      =("penalty_yards","sum"),
            n_plays_off            =("play_type",   "count"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # Passing only — precompute explosive flag to avoid lambda in agg
    pbp_pass = pbp[pbp["posteam"].notna() & pass_mask].copy()
    pbp_pass["_expl_pass"] = (pbp_pass["yards_gained"] >= 20).astype(float)
    off_pass = (
        pbp_pass
        .groupby(["game_id","posteam"])
        .agg(
            off_pass_epa      =("epa",         "mean"),
            off_pass_success  =("success",     "mean"),
            off_explosive_pass=("_expl_pass",  "mean"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # Rushing only
    off_rush = (
        pbp[pbp["posteam"].notna() & run_mask]
        .groupby(["game_id","posteam"])
        .agg(
            off_rush_epa           =("epa",         "mean"),
            off_rush_yards_per_att =("yards_gained","mean"),
            off_rush_success       =("success",     "mean"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # Explosive any — vectorized (lambda in agg unreliable in pandas 2.x)
    pbp_sc = pbp[pbp["posteam"].notna() & scrimmage_mask].copy()
    pbp_sc["_is_expl"] = (pbp_sc["yards_gained"] >= 20).astype(float)
    off_expl = (
        pbp_sc
        .groupby(["game_id","posteam"])
        .agg(off_explosive_play_rate=("_is_expl", "mean"))
        .reset_index()
        .rename(columns={"posteam":"team"})
    )
    pbp_sc = pbp_sc.drop(columns=["_is_expl"])

    # 3rd down offense
    td3_off = (
        pbp[pbp["down"] == 3]
        .groupby(["game_id","posteam"])
        .agg(off_third_down_rate=("first_down","mean"))
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # Red zone offense
    rz_off = (
        pbp[(pbp["yardline_100"] <= 20) & pbp["posteam"].notna()]
        .groupby(["game_id","posteam"])
        .agg(
            off_rz_success_rate=("success",   "mean"),
            off_rz_td_rate     =("touchdown", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # ── Defensive aggregates ─────────────────────────────────────────────
    # Precompute explosive flag for defense (lambda in agg unreliable pandas 2.x)
    pbp_def_expl = pbp[pbp["defteam"].notna() & scrimmage_mask].copy()
    pbp_def_expl["_is_expl_def"] = (pbp_def_expl["yards_gained"] >= 20).astype(float)

    def_base = (
        pbp[pbp["defteam"].notna() & scrimmage_mask]
        .groupby(["game_id","defteam"])
        .agg(
            def_epa_per_play   =("epa",         "mean"),
            def_success_rate   =("success",     "mean"),
            def_yards_per_play =("yards_gained","mean"),
            def_sack_rate      =("sack",        "mean"),
            def_pressure_rate  =("qb_hit",      "mean"),
            n_plays_def        =("play_type",   "count"),
        )
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    def_pass = (
        pbp[pbp["defteam"].notna() & pass_mask]
        .groupby(["game_id","defteam"])
        .agg(def_pass_epa=("epa","mean"))
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    def_rush = (
        pbp[pbp["defteam"].notna() & run_mask]
        .groupby(["game_id","defteam"])
        .agg(def_rush_epa=("epa","mean"))
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    def_expl = (
        pbp_def_expl
        .groupby(["game_id","defteam"])
        .agg(def_explosive_allowed=("_is_expl_def", "mean"))
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    td3_def = (
        pbp[pbp["down"] == 3]
        .groupby(["game_id","defteam"])
        .agg(def_third_down_allowed=("first_down","mean"))
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    rz_def = (
        pbp[(pbp["yardline_100"] <= 20) & pbp["defteam"].notna()]
        .groupby(["game_id","defteam"])
        .agg(
            def_rz_success_allowed=("success",   "mean"),
            def_rz_td_allowed     =("touchdown", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    # ── Turnovers ────────────────────────────────────────────────────────
    # nflverse PBP has no single 'turnover' column — derive from components

    # Ensure columns exist with safe fallbacks
    for col in ["interception", "fumble_lost", "fumble"]:
        if col not in pbp.columns:
            pbp[col] = 0
        else:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)

    # interception + fumble_lost = turnovers committed by offense
    pbp["_to_committed"] = pbp["interception"] + pbp["fumble_lost"]

    to_off = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id","posteam"])
        .agg(
            turnovers_committed=("_to_committed", "sum"),
            interceptions      =("interception",  "sum"),
            fumbles_lost       =("fumble_lost",   "sum"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # turnovers_forced = opponent's turnovers = what the defense forced
    to_def = (
        pbp[pbp["defteam"].notna()]
        .groupby(["game_id","defteam"])
        .agg(turnovers_forced=("_to_committed", "sum"))
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    # Clean up temp column
    pbp.drop(columns=["_to_committed"], inplace=True, errors="ignore")

    # ── Pace ─────────────────────────────────────────────────────────────

    pace = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id","posteam"])
        .agg(shotgun_rate=("shotgun","mean"), no_huddle_rate=("no_huddle","mean"))
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # ── Special teams ────────────────────────────────────────────────────
    # Precompute fg_made flag (lambda in agg unreliable pandas 2.x)
    pbp["_fg_made_flag"] = (pbp.get("field_goal_result", "") == "made").astype(float)

    fg = (
        pbp[pbp["play_type"] == "field_goal"]
        .groupby(["game_id","posteam"])
        .agg(
            fg_made    =("_fg_made_flag",    "sum"),
            fg_attempts=("field_goal_result", "count"),
        )
        .reset_index()
        .rename(columns={"posteam":"team"})
    )
    fg["fg_pct"] = fg["fg_made"] / fg["fg_attempts"].clip(lower=1)

    # ── Coverage Complexity (NFL Operations 2024 research) ───────────────
    # Two-high safety usage went from 44% (2019) to 63% (2024)
    # defenders_in_box and number_of_pass_rushers proxy coverage complexity

    cov_def = pd.DataFrame(columns=["game_id","team"])
    if "defenders_in_box" in pbp.columns and pbp["defenders_in_box"].notna().any():
        cov_def = (
            pbp[pbp["defteam"].notna() & pass_mask & (pbp["defenders_in_box"] > 0)]
            .groupby(["game_id","defteam"])
            .agg(
                def_avg_defenders_box    =("defenders_in_box",        "mean"),
                def_avg_pass_rushers     =("number_of_pass_rushers",  "mean"),
            )
            .reset_index()
            .rename(columns={"defteam":"team"})
        )
        # Coverage complexity score: combined signal (higher = more complex defense)
        if len(cov_def) > 0:
            cov_def["def_coverage_complexity"] = (
                cov_def["def_avg_defenders_box"].fillna(6.5) / 6.5 * 0.5 +
                cov_def["def_avg_pass_rushers"].fillna(4.0) / 4.0 * 0.5
            )
        logger.debug("  Coverage complexity: %d game-team rows", len(cov_def))

    # ── Merge all ────────────────────────────────────────────────────────

    result = off_base.copy()
    for df in [off_pass, off_rush, off_expl, td3_off, rz_off,
               def_base, def_pass, def_rush, def_expl, td3_def, rz_def,
               to_off, to_def, pace, fg, cov_def]:
        if len(df) > 0:
            result = result.merge(df, on=["game_id","team"], how="left")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TEAM-GAME TABLE + LAGGED ROLLING STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _build_team_games(games: pd.DataFrame, pbp_agg: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): the home and the away perspective stacked."""
    shared = ["game_id", "season", "week", "game_type", "game_date", "spread_line",
              "total_line", "is_dome", "is_primetime", "is_international",
              "is_division_game", "day_of_week"]
    home = games[shared + ["home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent",
                 "home_score": "team_score", "away_score": "opp_score"}).assign(is_home=1)
    away = games[shared + ["away_team", "home_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent",
                 "away_score": "team_score", "home_score": "opp_score"}).assign(is_home=0)
    tg = pd.concat([home, away], ignore_index=True)
    tg["franchise"] = _franchise(tg["team"])

    if len(pbp_agg) and "team" in pbp_agg.columns:
        tg = tg.merge(pbp_agg, on=["game_id", "team"], how="left")

    tg["wepa_available"]     = (tg["season"] >= WEPA_MIN_SEASON).astype(np.float32)
    tg["coverage_available"] = (tg["season"] >= COVERAGE_COMPLEXITY_MIN_SEASON).astype(np.float32)
    tg["ngs_available"]      = (tg["season"] >= NGS_MIN_SEASON).astype(np.float32)
    return tg.sort_values(["season", "week", "game_date", "game_id", "team"]).reset_index(drop=True)


def _lagged_rolling(df: pd.DataFrame, cols: list, windows=ROLLING_WINDOWS,
                    min_periods=lambda n: max(1, n // 2), group: str = "team") -> pd.DataFrame:
    """For every row: mean of the column over the previous N rows of the same
    team that HAVE a value (i.e. games already played). Unplayed games carry
    NaN and are skipped, so every upcoming game gets the window ending at the
    team's last played game. `df` must be sorted chronologically per team."""
    out = {}
    for col in cols:
        for _, sfx in windows:
            out[f"{col}_{sfx}"] = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(group, sort=False).groups.items():
        sub = df.loc[idx]
        for col in cols:
            v = sub[col]
            have = v.notna()
            for n, sfx in windows:
                r = v[have].rolling(n, min_periods=min_periods(n)).mean()
                out[f"{col}_{sfx}"].loc[idx] = r.reindex(sub.index).shift(1).ffill().values
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


_ROLLED_STATS = [
    "off_epa_per_play", "off_success_rate", "off_yards_per_play", "off_pass_epa", "off_cpoe",
    "off_air_yards_per_att", "off_rush_epa", "off_rush_yards_per_att", "off_third_down_rate",
    "off_rz_success_rate", "off_rz_td_rate", "off_explosive_play_rate", "off_sack_rate",
    "off_wpa", "off_wepa_per_play", "off_wepa_total",
    "def_epa_per_play", "def_success_rate", "def_yards_per_play", "def_pass_epa",
    "def_rush_epa", "def_explosive_allowed", "def_third_down_allowed",
    "def_rz_success_allowed", "def_rz_td_allowed", "def_sack_rate", "def_pressure_rate",
    "def_avg_defenders_box", "def_avg_pass_rushers", "def_coverage_complexity",
    "turnovers_committed", "turnovers_forced", "interceptions", "fumbles_lost",
    "fg_pct", "shotgun_rate", "no_huddle_rate", "team_score", "opp_score",
]


def _add_rolling(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["franchise", "game_date", "game_id"]).reset_index(drop=True)
    played = tg["team_score"].notna() & tg["opp_score"].notna()
    tg["turnover_diff"] = np.where(played, tg.get("turnovers_forced") - tg.get("turnovers_committed"), np.nan) \
        if "turnovers_forced" in tg.columns else np.nan
    tg["score_diff"] = np.where(played, tg["team_score"] - tg["opp_score"], np.nan)
    tg["win"] = np.where(played, (tg["team_score"] > tg["opp_score"]).astype(float), np.nan)

    tg = _lagged_rolling(tg, [c for c in _ROLLED_STATS if c in tg.columns], group="franchise")
    tg = _lagged_rolling(tg, ["turnover_diff", "score_diff", "win"], min_periods=lambda n: 1,
                         group="franchise")
    tg = tg.rename(columns={f"win_{s}": f"win_rate_{s}" for _, s in ROLLING_WINDOWS})
    return tg.drop(columns=["turnover_diff", "score_diff", "win"])


def _merge_weekly_and_roll(tg: pd.DataFrame, weekly: pd.DataFrame, cols: list,
                           windows, min_periods) -> pd.DataFrame:
    """Attach per-(team, season, week) values to every team-game (NaN for games
    not played yet) and roll them along each franchise's game timeline."""
    weekly = (weekly.assign(franchise=_franchise(weekly["team"]))
              .groupby(["franchise", "season", "week"], as_index=False)[cols].mean())
    base = tg[["franchise", "season", "week", "game_date", "game_id"]].merge(
        weekly, on=["franchise", "season", "week"], how="left")
    base = base.sort_values(["franchise", "game_date", "game_id"]).reset_index(drop=True)
    rolled = _lagged_rolling(base, cols, windows=windows, min_periods=min_periods,
                             group="franchise")
    new_cols = [c for c in rolled.columns if c not in base.columns]
    return tg.merge(rolled[["franchise", "game_id"] + new_cols],
                    on=["franchise", "game_id"], how="left")


def _add_qb_form(tg: pd.DataFrame) -> pd.DataFrame:
    """Rolling passing form of each week's starting QB (most attempts)."""
    from data_loader import get_table
    ps = get_table("player_stats_weekly")
    if ps is None or not {"team", "season", "week", "position", "attempts", "passing_epa"} <= set(ps.columns):
        logger.warning("player_stats_weekly unavailable — QB form features skipped")
        return tg
    ps = ps[ps["position"] == "QB"].copy()
    for c in ("season", "week", "attempts", "passing_epa", "pacr"):
        if c in ps.columns:
            ps[c] = pd.to_numeric(ps[c], errors="coerce")
    ps = (ps.sort_values("attempts", ascending=False)
            .groupby(["team", "season", "week"], as_index=False).first())
    ps["qb_epa_per_att"] = ps["passing_epa"] / ps["attempts"].clip(lower=1)
    cols = ["qb_epa_per_att"] + (["qb_pacr"] if "pacr" in ps.columns else [])
    ps = ps.rename(columns={"pacr": "qb_pacr"})
    return _merge_weekly_and_roll(tg, ps, cols, ROLLING_WINDOWS, lambda n: 1)


def _add_ngs_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Next Gen Stats team form, 4-game lagged rolling mean. Week-0 rows are
    season summaries and are dropped (they would leak the whole season)."""
    from data_loader import get_table
    configs = [
        ("ngs_passing",   "ngs_pass", ["avg_time_to_throw", "avg_completed_air_yards", "aggressiveness"]),
        ("ngs_rushing",   "ngs_rush", ["efficiency", "rush_yards_over_expected_per_att", "avg_time_to_los"]),
        ("ngs_receiving", "ngs_recv", ["avg_separation", "avg_intended_air_yards", "catch_percentage"]),
    ]
    for table, prefix, cols in configs:
        t = get_table(table)
        if t is None or "team_abbr" not in t.columns or "week" not in t.columns:
            continue
        cols = [c for c in cols if c in t.columns]
        t = t.rename(columns={"team_abbr": "team"})
        t["week"] = pd.to_numeric(t["week"], errors="coerce")
        t = t[t["week"] > 0].copy()
        for c in cols:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        agg = t.groupby(["team", "season", "week"], as_index=False)[cols].mean()
        agg = agg.rename(columns={c: f"{prefix}_{c}" for c in cols})
        tg = _merge_weekly_and_roll(tg, agg, [f"{prefix}_{c}" for c in cols],
                                    ((4, "r4"),), lambda n: 1)
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  ELO (full history since 1999, two-phase update)
# ══════════════════════════════════════════════════════════════════════════════

def _add_elo(tg: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo for every game since 1999 (no cold start inside the
    training window). Both teams' pre-game ratings are recorded before
    either is updated, so a game never sees its own result."""
    games = (hist[hist["is_home"] == 1]
             .sort_values(["season", "week", "game_date", "game_id"]))
    elo, pre, exp = {}, {}, {}
    prev_season = None
    for g in games.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            for t in elo:
                elo[t] = settings.ELO_START + settings.ELO_REGRESSION * (elo[t] - settings.ELO_START)
        prev_season = g.season
        h, a = g.team, g.opponent
        h_elo, a_elo = elo.get(h, settings.ELO_START), elo.get(a, settings.ELO_START)
        e_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400.0))
        pre[(g.game_id, h)], pre[(g.game_id, a)] = h_elo, a_elo
        exp[(g.game_id, h)], exp[(g.game_id, a)] = e_home, 1.0 - e_home
        if pd.notna(g.team_score) and pd.notna(g.opp_score):
            mov = np.log(abs(g.team_score - g.opp_score) + 1) * settings.ELO_MOV_MULT / (
                abs(h_elo - a_elo) * 0.001 + 1.0)
            actual = 1.0 if g.team_score > g.opp_score else (0.5 if g.team_score == g.opp_score else 0.0)
            delta = settings.ELO_K * mov * (actual - e_home)
            elo[h], elo[a] = h_elo + delta, a_elo - delta

    keys = list(zip(tg["game_id"], tg["franchise"], strict=True))
    tg["elo_pre_game"] = [pre.get(k, np.nan) for k in keys]
    tg["elo_expected_win_prob"] = [exp.get(k, np.nan) for k in keys]
    # spread_line is home-oriented and POSITIVE when the home team is
    # favoured, so the home side's implied edge is +spread/2.
    half = tg["spread_line"] / 2.0
    tg["vegas_implied_power"] = np.where(tg["is_home"] == 1, half, -half)
    tg["elo_vegas_divergence"] = tg["elo_expected_win_prob"] - (0.5 + tg["vegas_implied_power"] / 28.0)
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  QB / COACH CHANGES AND ROSTER CONTINUITY
# ══════════════════════════════════════════════════════════════════════════════

def _add_qb_and_coach_changes(tg: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Starting-QB and head-coach changes from the schedule, which names the
    starting QB of every played game and of next week's games, and the head
    coach of every game.
      qb_changed     — this game's starter is not the team's main starter of
                       the previous season
      qb_start_share — share of the team's previous 8 games started by this
                       game's QB (how much the team's rolling QB/offense form
                       actually reflects him)
      coach_changed  — this game's head coach did not coach the team's last
                       game of the previous season
    Starters not announced yet (games 2+ weeks out) are assumed to be the
    team's most recent starter."""
    h = hist.copy()
    h["qb_id"] = h.groupby("team")["qb_id"].ffill()
    played = h["team_score"].notna()

    primary = (h[played & (h["game_type"] == "REG")]
               .groupby(["team", "season"])["qb_id"]
               .agg(lambda x: x.value_counts().index[0] if x.notna().any() else np.nan))
    prev_primary = primary.rename("prev_primary").reset_index()
    prev_primary["season"] += 1
    h = h.merge(prev_primary, on=["team", "season"], how="left")
    h["qb_changed"] = (h["qb_id"].notna() & h["prev_primary"].notna()
                       & (h["qb_id"] != h["prev_primary"])).astype(int)

    share = np.full(len(h), np.nan)
    for _, idx in h.groupby("team", sort=False).groups.items():
        qbs = h.loc[idx, "qb_id"].to_numpy()
        was_played = played.loc[idx].to_numpy()
        prior: list = []
        for j, i in enumerate(idx):
            if prior and pd.notna(qbs[j]):
                share[h.index.get_loc(i)] = np.mean([q == qbs[j] for q in prior[-8:]])
            if was_played[j]:
                prior.append(qbs[j])
    h["qb_start_share"] = share

    last_coach = (h[played].groupby(["team", "season"])["coach"].last()
                  .rename("prev_coach").reset_index())
    last_coach["season"] += 1
    h = h.merge(last_coach, on=["team", "season"], how="left")
    h["coach_changed"] = (h["coach"].notna() & h["prev_coach"].notna()
                          & (h["coach"] != h["prev_coach"])).astype(int)

    feats = h[["game_id", "team", "qb_changed", "qb_start_share", "coach_changed"]].rename(
        columns={"team": "franchise"})
    tg = tg.merge(feats, on=["game_id", "franchise"], how="left")
    tg["qb_changed"] = tg["qb_changed"].fillna(0)
    tg["coach_changed"] = tg["coach_changed"].fillna(0)
    tg["team_instability"] = 0.6 * tg["qb_changed"] + 0.4 * tg["coach_changed"]
    logger.info("  QB changes: %d team-games, coach changes: %d team-games",
                int(tg["qb_changed"].sum()), int(tg["coach_changed"].sum()))
    return tg


def _add_roster_continuity(tg: pd.DataFrame) -> pd.DataFrame:
    """Share of last season's final active roster that is on this season's
    week-1 active roster (both known before the season's games)."""
    from data_loader import get_table
    rw = get_table("rosters_weekly")
    tg["roster_continuity"] = np.nan
    if rw is None or not {"team", "season", "week", "gsis_id", "status"} <= set(rw.columns):
        return tg
    rw = rw[(rw["status"] == "ACT") & rw["gsis_id"].notna()].copy()
    rw["team"] = _franchise(rw["team"])
    rw["season"] = pd.to_numeric(rw["season"], errors="coerce")
    rw["week"] = pd.to_numeric(rw["week"], errors="coerce")
    if "game_type" in rw.columns:
        rw = rw[rw["game_type"].fillna("REG") == "REG"]
    first = rw[rw["week"] == rw.groupby(["team", "season"])["week"].transform("min")]
    last = rw[rw["week"] == rw.groupby(["team", "season"])["week"].transform("max")]
    first_ids = first.groupby(["team", "season"])["gsis_id"].agg(set)
    last_ids = last.groupby(["team", "season"])["gsis_id"].agg(set)
    cont = {}
    for (team, season), cur in first_ids.items():
        prev = last_ids.get((team, season - 1))
        if prev:
            cont[(team, season)] = len(cur & prev) / len(prev)
    tg["roster_continuity"] = [cont.get(k, np.nan) for k in zip(tg["franchise"], tg["season"], strict=True)]
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT: REST, CALENDAR, SCORING ERA, HOME-FIELD ADVANTAGE
# ══════════════════════════════════════════════════════════════════════════════

def _hfa_by_season(hist: pd.DataFrame) -> pd.Series:
    """League home-field advantage per completed season: average home minus
    average away points, neutral-site games excluded."""
    done = hist[hist["team_score"].notna() & ~hist["neutral"]]
    home = done[done["is_home"] == 1].groupby("season")["team_score"].mean()
    away = done[done["is_home"] == 0].groupby("season")["team_score"].mean()
    return (home - away).dropna()


def _add_contextual(tg: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team", "season", "game_date"]).copy()
    prev_date = tg.groupby(["team", "season"])["game_date"].shift(1)
    tg["rest_days"]     = (tg["game_date"] - prev_date).dt.days.fillna(10).clip(1, 21)
    tg["is_bye_week"]   = (tg["rest_days"] > 10).astype(int)
    tg["is_short_week"] = (tg["rest_days"] < 6).astype(int)

    tg["season_half"] = (tg["week"] > 9).astype(int)
    from data_loader import regular_season_weeks
    tg["week_norm"] = tg["week"] / tg["season"].map(regular_season_weeks).astype(float)

    dow = tg["day_of_week"]
    tg["is_sunday"], tg["is_thursday"] = (dow == 6).astype(int), (dow == 3).astype(int)
    tg["is_monday"], tg["is_saturday"] = (dow == 0).astype(int), (dow == 5).astype(int)

    # Scoring environment: previous season's average game total.
    home_rows = hist[(hist["is_home"] == 1) & hist["team_score"].notna()]
    season_total = (home_rows["team_score"] + home_rows["opp_score"]).groupby(home_rows["season"]).mean()
    tg["season_avg_total"] = tg["season"].map({s + 1: v for s, v in season_total.items()})
    tg["scoring_era_adj"] = tg["season_avg_total"] - 45.5

    # League HFA: exponentially weighted mean of earlier seasons (2020 excluded).
    hfa = _hfa_by_season(hist).drop(index=settings.COVID_SEASON, errors="ignore")

    def _rolling_hfa(season, decay=0.7):
        past = hfa[hfa.index < season]
        if past.empty:
            return 2.0
        w = decay ** np.arange(len(past) - 1, -1, -1)
        return float(np.sum(w * past.values) / np.sum(w))

    tg["rolling_hfa"] = tg["season"].map({s: _rolling_hfa(s) for s in tg["season"].unique()})

    # Team-specific HFA: mean over the franchise's earlier seasons (2020 out).
    done = hist[hist["team_score"].notna() & ~hist["neutral"]
                & (hist["season"] != settings.COVID_SEASON)]
    per_ts = done.groupby(["team", "season", "is_home"])["team_score"].mean().unstack("is_home")
    team_hfa = (per_ts[1] - per_ts[0]).dropna().rename("hfa").reset_index().sort_values(["team", "season"])
    g = team_hfa.groupby("team")
    team_hfa["prior"] = (g["hfa"].cumsum() - team_hfa["hfa"]) / g.cumcount().replace(0, np.nan)
    lookup = team_hfa.set_index(["team", "season"])["prior"]
    idx = pd.MultiIndex.from_arrays([tg["franchise"], tg["season"]])
    tg["team_specific_hfa"] = lookup.reindex(idx).values
    tg["team_specific_hfa"] = tg["team_specific_hfa"].fillna(tg["rolling_hfa"])
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  PIVOT: ONE ROW PER GAME
# ══════════════════════════════════════════════════════════════════════════════

_SHARED_GAME_COLS = [
    "season", "week", "game_type", "game_date", "spread_line", "total_line",
    "is_dome", "is_primetime", "is_international", "is_division_game",
    "is_sunday", "is_thursday", "is_monday", "is_saturday",
    "season_avg_total", "scoring_era_adj", "rolling_hfa", "week_norm", "season_half",
    "wepa_available", "coverage_available", "ngs_available",
]


def _pivot_to_game(tg: pd.DataFrame) -> pd.DataFrame:
    """One row per game: per-team columns get home_/away_ prefixes."""
    skip = set(_SHARED_GAME_COLS) | {"game_id", "is_home", "team", "opponent",
                                     "franchise", "day_of_week"}
    team_cols = [c for c in tg.columns if c not in skip]
    home = tg[tg["is_home"] == 1]
    away = tg[tg["is_home"] == 0]
    game_df = (home[["game_id"] + team_cols].add_prefix("home_")
               .rename(columns={"home_game_id": "game_id"})
               .merge(away[["game_id"] + team_cols].add_prefix("away_")
                      .rename(columns={"away_game_id": "game_id"}), on="game_id"))
    game_df = game_df.merge(
        home[["game_id", "team", "franchise"] + _SHARED_GAME_COLS].rename(
            columns={"team": "home_team", "franchise": "home_franchise"}), on="game_id")
    game_df = game_df.merge(
        away[["game_id", "team", "franchise"]].rename(
            columns={"team": "away_team", "franchise": "away_franchise"}), on="game_id")

    game_df["rest_diff"] = game_df["home_rest_days"] - game_df["away_rest_days"]
    game_df["elo_gap"] = game_df["home_elo_pre_game"] - game_df["away_elo_pre_game"]
    # Elo-implied home margin (25 Elo ≈ 1 point) minus the Vegas home margin.
    game_df["elo_vs_spread"] = game_df["elo_gap"] / 25.0 - game_df["spread_line"]
    _add_gap(game_df, "home_off_cpoe_r8", "away_off_cpoe_r8", "cpoe_gap_r8")
    _add_gap(game_df, "home_def_coverage_complexity_r8", "away_def_coverage_complexity_r8",
             "coverage_complexity_gap")
    for sfx in ("r4", "r8", "r16"):
        _add_gap(game_df, f"home_turnover_diff_{sfx}", f"away_turnover_diff_{sfx}",
                 f"turnover_diff_gap_{sfx}")

    # Offense-vs-defense matchup gaps; offensive EPA is the more stable signal
    # (nfelo), so it is weighted sqrt(w) and defense 1/sqrt(w).
    off_scale = float(np.sqrt(settings.OFF_EPA_WEIGHT))
    for side, opp in (("home", "away"), ("away", "home")):
        for off_col, def_col, name in (
            ("off_epa_per_play_r8", "def_epa_per_play_r8", f"{side}_off_vs_{opp}_def_epa_r8"),
            ("off_pass_epa_r8", "def_pass_epa_r8", f"{side}_pass_vs_{opp}_def_r8"),
            ("off_rush_epa_r8", "def_rush_epa_r8", f"{side}_rush_vs_{opp}_def_r8"),
            ("off_wepa_per_play_r8", "def_epa_per_play_r8", f"{side}_wepa_vs_{opp}_def_r8"),
        ):
            a, b = f"{side}_{off_col}", f"{opp}_{def_col}"
            if a in game_df.columns and b in game_df.columns:
                game_df[name] = game_df[a].fillna(0) * off_scale - game_df[b].fillna(0) / off_scale

    game_df["target_home_score"] = pd.to_numeric(game_df["home_team_score"], errors="coerce")
    game_df["target_away_score"] = pd.to_numeric(game_df["away_team_score"], errors="coerce")
    game_df["target_total"]  = game_df["target_home_score"] + game_df["target_away_score"]
    game_df["target_spread"] = game_df["target_home_score"] - game_df["target_away_score"]
    return game_df.sort_values(["season", "week", "game_date", "game_id"]).reset_index(drop=True)


def _add_gap(df: pd.DataFrame, col_a: str, col_b: str, name: str) -> None:
    if col_a in df.columns and col_b in df.columns:
        df[name] = df[col_a] - df[col_b]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gf = build_all_features()
    print(f"Games: {len(gf)} | Features: {len(get_feature_columns(gf))}")

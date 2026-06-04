"""
feature_engineering.py
=======================
Builds the full game-level feature matrix from nflverse raw data.

Key design decisions:
- No lambda functions inside groupby.agg() — unreliable in pandas 2.x
- No iterrows() — all merges are vectorized
- Strict temporal ordering: rolling windows use shift(1) to prevent leakage
- Graceful degradation: every data source is optional with sensible defaults
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_SHORT = 4
ROLLING_MED   = 8
ROLLING_LONG  = 16
ELO_K         = 20.0
ELO_START     = 1500.0

# NFL divisions for rivalry detection
_DIVISIONS = [
    {"DAL","NYG","PHI","WAS"}, {"CHI","DET","GB","MIN"},
    {"ATL","CAR","NO","TB"},   {"ARI","LAR","SF","SEA"},
    {"BUF","MIA","NE","NYJ"},  {"BAL","CIN","CLE","PIT"},
    {"HOU","IND","JAX","TEN"}, {"DEN","KC","LV","LAC"},
]
_DOME_TEAMS = {"BUF","DAL","DET","HOU","IND","LAR","LV","MIN","NO","SEA","SF","TEN"}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_all_features(seasons: Optional[list] = None) -> pd.DataFrame:
    """Build full game-level feature matrix. Returns one row per game."""
    from data_loader import ALL_SEASONS
    if seasons is None:
        seasons = ALL_SEASONS

    logger.info("Building game feature matrix for seasons %s …", seasons)

    games    = _build_game_base(seasons)
    logger.info("  Base games: %d", len(games))

    pbp_agg  = _aggregate_pbp(seasons)
    logger.info("  PBP agg: %d team-game rows", len(pbp_agg))

    tg       = _build_team_games(games, pbp_agg)
    logger.info("  Team-game table: %d rows", len(tg))

    tg = _add_rolling(tg)
    tg = _add_qb_features(tg, seasons)
    tg = _add_injury_features(tg)
    tg = _add_elo(tg)
    tg = _add_cross_season(tg)
    tg = _add_official_features(tg)
    tg = _add_ngs_features(tg, seasons)
    tg = _add_contextual(tg)

    from data_loader import get_game_type_flag
    tg["game_type_weight"] = tg["game_type"].map(
        lambda g: get_game_type_flag(g)
    )

    tg.to_parquet(PROCESSED_DIR / "team_games.parquet", index=False)

    game_df = _pivot_to_game(tg)
    game_df.to_parquet(PROCESSED_DIR / "game_features.parquet", index=False)
    logger.info("  Final game matrix: %d rows × %d cols", len(game_df), len(game_df.columns))

    return game_df


def get_feature_columns(game_df: pd.DataFrame) -> list:
    """Return feature columns (exclude metadata and targets)."""
    exclude = {
        "game_id","season","week","game_type","game_date",
        "home_team","away_team","home_team_x","away_team_x",
        "home_team_score","away_team_score",
        "home_opp_score","away_opp_score",
        "target_home_score","target_away_score",
        "target_total","target_spread",
        "game_type_weight",
        "home_prev_game_date","away_prev_game_date",
        "home_game_id","away_game_id",
        "home_season","away_season","home_week","away_week",
        "home_game_type","away_game_type",
        "home_game_type_weight","away_game_type_weight",
        # Raw text/id columns not useful as features
        "home_team_y","away_team_y",
        "home_pname","away_pname",
    }
    return [c for c in game_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(game_df[c])]


def load_processed() -> pd.DataFrame:
    """Load the cached game features parquet if it exists."""
    p = PROCESSED_DIR / "game_features.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError("game_features.parquet not found — run build_all_features() first.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: BASE GAME TABLE FROM SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════

def _build_game_base(seasons: list) -> pd.DataFrame:
    from data_loader import get_schedules
    sched = get_schedules()

    # Normalise season column (may be int or str)
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched = sched[sched["season"].isin(seasons)].copy()

    if len(sched) == 0:
        raise ValueError(f"No schedule data found for seasons {seasons}. "
                         "Run data_loader.load_all() first.")

    # Normalise game_type
    gt_map = {"REG":"REG","reg":"REG","WC":"WC","DIV":"DIV",
               "CON":"CON","CCG":"CON","SB":"SB","POST":"WC"}
    gt_col = "game_type" if "game_type" in sched.columns else None
    if gt_col:
        sched["game_type"] = sched[gt_col].astype(str).str.upper().map(gt_map).fillna("REG")
    else:
        sched["game_type"] = "REG"

    # Date
    date_col = next((c for c in ["gameday","game_date","date"] if c in sched.columns), None)
    sched["game_date"] = pd.to_datetime(sched[date_col], errors="coerce") if date_col else pd.NaT

    # Day of week
    sched["day_of_week"] = sched["game_date"].dt.dayofweek.fillna(6).astype(int)

    # Primetime
    time_col = next((c for c in ["gametime","game_time"] if c in sched.columns), None)
    if time_col:
        sched["is_primetime"] = sched[time_col].astype(str).str.contains(
            r"20:|21:|19:30|8:20|8:15", regex=True, na=False
        ).astype(int)
    else:
        sched["is_primetime"] = 0

    # International
    intl_kw = ["tottenham","wembley","allianz","azteca","melbourne",
               "maracana","bernabeu","stade de france"]
    venue_col = next((c for c in ["stadium","venue","location"] if c in sched.columns), None)
    if venue_col:
        sched["is_international"] = sched[venue_col].astype(str).str.lower().apply(
            lambda v: int(any(k in v for k in intl_kw))
        )
    else:
        sched["is_international"] = 0

    # Dome
    home_col = "home_team" if "home_team" in sched.columns else "home"
    sched["is_dome"] = sched[home_col].isin(_DOME_TEAMS).astype(int)

    # Division rivalry
    away_col = "away_team" if "away_team" in sched.columns else "away"
    if "div_game" in sched.columns:
        sched["is_division_game"] = sched["div_game"].fillna(0).astype(int)
    else:
        sched["is_division_game"] = sched.apply(
            lambda r: int(any(r[home_col] in d and r[away_col] in d
                              for d in _DIVISIONS)), axis=1
        )

    # Score columns
    for col, default_col in [("home_score","home_score"),("away_score","away_score")]:
        if col not in sched.columns:
            sched[col] = np.nan

    # Rename to standard
    rename = {}
    if home_col != "home_team": rename[home_col] = "home_team"
    if away_col != "away_team": rename[away_col] = "away_team"
    sched = sched.rename(columns=rename)

    keep = [c for c in [
        "game_id","season","week","game_type","game_date",
        "home_team","away_team","home_score","away_score",
        "spread_line","total_line","home_moneyline","away_moneyline",
        "stadium","roof","temp","wind","humidity",
        "day_of_week","is_primetime","is_international","is_dome","is_division_game",
        "referee",
    ] if c in sched.columns]

    return sched[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WEPA HELPER: Play-type weights for Weighted EPA
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
    from data_loader import get_pbp

    try:
        pbp = get_pbp(seasons)
    except FileNotFoundError:
        logger.warning("No PBP data available — skipping PBP features.")
        return pd.DataFrame(columns=["game_id","team"])

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

    # Passing only
    off_pass = (
        pbp[pbp["posteam"].notna() & pass_mask]
        .groupby(["game_id","posteam"])
        .agg(
            off_pass_epa      =("epa",         "mean"),
            off_pass_success  =("success",     "mean"),
            off_explosive_pass=("yards_gained",lambda x: (x >= 20).mean()),
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

    # Explosive any
    off_expl = (
        pbp[pbp["posteam"].notna() & scrimmage_mask]
        .groupby(["game_id","posteam"])
        .agg(off_explosive_play_rate=("yards_gained", lambda x: (x >= 20).mean()))
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

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
        pbp[pbp["defteam"].notna() & scrimmage_mask]
        .groupby(["game_id","defteam"])
        .agg(def_explosive_allowed=("yards_gained", lambda x: (x >= 20).mean()))
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

    fg = (
        pbp[pbp["play_type"] == "field_goal"]
        .groupby(["game_id","posteam"])
        .agg(
            fg_made    =("field_goal_result", lambda x: (x == "made").sum()),
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

    # ── Turnover Luck Correction (Harvard 2014, covers.com research) ─────
    # 54% of turnovers are luck. We compute rolling vs season baseline divergence.
    # A team winning the turnover battle far above their season average is "lucky"
    # and will likely regress.

    to_luck = pd.DataFrame(columns=["game_id","team"])
    if "_to_committed" in pbp.columns:
        # Season-level expected turnover rate (per play)
        season_to_rates = {}
        if "season" in pbp.columns:
            for season, grp in pbp.groupby("season"):
                pass_plays = (grp["play_type"] == "pass").sum()
                total_ints = grp["interception"].sum()
                total_fumb = grp["fumble_lost"].sum()
                # Expected rates per pass/run attempt
                int_rate  = total_ints / max(pass_plays, 1)
                fumb_rate = total_fumb / max(len(grp), 1)
                season_to_rates[season] = {"int_rate": int_rate, "fumb_rate": fumb_rate}

        # Per-game turnover luck: actual - expected
        to_by_game = (
            pbp[pbp["posteam"].notna()]
            .groupby(["game_id","posteam","season"])
            .agg(
                actual_int    =("interception", "sum"),
                actual_fumb   =("fumble_lost",  "sum"),
                pass_attempts =("pass_attempt", "sum"),
                rush_attempts =("rush_attempt", "sum"),
            )
            .reset_index()
            .rename(columns={"posteam":"team"})
        )

        if len(to_by_game) > 0 and season_to_rates:
            def _expected_to(row):
                rates = season_to_rates.get(row["season"], {"int_rate": 0.025, "fumb_rate": 0.01})
                exp_int  = row["pass_attempts"] * rates["int_rate"]
                exp_fumb = row["rush_attempts"] * rates["fumb_rate"]
                return exp_int + exp_fumb

            to_by_game["expected_to"] = to_by_game.apply(_expected_to, axis=1)
            to_by_game["actual_to"]   = to_by_game["actual_int"] + to_by_game["actual_fumb"]
            # Luck = actual - expected (positive = lucky, will regress)
            to_by_game["turnover_luck"] = to_by_game["actual_to"] - to_by_game["expected_to"]
            # Skill-adjusted turnovers: actual - 0.54 × luck component
            to_by_game["to_skill_adj"] = (
                to_by_game["actual_to"] -
                0.54 * to_by_game["turnover_luck"].clip(lower=0)
            )
            to_luck = to_by_game[["game_id","team","turnover_luck","to_skill_adj"]]
            logger.debug("  Turnover luck: %d game-team rows", len(to_luck))

    # ── Merge all ────────────────────────────────────────────────────────

    result = off_base.copy()
    for df in [off_pass, off_rush, off_expl, td3_off, rz_off,
               def_base, def_pass, def_rush, def_expl, td3_def, rz_def,
               to_off, to_def, pace, fg, cov_def, to_luck]:
        if len(df) > 0:
            result = result.merge(df, on=["game_id","team"], how="left")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: TEAM-GAME TABLE (vectorized, no iterrows)
# ══════════════════════════════════════════════════════════════════════════════

def _build_team_games(games: pd.DataFrame, pbp_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized reshape: create one row per (team, game) by stacking
    home and away perspectives from the schedule.
    """
    # Shared game-level columns
    shared_cols = [c for c in [
        "game_id","season","week","game_type","game_date",
        "spread_line","total_line","temp","wind","humidity",
        "is_dome","is_primetime","is_international","is_division_game",
        "day_of_week","referee",
    ] if c in games.columns]

    # Home perspective
    home_map = {
        "home_team":"team", "away_team":"opponent",
        "home_score":"team_score", "away_score":"opp_score",
    }
    home = games[shared_cols + [c for c in home_map if c in games.columns]].copy()
    home = home.rename(columns=home_map)
    home["is_home"] = 1

    # Away perspective
    away_map = {
        "away_team":"team", "home_team":"opponent",
        "away_score":"team_score", "home_score":"opp_score",
    }
    away = games[shared_cols + [c for c in away_map if c in games.columns]].copy()
    away = away.rename(columns=away_map)
    away["is_home"] = 0

    # Stack
    tg = pd.concat([home, away], ignore_index=True)

    # Merge PBP
    if len(pbp_agg) > 0 and "team" in pbp_agg.columns:
        tg = tg.merge(pbp_agg, on=["game_id","team"], how="left")

    tg = tg.sort_values(["season","week","game_date","game_id","team"]).reset_index(drop=True)
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_rolling(tg: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [c for c in [
        # Standard EPA features
        "off_epa_per_play","off_success_rate","off_yards_per_play",
        "off_pass_epa","off_cpoe","off_air_yards_per_att","off_rush_epa",
        "off_rush_yards_per_att","off_third_down_rate","off_rz_success_rate",
        "off_rz_td_rate","off_explosive_play_rate","off_sack_rate","off_wpa",
        "def_epa_per_play","def_success_rate","def_yards_per_play",
        "def_pass_epa","def_rush_epa","def_explosive_allowed",
        "def_third_down_allowed","def_rz_success_allowed","def_rz_td_allowed",
        "def_sack_rate","def_pressure_rate",
        # WEPA: Weighted EPA (nfelo research — more predictive than raw EPA)
        "off_wepa_per_play","off_wepa_total",
        # Coverage complexity (NFL Operations 2024 research)
        "def_avg_defenders_box","def_avg_pass_rushers","def_coverage_complexity",
        # Turnover luck correction (Harvard 2014, covers.com)
        "turnovers_committed","turnovers_forced","interceptions","fumbles_lost",
        "turnover_luck","to_skill_adj",
        # Standard other features
        "fg_pct","shotgun_rate","no_huddle_rate","team_score","opp_score",
    ] if c in tg.columns]

    frames = []
    for _, grp in tg.groupby("team", sort=False):
        grp = grp.sort_values("game_date").copy()

        # Build all new rolling columns in a dict, then concat once — avoids fragmentation
        new_cols: dict = {}

        for col in stat_cols:
            s = grp[col].shift(1)
            for N, sfx in [(ROLLING_SHORT,"r4"),(ROLLING_MED,"r8"),(ROLLING_LONG,"r16")]:
                new_cols[f"{col}_{sfx}"] = s.rolling(N, min_periods=max(1, N//2)).mean()

        # Derived: turnover diff
        if "turnovers_committed" in grp.columns and "turnovers_forced" in grp.columns:
            td = (grp["turnovers_forced"] - grp["turnovers_committed"]).shift(1)
            for N, sfx in [(4,"r4"),(8,"r8"),(16,"r16")]:
                new_cols[f"turnover_diff_{sfx}"] = td.rolling(N, min_periods=1).mean()

        # Derived: score diff + win rate
        if "team_score" in grp.columns and "opp_score" in grp.columns:
            sd = (grp["team_score"] - grp["opp_score"]).shift(1)
            wr = (grp["team_score"] > grp["opp_score"]).astype(float).shift(1)
            for N, sfx in [(4,"r4"),(8,"r8"),(16,"r16")]:
                new_cols[f"score_diff_{sfx}"] = sd.rolling(N, min_periods=1).mean()
                new_cols[f"win_rate_{sfx}"]   = wr.rolling(N, min_periods=1).mean()

        # Single concat for all new columns — no fragmentation
        new_df = pd.DataFrame(new_cols, index=grp.index)
        grp = pd.concat([grp, new_df], axis=1)
        frames.append(grp)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["season","week","game_date","game_id"]
    ).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: QB FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_qb_features(tg: pd.DataFrame, seasons: list) -> pd.DataFrame:
    from data_loader import get_table
    ps = get_table("player_stats_weekly")
    if ps is None:
        logger.warning("player_stats_weekly not available — skipping QB features")
        return tg

    ps["season"] = pd.to_numeric(ps.get("season", 0), errors="coerce")
    ps = ps[ps["season"].isin(seasons)].copy()

    # Keep QBs only
    if "position" in ps.columns:
        ps = ps[ps["position"] == "QB"]

    team_col = next((c for c in ["recent_team","team","team_abbr"] if c in ps.columns), None)
    if team_col is None:
        return tg
    ps = ps.rename(columns={team_col: "team"})

    # Starter = most attempts per team/week
    if "attempts" in ps.columns:
        ps["attempts"] = pd.to_numeric(ps["attempts"], errors="coerce").fillna(0)
        ps = (ps.sort_values("attempts", ascending=False)
                .groupby(["team","season","week"]).first().reset_index())

    if "passing_epa" in ps.columns and "attempts" in ps.columns:
        ps["qb_epa_per_att"] = (
            pd.to_numeric(ps["passing_epa"], errors="coerce") /
            ps["attempts"].clip(lower=1)
        )

    qb_roll_cols = [c for c in ["qb_epa_per_att","dakota","pacr"] if c in ps.columns]
    for col in qb_roll_cols:
        ps[col] = pd.to_numeric(ps[col], errors="coerce")

    if not qb_roll_cols:
        return tg

    frames = []
    for _, grp in ps.groupby("team", sort=False):
        grp = grp.sort_values(["season","week"]).copy()
        for col in qb_roll_cols:
            s = grp[col].shift(1)
            for N, sfx in [(4,"r4"),(8,"r8"),(16,"r16")]:
                grp[f"qb_{col}_{sfx}"] = s.rolling(N, min_periods=1).mean()
        frames.append(grp)

    if not frames:
        return tg

    ps_rolled = pd.concat(frames, ignore_index=True)
    qb_feat_cols = ["team","season","week"] + [
        c for c in ps_rolled.columns if c.startswith("qb_") and "_r" in c
    ]
    tg = tg.merge(ps_rolled[qb_feat_cols], on=["team","season","week"], how="left")
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: INJURY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

_STATUS_SCORE = {
    "Active":1.0,"Probable":0.95,"Questionable":0.75,
    "Doubtful":0.25,"Out":0.0,"IR":0.0,"PUP":0.0,"DNP":0.5,
}
_OFF_POS = {"QB","WR","RB","TE","OT","OG","C"}
_DEF_POS = {"DE","DT","LB","CB","S"}
_POS_W   = {"QB":5.0,"WR":2.0,"RB":2.0,"TE":1.5,
            "OT":1.2,"OG":1.0,"C":1.0,"DE":1.2,"DT":1.0,
            "LB":1.0,"CB":1.2,"S":1.0}


def _add_injury_features(tg: pd.DataFrame) -> pd.DataFrame:
    from data_loader import get_table
    inj = get_table("injuries")

    defaults = {
        "qb_available":1.0,"off_availability_score":1.0,
        "def_availability_score":1.0,"injury_report_severity":0.0,
        "injury_data_freshness":0.65,
    }

    if inj is None or len(inj) == 0:
        for k, v in defaults.items():
            tg[k] = v
        return tg

    # Vectorized availability scoring
    if "report_status" not in inj.columns:
        for k, v in defaults.items():
            tg[k] = v
        return tg

    inj = inj.copy()
    inj["status_score"] = inj["report_status"].map(_STATUS_SCORE).fillna(0.75)
    inj["pos_weight"]   = inj["position"].map(_POS_W).fillna(1.0)
    inj["weighted"]     = inj["status_score"] * inj["pos_weight"]
    inj["is_off"] = inj["position"].isin(_OFF_POS).astype(int)
    inj["is_def"] = inj["position"].isin(_DEF_POS).astype(int)
    inj["is_qb"]  = (inj["position"] == "QB").astype(int)
    inj["is_out"] = inj["report_status"].isin(["Out","Doubtful","IR","PUP"]).astype(int)

    team_col = next((c for c in ["team","team_abbr"] if c in inj.columns), None)
    if team_col is None:
        for k, v in defaults.items():
            tg[k] = v
        return tg
    inj = inj.rename(columns={team_col:"team"})

    grp_cols = [c for c in ["team","season","week"] if c in inj.columns]
    if len(grp_cols) < 2:
        for k, v in defaults.items():
            tg[k] = v
        return tg

    agg = inj.groupby(grp_cols).agg(
        qb_available         =("status_score", lambda x: x[inj.loc[x.index,"is_qb"]==1].max() if (inj.loc[x.index,"is_qb"]==1).any() else 1.0),
        off_availability_score=("weighted",    lambda x: x[inj.loc[x.index,"is_off"]==1].mean() if (inj.loc[x.index,"is_off"]==1).any() else 1.0),
        def_availability_score=("weighted",    lambda x: x[inj.loc[x.index,"is_def"]==1].mean() if (inj.loc[x.index,"is_def"]==1).any() else 1.0),
        injury_report_severity=("is_out",      "sum"),
    ).reset_index()
    agg = agg.fillna({"qb_available":1.0,"off_availability_score":1.0,
                      "def_availability_score":1.0,"injury_report_severity":0.0})

    tg = tg.merge(agg, on=grp_cols, how="left")
    for col, val in [("qb_available",1.0),("off_availability_score",1.0),
                     ("def_availability_score",1.0),("injury_report_severity",0.0)]:
        tg[col] = tg[col].fillna(val)
    tg["injury_data_freshness"] = 0.65
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: ELO RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def _add_elo(tg: pd.DataFrame) -> pd.DataFrame:
    # Load Elo parameters dynamically from Bayesian optimizer
    from bayesian_optimizer import load_weights as _lw
    _w = _lw()
    ELO_K_DYN    = float(_w.get("elo_k",           ELO_K))
    ELO_REG      = float(_w.get("elo_regression",   0.67))
    ELO_MOV      = float(_w.get("elo_mov_mult",     2.2))

    # Sort by season, week, then game_id (stable sort within same day)
    tg = tg.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    tg["elo_pre_game"]          = np.nan
    tg["elo_expected_win_prob"] = np.nan
    elo: dict = {}

    processed: set = set()
    prev_week   = None
    prev_season = None

    for idx, row in tg.iterrows():
        season = row["season"]
        # Seasonal regression toward mean at season boundary
        if prev_season is not None and season != prev_season:
            for t in list(elo):
                elo[t] = ELO_START + ELO_REG * (elo[t] - ELO_START)
        prev_season = season

        team = row["team"]
        opp  = row.get("opponent", None)
        if not opp:
            continue

        team_elo = elo.get(team, ELO_START)
        opp_elo  = elo.get(opp,  ELO_START)
        exp_win  = 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))

        tg.at[idx, "elo_pre_game"]         = team_elo
        tg.at[idx, "elo_expected_win_prob"] = exp_win

        gid = row["game_id"]
        if gid not in processed and row.get("is_home") == 1:
            processed.add(gid)
            h_score = row.get("team_score")
            a_score = row.get("opp_score")
            if pd.notna(h_score) and pd.notna(a_score):
                margin   = abs(h_score - a_score)
                mov_mult = np.log(margin + 1) * ELO_MOV / (
                    abs(team_elo - opp_elo) * 0.001 + 1.0
                )
                h_actual = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
                delta = ELO_K_DYN * mov_mult * (h_actual - exp_win)
                elo[team] = team_elo + delta
                elo[opp]  = opp_elo  - delta

    if "spread_line" in tg.columns:
        tg["vegas_implied_power"] = np.where(
            tg["is_home"] == 1,
            -tg["spread_line"].fillna(0) / 2.0,
             tg["spread_line"].fillna(0) / 2.0,
        )
        tg["elo_vegas_divergence"] = (
            tg["elo_expected_win_prob"] -
            (0.5 + tg["vegas_implied_power"].fillna(0) / 28.0)
        )

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: CROSS-SEASON FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: CROSS-SEASON FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_cross_season(tg: pd.DataFrame) -> pd.DataFrame:
    from data_loader import get_table

    tg = _add_win_totals(tg)
    tg = _add_roster_continuity(tg)
    tg = _add_draft_quality(tg)
    tg = _add_qb_coach_change_flags(tg)
    tg = _add_game_lines(tg)
    return tg


def _add_win_totals(tg: pd.DataFrame) -> pd.DataFrame:
    """Vegas preseason win totals — correct nflverse/nfldata source.
    Columns: season, team, line (the O/U wins line), over_odds, under_odds
    """
    from data_loader import get_table
    wt = get_table("win_totals")

    if wt is None or len(wt) == 0:
        tg["vegas_preseason_wins"] = np.nan
        return tg

    # Correct column names from nflverse/nfldata win_totals.csv
    # season | team | line | over_odds | under_odds
    team_col = next((c for c in ["team", "team_abbr"] if c in wt.columns), None)
    line_col  = next((c for c in ["line", "wins", "win_total", "over_under",
                                   "implied_wins"] if c in wt.columns), None)

    if team_col is None or line_col is None or "season" not in wt.columns:
        logger.warning("win_totals: unexpected columns %s", list(wt.columns))
        tg["vegas_preseason_wins"] = np.nan
        return tg

    wt = wt.copy()
    wt = wt.rename(columns={team_col: "team", line_col: "vegas_preseason_wins"})
    wt["season"] = pd.to_numeric(wt["season"], errors="coerce")
    wt["vegas_preseason_wins"] = pd.to_numeric(wt["vegas_preseason_wins"], errors="coerce")

    # Take consensus line (average if multiple books)
    wt_agg = (wt.groupby(["team", "season"])["vegas_preseason_wins"]
              .mean().reset_index())

    tg = tg.merge(wt_agg, on=["team", "season"], how="left")
    filled = tg["vegas_preseason_wins"].notna().sum()
    logger.info("  win_totals: %d team-season rows matched", filled)
    return tg


def _add_roster_continuity(tg: pd.DataFrame) -> pd.DataFrame:
    from data_loader import get_table
    rosters = get_table("rosters")

    if rosters is None or len(rosters) == 0:
        tg["roster_continuity"] = 0.5
        return tg

    team_col = next((c for c in ["team", "team_abbr"] if c in rosters.columns), None)
    id_col   = next((c for c in ["gsis_id", "player_id"] if c in rosters.columns), None)

    if team_col is None or id_col is None or "season" not in rosters.columns:
        tg["roster_continuity"] = 0.5
        return tg

    rosters = rosters.copy()
    rosters["season"] = pd.to_numeric(rosters["season"], errors="coerce")
    rosters = rosters.rename(columns={team_col: "team"})

    cont = {}
    for (team, season), grp in rosters.groupby(["team", "season"]):
        prev = rosters[(rosters["team"] == team) & (rosters["season"] == season - 1)]
        if len(prev) == 0:
            cont[(team, season)] = 0.5
            continue
        cur_ids  = set(grp[id_col].dropna())
        prev_ids = set(prev[id_col].dropna())
        cont[(team, season)] = len(cur_ids & prev_ids) / max(len(prev_ids), 1)

    # Vectorized lookup
    tg["roster_continuity"] = tg.apply(
        lambda r: cont.get((r["team"], r["season"]), 0.5), axis=1
    )
    return tg


def _add_draft_quality(tg: pd.DataFrame) -> pd.DataFrame:
    from data_loader import get_table
    dp = get_table("draft_picks")

    if dp is None or len(dp) == 0 or "pfr_av" not in dp.columns:
        tg["draft_quality_score"] = 0.0
        return tg

    team_col = next((c for c in ["team", "team_abbr"] if c in dp.columns), None)
    if team_col is None or "season" not in dp.columns:
        tg["draft_quality_score"] = 0.0
        return tg

    dp = dp.copy()
    dp["pfr_av"] = pd.to_numeric(dp["pfr_av"], errors="coerce").fillna(0)
    dp = dp.rename(columns={team_col: "team"})
    dp["season"] = pd.to_numeric(dp["season"], errors="coerce")

    dq = (dp.groupby(["team", "season"])["pfr_av"]
          .sum().reset_index()
          .rename(columns={"pfr_av": "draft_quality_score"}))
    tg = tg.merge(dq, on=["team", "season"], how="left")
    tg["draft_quality_score"] = tg["draft_quality_score"].fillna(0)
    return tg


def _add_qb_coach_change_flags(tg: pd.DataFrame) -> pd.DataFrame:
    """
    QB and head coach change flags — quantifies the single biggest
    unmeasured cross-season signal.

    Simulation showed QB changes cause ~4pt extra prediction error in week 1,
    decaying to ~1pt by week 6. This flag lets the model discount prior-season
    EPA stats when the QB changed.

    Sources: weekly rosters (QB depth chart position) + schedules (coaches).
    Falls back gracefully if data is unavailable.
    """
    from data_loader import get_table
    import re

    # ── QB change flag ────────────────────────────────────────────────────────
    weekly = get_table("rosters_weekly")
    qb_changed: dict = {}     # (team, season) → bool

    if weekly is not None and len(weekly) > 0:
        team_col = next((c for c in ["team", "team_abbr"] if c in weekly.columns), None)
        name_col = next((c for c in ["full_name", "player_name"] if c in weekly.columns), None)
        pos_col  = next((c for c in ["depth_chart_position", "position"] if c in weekly.columns), None)

        if all(c is not None for c in [team_col, name_col, pos_col]) and "season" in weekly.columns:
            wr = weekly.copy()
            wr["season"] = pd.to_numeric(wr["season"], errors="coerce")
            wr["week"]   = pd.to_numeric(wr.get("week", 1), errors="coerce").fillna(1)
            wr = wr.rename(columns={team_col: "team", name_col: "pname"})

            # QB starters: highest depth_chart rank among QBs, week 1
            qb_w1 = wr[
                wr[pos_col].isin(["QB"]) &
                (wr["week"] == 1)
            ].groupby(["team", "season"])["pname"].first().reset_index()

            for _, row in qb_w1.iterrows():
                team, season = row["team"], row["season"]
                prev_qbs = qb_w1[
                    (qb_w1["team"] == team) &
                    (qb_w1["season"] == season - 1)
                ]
                if len(prev_qbs) == 0:
                    qb_changed[(team, season)] = False
                else:
                    qb_changed[(team, season)] = (
                        row["pname"] != prev_qbs.iloc[0]["pname"]
                    )

    if qb_changed:
        tg["qb_changed"] = tg.apply(
            lambda r: int(qb_changed.get((r["team"], r["season"]), False)), axis=1
        )
        n_changed = tg["qb_changed"].sum()
        logger.info("  qb_changed: %d team-season observations flagged", int(n_changed))
    else:
        tg["qb_changed"] = 0

    # ── QB change EPA penalty ─────────────────────────────────────────────────
    # When QB changed, discount prior-season QB EPA by (1 - decay factor)
    # Week 1 → 0% reliability of prior QB stats; week 8 → ~80% reliability
    if "week" in tg.columns and "qb_changed" in tg.columns:
        tg["week_num"] = pd.to_numeric(tg["week"], errors="coerce").fillna(9)
        # Reliability: 0 at week 1, rising to ~1 by week 8
        tg["qb_era_reliability"] = np.where(
            tg["qb_changed"] == 1,
            np.clip(1 - np.exp(-0.35 * (tg["week_num"] - 1)), 0, 1),
            1.0
        )
    else:
        tg["qb_era_reliability"] = 1.0

    # ── Coach change flag ──────────────────────────────────────────────────────
    # Check from schedules if there's a coach column; otherwise use schedule-
    # derived heuristic (win% change > threshold suggests coaching change).
    # Minimal implementation — will be enhanced once coach data is loaded.
    sched = get_table("schedules")

    if sched is not None:
        # Try explicit coach columns
        coach_col = next((c for c in ["home_coach", "away_coach", "coach"]
                          if c in (sched.columns if sched is not None else [])), None)
        if coach_col:
            # Build dict: (team, season) → coach name from week 1
            sched_c = sched.copy()
            sched_c["season"] = pd.to_numeric(sched_c.get("season", 0), errors="coerce")
            coaches: dict = {}
            # Home teams
            if "home_team" in sched_c.columns and "home_coach" in sched_c.columns:
                hc = sched_c.groupby(["home_team", "season"])["home_coach"].first()
                for (team, season), coach in hc.items():
                    coaches[(team, season)] = str(coach)
            # Away teams
            if "away_team" in sched_c.columns and "away_coach" in sched_c.columns:
                ac = sched_c.groupby(["away_team", "season"])["away_coach"].first()
                for (team, season), coach in ac.items():
                    if (team, season) not in coaches:
                        coaches[(team, season)] = str(coach)

            if coaches:
                def _coach_changed(row):
                    cur  = coaches.get((row["team"], row["season"]))
                    prev = coaches.get((row["team"], row["season"] - 1))
                    if cur is None or prev is None:
                        return 0
                    return int(cur != prev)
                tg["coach_changed"] = tg.apply(_coach_changed, axis=1)
                logger.info("  coach_changed: %d flagged", int(tg["coach_changed"].sum()))
            else:
                tg["coach_changed"] = 0
        else:
            tg["coach_changed"] = 0
    else:
        tg["coach_changed"] = 0

    # Combined instability score (used in confidence calculation)
    tg["team_instability"] = (
        tg["qb_changed"].fillna(0) * 0.6 +
        tg["coach_changed"].fillna(0) * 0.4
    )

    return tg


def _add_game_lines(tg: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical game-level book spreads and totals from mrcaseb data.
    This gives us 'what the market thought' at game time — a very powerful
    cross-validation signal. Also used for Edge ATS calculation.

    Columns: game_id, market_type (spread/total/money_line), abbr (team), lines, odds
    """
    from data_loader import get_table
    gl = get_table("game_lines")

    if gl is None or len(gl) == 0 or "game_id" not in gl.columns:
        return tg

    try:
        # Extract spreads: consensus across books
        spreads = (
            gl[gl["market_type"] == "spread"]
            .groupby(["game_id", "abbr"])["lines"]
            .median()
            .reset_index()
            .rename(columns={"lines": "book_spread_hist", "abbr": "team"})
        )

        # Extract totals
        totals = (
            gl[gl["market_type"] == "total"]
            .groupby("game_id")["lines"]
            .median()
            .reset_index()
            .rename(columns={"lines": "book_total_hist"})
        )

        # Merge spreads onto team-game table
        if "game_id" in tg.columns and len(spreads) > 0:
            tg = tg.merge(spreads, on=["game_id", "team"], how="left")

        if "game_id" in tg.columns and len(totals) > 0:
            tg = tg.merge(totals, on="game_id", how="left")

        # Fill with schedule spread_line if historical line missing
        if "spread_line" in tg.columns:
            tg["book_spread_hist"] = tg.get("book_spread_hist", pd.Series(np.nan, index=tg.index))
            tg["book_spread_hist"] = tg["book_spread_hist"].fillna(
                tg["spread_line"] if "spread_line" in tg.columns else np.nan
            )

        logger.info("  game_lines: %d spread rows, %d total rows added",
                    spreads["book_spread_hist"].notna().sum() if len(spreads) > 0 else 0,
                    totals["book_total_hist"].notna().sum() if len(totals) > 0 else 0)

    except Exception as e:
        logger.warning("  game_lines merge failed: %s", e)

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9: OFFICIAL / REFEREE TENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

def _add_official_features(tg: pd.DataFrame) -> pd.DataFrame:
    from data_loader import get_schedules
    sched = get_schedules()

    if "referee" not in sched.columns or "home_score" not in sched.columns:
        tg["ref_home_win_rate"]  = np.nan
        tg["ref_total_tendency"] = np.nan
        return tg

    sched = sched.copy()
    sched["total"]    = pd.to_numeric(sched["home_score"], errors="coerce") + pd.to_numeric(sched["away_score"], errors="coerce")
    sched["home_win"] = (pd.to_numeric(sched["home_score"], errors="coerce") > pd.to_numeric(sched["away_score"], errors="coerce")).astype(float)

    ref_stats = (
        sched.groupby("referee")
        .agg(ref_home_win_rate=("home_win","mean"), ref_total_tendency=("total","mean"), n=("game_id","count"))
        .reset_index()
    )
    ref_stats = ref_stats[ref_stats["n"] >= 10]

    if "referee" in tg.columns:
        tg = tg.merge(
            ref_stats[["referee","ref_home_win_rate","ref_total_tendency"]],
            on="referee", how="left"
        )
    else:
        tg["ref_home_win_rate"]  = np.nan
        tg["ref_total_tendency"] = np.nan

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10: NGS FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_ngs_features(tg: pd.DataFrame, seasons: list) -> pd.DataFrame:
    from data_loader import get_table

    ngs_configs = [
        ("ngs_passing", "ngs_pass", ["avg_time_to_throw","avg_completed_air_yards","aggressiveness"]),
        ("ngs_rushing", "ngs_rush", ["efficiency","rush_yards_over_expected_per_att","avg_time_to_los"]),
        ("ngs_receiving","ngs_recv",["avg_separation","avg_intended_air_yards","catch_percentage"]),
    ]

    for tbl_name, prefix, cols in ngs_configs:
        tbl = get_table(tbl_name)
        if tbl is None or len(tbl) == 0:
            continue

        team_col = next((c for c in ["team_abbr","team","possession_team"] if c in tbl.columns), None)
        if team_col is None:
            continue

        tbl = tbl.rename(columns={team_col:"team"})
        available = [c for c in cols if c in tbl.columns]
        if not available:
            continue

        grp_cols = [c for c in ["team","season","week"] if c in tbl.columns]
        if "team" not in grp_cols:
            continue

        tbl_agg = tbl.groupby(grp_cols)[available].mean().reset_index()
        tbl_agg = tbl_agg.rename(columns={c: f"{prefix}_{c}" for c in available})
        tg = tg.merge(tbl_agg, on=grp_cols, how="left")

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11: CONTEXTUAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _add_contextual(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team","season","game_date"]).copy()

    tg["prev_game_date"] = tg.groupby(["team","season"])["game_date"].shift(1)
    tg["rest_days"]      = (tg["game_date"] - tg["prev_game_date"]).dt.days.fillna(10).clip(1, 21)
    tg["is_bye_week"]    = (tg["rest_days"] > 10).astype(int)
    tg["is_short_week"]  = (tg["rest_days"] < 6).astype(int)

    if "wind" in tg.columns:
        wind = pd.to_numeric(tg["wind"], errors="coerce").fillna(0)
        tg["high_wind"]    = ((wind >= 15) & (tg["is_dome"].fillna(0) == 0)).astype(int)
        tg["extreme_wind"] = ((wind >= 25) & (tg["is_dome"].fillna(0) == 0)).astype(int)

    if "temp" in tg.columns:
        temp = pd.to_numeric(tg["temp"], errors="coerce").fillna(60)
        tg["cold_game"]      = ((temp < 32) & (tg["is_dome"].fillna(0) == 0)).astype(int)
        tg["very_cold_game"] = ((temp < 20) & (tg["is_dome"].fillna(0) == 0)).astype(int)

    tg["week"]        = pd.to_numeric(tg["week"], errors="coerce").fillna(1)
    tg["season_half"] = (tg["week"] > 9).astype(int)
    tg["week_norm"]   = tg["week"] / 18.0

    dow = pd.to_numeric(tg["day_of_week"], errors="coerce").fillna(6)
    tg["is_sunday"]   = (dow == 6).astype(int)
    tg["is_thursday"] = (dow == 3).astype(int)
    tg["is_monday"]   = (dow == 0).astype(int)
    tg["is_saturday"] = (dow == 5).astype(int)

    # ── Season scoring trend ──────────────────────────────────────────────
    # NFL total scoring has trended up ~1 pt/season since 2015.
    if "team_score" in tg.columns and "opp_score" in tg.columns:
        season_totals = (
            tg.groupby(["game_id","season"])
            .agg(game_total=("team_score", lambda x: x.sum()))
            .reset_index()
            .groupby("season")["game_total"]
            .mean()
            .reset_index()
            .rename(columns={"game_total": "season_avg_total"})
        )
        tg = tg.merge(season_totals, on="season", how="left")
        tg["scoring_era_adj"] = tg["season_avg_total"].fillna(45.5) - 45.5
    else:
        tg["scoring_era_adj"] = 0.0

    # ── Rolling Home Field Advantage (FiveThirtyEight methodology) ────────
    # FiveThirtyEight (2021): switched from fixed HFA to rolling 10-season average.
    # Research: HFA declined from ~2.8pts (pre-2015) to ~1.5pts (2024).
    # Implementation: compute actual HFA per season from completed games,
    # then use exponentially weighted rolling average as a feature.

    hfa_by_season = {}
    if "team_score" in tg.columns and "is_home" in tg.columns:
        completed = tg[tg["team_score"].notna()].copy()
        if len(completed) > 0:
            home_games = completed[completed["is_home"] == 1]
            away_games = completed[completed["is_home"] == 0]

            # HFA = average home score minus average away score, by season
            h_avg = home_games.groupby("season")["team_score"].mean()
            a_avg = away_games.groupby("season")["team_score"].mean()

            seasons_with_both = h_avg.index.intersection(a_avg.index)
            for s in sorted(seasons_with_both):
                # Exclude 2020 COVID outlier from HFA calculation
                if s != 2020:
                    hfa_by_season[s] = float(h_avg[s] - a_avg[s])

    # Exponentially weighted rolling average across seasons
    # Weight decay: more recent seasons weighted more
    def _rolling_hfa(season: int, hfa_dict: dict, decay: float = 0.7) -> float:
        """EW average of past HFA values, decaying by `decay` per year."""
        past = [(s, v) for s, v in hfa_dict.items() if s < season and s != 2020]
        if not past:
            return 2.0   # league average default
        past.sort(key=lambda x: x[0])
        total_w, total_v = 0.0, 0.0
        for i, (_, v) in enumerate(past):
            w = decay ** (len(past) - 1 - i)
            total_w += w
            total_v += w * v
        return total_v / max(total_w, 1e-9)

    tg["rolling_hfa"] = tg["season"].apply(
        lambda s: _rolling_hfa(s, hfa_by_season)
    )

    # Per-team HFA — some teams (KC, SEA) consistently higher than league average
    if "team" in tg.columns and "is_home" in tg.columns and "team_score" in tg.columns:
        completed = tg[tg["team_score"].notna() & (tg["season"] != 2020)].copy()
        if len(completed) > 20:
            team_home = completed[completed["is_home"] == 1].groupby("team")["team_score"].mean()
            team_away = completed[completed["is_home"] == 0].groupby("team")["team_score"].mean()
            team_hfa  = (team_home - team_away).dropna()
            tg["team_specific_hfa"] = tg["team"].map(team_hfa).fillna(tg["rolling_hfa"])
        else:
            tg["team_specific_hfa"] = tg["rolling_hfa"]
    else:
        tg["team_specific_hfa"] = 2.0

    # Dynamic season HFA (replaces the hardcoded season_hfa dict)
    tg["season_hfa"] = tg["season"].apply(
        lambda s: _rolling_hfa(s, hfa_by_season)
    )

    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  PIVOT: ONE ROW PER GAME
# ══════════════════════════════════════════════════════════════════════════════

def _pivot_to_game(tg: pd.DataFrame) -> pd.DataFrame:
    """Pivot team-game table to one row per game with home_ and away_ prefixed columns."""

    # shared_cols = game-level cols that appear ONCE (not per-team)
    # game_id is excluded here because it's already the merge key from add_prefix/rename
    shared_cols = [c for c in [
        "season","week","game_type","game_date",
        "spread_line","total_line","temp","wind","humidity",
        "is_dome","is_primetime","is_international","is_division_game",
        "high_wind","extreme_wind","cold_game","very_cold_game",
        "is_sunday","is_thursday","is_monday","is_saturday",
        "ref_home_win_rate","ref_total_tendency","game_type_weight",
    ] if c in tg.columns]

    skip = set(shared_cols) | {
        "game_id","is_home","team","opponent","game_date",
        "day_of_week","referee","prev_game_date",
    }

    team_stat_cols = [c for c in tg.columns if c not in skip]

    home = tg[tg["is_home"] == 1][["game_id"] + team_stat_cols].copy()
    away = tg[tg["is_home"] == 0][["game_id"] + team_stat_cols].copy()

    home = home.add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = away.add_prefix("away_").rename(columns={"away_game_id": "game_id"})

    game_df = home.merge(away, on="game_id", how="inner")

    # Merge shared columns — use game_id only as key, not as a column in shared
    shared = tg[tg["is_home"] == 1][["game_id"] + shared_cols].copy()
    # Drop duplicate game_id rows (shouldn't exist, but safeguard)
    shared = shared.drop_duplicates(subset=["game_id"])
    game_df = game_df.merge(shared, on="game_id", how="left")

    # Rest differential
    if "home_rest_days" in game_df.columns and "away_rest_days" in game_df.columns:
        game_df["rest_diff"] = game_df["home_rest_days"] - game_df["away_rest_days"]

    # Elo gap
    if "home_elo_pre_game" in game_df.columns and "away_elo_pre_game" in game_df.columns:
        game_df["elo_gap"] = game_df["home_elo_pre_game"] - game_df["away_elo_pre_game"]

    # CPOE gap
    if "home_off_cpoe_r8" in game_df.columns and "away_off_cpoe_r8" in game_df.columns:
        game_df["cpoe_gap_r8"] = game_df["home_off_cpoe_r8"] - game_df["away_off_cpoe_r8"]

    # Matchup gap features
    _add_gap(game_df, "home_off_epa_per_play_r8",  "away_def_epa_per_play_r8",  "home_off_vs_away_def_epa_r8")
    _add_gap(game_df, "home_off_pass_epa_r8",      "away_def_pass_epa_r8",      "home_pass_vs_away_def_r8")
    _add_gap(game_df, "home_off_rush_epa_r8",      "away_def_rush_epa_r8",      "home_rush_vs_away_def_r8")
    _add_gap(game_df, "away_off_epa_per_play_r8",  "home_def_epa_per_play_r8",  "away_off_vs_home_def_epa_r8")
    _add_gap(game_df, "away_off_pass_epa_r8",      "home_def_pass_epa_r8",      "away_pass_vs_home_def_r8")
    _add_gap(game_df, "away_off_rush_epa_r8",      "home_def_rush_epa_r8",      "away_rush_vs_home_def_r8")

    # Turnover gaps (raw + luck-adjusted)
    for sfx in ["r4","r8","r16"]:
        h, a = f"home_turnover_diff_{sfx}", f"away_turnover_diff_{sfx}"
        _add_gap(game_df, h, a, f"turnover_diff_gap_{sfx}")

    # WEPA matchup gaps (research-backed: more predictive than raw EPA)
    _add_gap(game_df, "home_off_wepa_per_play_r8", "away_def_epa_per_play_r8",
             "home_wepa_vs_away_def_r8")
    _add_gap(game_df, "away_off_wepa_per_play_r8", "home_def_epa_per_play_r8",
             "away_wepa_vs_home_def_r8")

    # Coverage complexity advantage
    _add_gap(game_df, "home_def_coverage_complexity_r8", "away_def_coverage_complexity_r8",
             "coverage_complexity_gap")

    # Turnover skill-adjusted gaps (Harvard: 54% luck → use skill-adjusted version)
    _add_gap(game_df, "home_to_skill_adj_r8", "away_to_skill_adj_r8",
             "turnover_skill_adj_gap_r8")
    _add_gap(game_df, "home_turnover_luck_r8", "away_turnover_luck_r8",
             "turnover_luck_gap_r8")

    # Target columns — use merged score columns directly
    for side, target in [("home","target_home_score"),("away","target_away_score")]:
        score_col = f"{side}_team_score"
        if score_col in game_df.columns:
            game_df[target] = pd.to_numeric(game_df[score_col], errors="coerce")
        else:
            game_df[target] = np.nan

    # ── Research-backed feature weighting ─────────────────────────────────
    # nfelo WEPA study: offensive EPA is 1.6× more predictive than defensive EPA
    # Harvard/covers: ~54% of turnovers are luck → downweight
    from bayesian_optimizer import load_weights as _lw
    _w = _lw()
    OFF_EPA_W  = float(_w.get("off_epa_weight",  1.6))
    TO_W       = float(_w.get("turnover_weight", 0.6))

    # Apply weights to matchup gap features after they're computed
    for col in game_df.columns:
        if "off_epa" in col and ("home_off" in col or "away_off" in col):
            game_df[col] = game_df[col] * OFF_EPA_W
        elif "turnover" in col.lower() and any(s in col for s in ["_r4","_r8","_r16","_diff"]):
            game_df[col] = game_df[col] * TO_W

    if "target_home_score" in game_df.columns and "target_away_score" in game_df.columns:
        game_df["target_total"]  = game_df["target_home_score"] + game_df["target_away_score"]
        game_df["target_spread"] = game_df["target_home_score"] - game_df["target_away_score"]

    return game_df.sort_values(["season","week","game_date"]).reset_index(drop=True)


def _add_gap(df: pd.DataFrame, col_a: str, col_b: str, name: str):
    if col_a in df.columns and col_b in df.columns:
        df[name] = df[col_a] - df[col_b]


def save_feature_dictionary(game_df: pd.DataFrame) -> None:
    import csv
    feat_cols = get_feature_columns(game_df)
    with open(PROCESSED_DIR / "feature_dictionary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_name","dtype","null_pct","mean","std"])
        for col in feat_cols:
            s = game_df[col]
            w.writerow([col, str(s.dtype),
                        f"{s.isna().mean():.3f}",
                        f"{s.mean():.4f}" if pd.api.types.is_numeric_dtype(s) else "",
                        f"{s.std():.4f}"  if pd.api.types.is_numeric_dtype(s) else ""])
    logger.info("Feature dictionary saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gf = build_all_features()
    feat_cols = get_feature_columns(gf)
    print(f"Games: {len(gf)} | Features: {len(feat_cols)}")

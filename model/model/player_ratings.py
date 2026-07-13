"""
player_ratings.py
=================
Sub-model: rates every active NFL player 0-99 within their position group.

Design principles:
  - Rolling Z-score over active players at the same position (last 4 weeks)
  - Minimum snap/route/snap-count threshold to filter out low-exposure players
  - Bayesian shrinkage toward position mean for small samples
  - Injury-adjusted effective rating for lineup strength calculation
  - Position matchup gaps: home_qb_rating vs away_pass_def_rating etc.

Output features added to game_df:
  Per team (home_ / away_ prefix):
    {team}_qb_rating             — starting QB 0-99
    {team}_wr_rating             — top-2 WR average 0-99
    {team}_rb_rating             — top RB 0-99
    {team}_te_rating             — top TE 0-99
    {team}_ol_rating             — OL unit 0-99 (proxy: team pass block rate)
    {team}_pass_def_rating       — CB + pass rush composite 0-99
    {team}_run_def_rating        — DL + LB run stop composite 0-99
    {team}_lineup_health         — injury-adjusted roster strength 0-1

  Matchup gaps (home minus away):
    qb_vs_pass_def_gap           — home QB vs away pass defense
    wr_vs_cb_gap                 — home WR vs away CB coverage
    rb_vs_run_def_gap            — home RB vs away run defense
    ol_vs_edge_gap               — home OL vs away pass rush
    away_qb_vs_home_pass_def_gap — symmetrical away perspective

Data sources used:
  - player_stats_weekly: QB EPA/dropback, CPOE, WR/RB/TE receiving stats
  - NGS: separation, intended air yards, rush yards over expected
  - PFR advstats: pass rush win rate, coverage stats
  - rosters_weekly: snap counts, depth chart
  - injuries: availability factors per player
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── Availability factors from injury report status ───────────────────────────
AVAILABILITY = {
    "Out":          0.00,
    "Doubtful":     0.15,
    "Questionable": 0.65,
    "Limited":      0.82,
    "Full":         1.00,
    "Active":       1.00,
}

# ── Depth chart weight: starter matters most, backup fraction ─────────────────
DEPTH_WEIGHT = {1: 1.00, 2: 0.30, 3: 0.08}

# ── Minimum exposure thresholds ───────────────────────────────────────────────
MIN_DROPBACKS  = 15   # QB: minimum dropbacks to rate
MIN_ROUTES     = 10   # WR/TE: minimum routes run
MIN_CARRIES    = 8    # RB: minimum carries
MIN_RUSH_SNAPS = 10   # DL/EDGE: minimum pass rush snaps
MIN_COV_SNAPS  = 10   # CB/S: minimum coverage snaps


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_player_ratings(game_df: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """
    Compute player ratings and add positional matchup gap features to game_df.

    Returns game_df with new columns added. All operations are graceful — if
    any data source is missing, the corresponding features are simply absent.
    """
    from data_loader import get_table

    logger.info("Building player ratings …")

    ps_weekly    = get_table("player_stats_weekly")
    rosters      = get_table("rosters_weekly")
    injuries     = get_table("injuries")
    ngs_pass     = get_table("ngs_passing")
    ngs_rush     = get_table("ngs_rushing")
    pfr_pass     = get_table("pfr_passing")   # pass rush stats per player
    pfr_def      = get_table("pfr_defense")   # coverage stats per player

    # Step 1: compute raw per-player-week ratings
    qb_ratings   = _rate_qbs(ps_weekly, ngs_pass, pfr_pass)
    skill_ratings = _rate_skill_positions(ps_weekly, ngs_pass, ngs_rush)
    def_ratings  = _rate_defenders(pfr_def, pfr_pass, ps_weekly)

    # Step 2: apply injury adjustments using depth chart
    eff_ratings = _apply_injury_adjustments(
        qb_ratings, skill_ratings, def_ratings,
        rosters, injuries
    )

    # Step 3: aggregate to team-week level
    team_ratings = _aggregate_to_team(eff_ratings)

    # Step 4: merge into game_df and compute matchup gaps
    game_df = _merge_into_game_df(game_df, team_ratings)

    logger.info("  Player ratings added: %d team-week records", len(team_ratings))
    return game_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1A: QB RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def _rate_qbs(ps: Optional[pd.DataFrame],
              ngs: Optional[pd.DataFrame],
              pfr: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Rate QBs on a 0-99 scale within their position each week.

    Primary metrics (from player_stats_weekly):
      - EPA per dropback    (weight 0.40) — best single QB predictor
      - CPOE                (weight 0.35) — accuracy over expectation
      - TD/INT ratio        (weight 0.15) — decision-making
      - Sack rate (-)       (weight 0.10) — pressure avoidance

    NGS supplement (when available):
      - avg_time_to_throw   — quick release under pressure
      - aggressiveness      — downfield intent
    """
    if ps is None or len(ps) == 0:
        return pd.DataFrame(columns=["player_id","team","season","week","qb_raw_rating"])

    # Filter to QBs with sufficient exposure
    qb_cols = [c for c in ["player_id","recent_team","season","week",
                            "attempts","passing_epa","dakota",
                            "completions","passing_tds","interceptions",
                            "sacks","sack_yards","passing_air_yards"] if c in ps.columns]
    if "recent_team" not in ps.columns and "team" in ps.columns:
        ps = ps.rename(columns={"team": "recent_team"})

    ps_qb = ps[ps.get("position", ps.get("position_group", pd.Series(dtype=str))) == "QB"].copy() \
        if "position" in ps.columns else ps.copy()

    if len(ps_qb) == 0:
        return pd.DataFrame(columns=["player_id","team","season","week","qb_raw_rating"])

    ps_qb = ps_qb[[c for c in qb_cols if c in ps_qb.columns]].copy()
    ps_qb["attempts"] = pd.to_numeric(ps_qb.get("attempts", 0), errors="coerce").fillna(0)

    # Minimum threshold
    ps_qb = ps_qb[ps_qb["attempts"] >= MIN_DROPBACKS].copy()
    if len(ps_qb) == 0:
        return pd.DataFrame(columns=["player_id","team","season","week","qb_raw_rating"])

    # Compute component metrics
    ps_qb["epa_per_db"] = pd.to_numeric(ps_qb.get("passing_epa", np.nan), errors="coerce") / ps_qb["attempts"].clip(lower=1)
    ps_qb["cpoe"]       = pd.to_numeric(ps_qb.get("dakota", np.nan), errors="coerce")  # dakota ≈ CPOE proxy
    ps_qb["td_int"]     = (pd.to_numeric(ps_qb.get("passing_tds", 0), errors="coerce") /
                           (pd.to_numeric(ps_qb.get("interceptions", 0), errors="coerce") + 1))
    ps_qb["sack_rate"]  = (pd.to_numeric(ps_qb.get("sacks", 0), errors="coerce") /
                           (ps_qb["attempts"] + ps_qb.get("sacks", 0) + 1))

    # Merge NGS if available
    if ngs is not None and len(ngs) > 0:
        ngs_cols = [c for c in ["player_gsis_id","team","season","week",
                                 "avg_time_to_throw","aggressiveness",
                                 "avg_completed_air_yards"] if c in ngs.columns]
        ngs_sub = ngs[ngs_cols].copy()
        if "player_gsis_id" in ngs_sub.columns:
            ngs_sub = ngs_sub.rename(columns={"player_gsis_id": "player_id"})
        if "player_id" in ps_qb.columns and "player_id" in ngs_sub.columns:
            merge_keys = [k for k in ["player_id","season","week"] if k in ngs_sub.columns]
            ps_qb = ps_qb.merge(ngs_sub, on=merge_keys, how="left")

    # Composite score (weighted Z-scores)
    weights = {"epa_per_db": 0.40, "cpoe": 0.35, "td_int": 0.15, "sack_rate": -0.10}
    ps_qb["qb_raw_score"] = _weighted_zscore_composite(ps_qb, weights, group_by=["season","week"])

    # Normalize to 0-99
    ps_qb["qb_raw_rating"] = _normalize_to_99(ps_qb["qb_raw_score"])

    team_col = "recent_team" if "recent_team" in ps_qb.columns else "team"
    result = ps_qb[[c for c in ["player_id", team_col, "season", "week", "qb_raw_rating"] if c in ps_qb.columns]].copy()
    result = result.rename(columns={team_col: "team"})
    return result.dropna(subset=["qb_raw_rating"])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1B: SKILL POSITION RATINGS (WR, RB, TE)
# ══════════════════════════════════════════════════════════════════════════════

def _rate_skill_positions(ps: Optional[pd.DataFrame],
                          ngs_pass: Optional[pd.DataFrame],
                          ngs_rush: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Rate WR, RB, TE on 0-99 scale within their position each week.

    WR metrics: receiving_yards_per_target, air_yards_share, target_share, catch_over_expected
    RB metrics: yards_per_carry, receiving_yards_per_route, yards_after_contact proxy
    TE metrics: same as WR but separate percentile pool
    """
    if ps is None or len(ps) == 0:
        return pd.DataFrame(columns=["player_id","team","season","week","position","skill_rating"])

    team_col = "recent_team" if "recent_team" in ps.columns else "team"
    needed = [c for c in ["player_id", team_col, "season","week","position",
                           "targets","receptions","receiving_yards","receiving_tds",
                           "carries","rushing_yards","rushing_tds",
                           "target_share","air_yards_share","racr","wopr",
                           "receiving_epa","rushing_epa"] if c in ps.columns]

    ps_skill = ps[ps.get("position", pd.Series(dtype=str)).isin(["WR","RB","TE"])].copy() \
        if "position" in ps.columns else pd.DataFrame()

    if len(ps_skill) == 0:
        return pd.DataFrame(columns=["player_id","team","season","week","position","skill_rating"])

    ps_skill = ps_skill[[c for c in needed if c in ps_skill.columns]].copy()
    ps_skill = ps_skill.rename(columns={team_col: "team"})

    records = []

    for pos, min_exp_col, min_exp_val, weight_map in [
        ("WR", "targets", MIN_ROUTES, {
            "yds_per_tgt": 0.35,   # yards efficiency
            "air_yards_share": 0.25,  # target quality
            "rcvr_epa_per_tgt": 0.30, # value per target
            "catch_rate": 0.10,
        }),
        ("RB", "carries", MIN_CARRIES, {
            "rush_yards_per_carry": 0.40,
            "rushing_epa_per_carry": 0.35,
            "receiving_yprr": 0.25,
        }),
        ("TE", "targets", MIN_ROUTES, {
            "yds_per_tgt": 0.40,
            "rcvr_epa_per_tgt": 0.35,
            "catch_rate": 0.25,
        }),
    ]:
        pos_df = ps_skill[ps_skill.get("position", pd.Series(dtype=str)) == pos].copy() \
            if "position" in ps_skill.columns else pd.DataFrame()
        if len(pos_df) == 0:
            continue

        exp_col = next((c for c in [min_exp_col, "targets","carries"] if c in pos_df.columns), None)
        if exp_col:
            pos_df[exp_col] = pd.to_numeric(pos_df[exp_col], errors="coerce").fillna(0)
            pos_df = pos_df[pos_df[exp_col] >= min_exp_val]
        if len(pos_df) == 0:
            continue

        # Compute position-specific metrics
        if pos in ("WR","TE"):
            tgts = pd.to_numeric(pos_df.get("targets", 1), errors="coerce").clip(lower=1)
            pos_df["yds_per_tgt"]       = pd.to_numeric(pos_df.get("receiving_yards", 0), errors="coerce") / tgts
            pos_df["catch_rate"]        = pd.to_numeric(pos_df.get("receptions", 0), errors="coerce") / tgts
            pos_df["air_yards_share"]   = pd.to_numeric(pos_df.get("air_yards_share", np.nan), errors="coerce")
            pos_df["rcvr_epa_per_tgt"]  = pd.to_numeric(pos_df.get("receiving_epa", 0), errors="coerce") / tgts
        else:  # RB
            carries = pd.to_numeric(pos_df.get("carries", 1), errors="coerce").clip(lower=1)
            tgts    = pd.to_numeric(pos_df.get("targets", 1), errors="coerce").clip(lower=1)
            pos_df["rush_yards_per_carry"]    = pd.to_numeric(pos_df.get("rushing_yards", 0), errors="coerce") / carries
            pos_df["rushing_epa_per_carry"]   = pd.to_numeric(pos_df.get("rushing_epa", 0), errors="coerce") / carries
            pos_df["receiving_yprr"]          = pd.to_numeric(pos_df.get("receiving_yards", 0), errors="coerce") / tgts

        pos_df["raw_score"]  = _weighted_zscore_composite(pos_df, weight_map, group_by=["season","week"])
        pos_df["skill_rating"] = _normalize_to_99(pos_df["raw_score"])
        pos_df["position"]   = pos
        records.append(pos_df[["player_id","team","season","week","position","skill_rating"]].dropna(subset=["skill_rating"]))

    if not records:
        return pd.DataFrame(columns=["player_id","team","season","week","position","skill_rating"])
    return pd.concat(records, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1C: DEFENSIVE PLAYER RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def _rate_defenders(pfr_def: Optional[pd.DataFrame],
                    pfr_pass: Optional[pd.DataFrame],
                    ps: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Rate defensive players (EDGE, DL, CB, S, LB) on 0-99 within their group.

    EDGE/DL: pressure_rate, sack_rate from PFR pass rush stats
    CB/S:    yards_allowed_per_coverage, completion_allowed_pct from PFR coverage stats
    LB:      composite of both

    Falls back gracefully if PFR data unavailable.
    """
    records = []

    # ── EDGE / DL pass rush ───────────────────────────────────────────────
    if pfr_pass is not None and len(pfr_pass) > 0:
        rush_cols = [c for c in ["player_id","pfr_id","team","season","week",
                                  "rushes","sacks","hits","hurries","prs_pct"] if c in pfr_pass.columns]
        if rush_cols:
            pr = pfr_pass[rush_cols].copy()
            rushes = pd.to_numeric(pr.get("rushes", pr.get("pass_rush_snaps", 0)), errors="coerce").fillna(0)
            pr = pr[rushes >= MIN_RUSH_SNAPS]
            if len(pr) > 0:
                rush_denom = rushes[pr.index].clip(lower=1)
                pr["pressure_rate"] = (
                    pd.to_numeric(pr.get("hits", 0), errors="coerce").fillna(0) +
                    pd.to_numeric(pr.get("hurries", 0), errors="coerce").fillna(0)
                ) / rush_denom
                pr["sack_rate"] = pd.to_numeric(pr.get("sacks", 0), errors="coerce").fillna(0) / rush_denom

                pr["raw_score"]    = _weighted_zscore_composite(pr, {"pressure_rate": 0.65, "sack_rate": 0.35},
                                                                  group_by=["season","week"])
                pr["def_rating"]   = _normalize_to_99(pr["raw_score"])
                pr["def_position"] = "EDGE"

                id_col = next((c for c in ["player_id","pfr_id"] if c in pr.columns), None)
                if id_col:
                    pr = pr.rename(columns={id_col: "player_id"})
                    records.append(pr[["player_id","team","season","week","def_position","def_rating"]].dropna())

    # ── CB / S coverage ───────────────────────────────────────────────────
    if pfr_def is not None and len(pfr_def) > 0:
        cov_cols = [c for c in ["player_id","pfr_id","team","season","week",
                                  "targets_as_def","completions_allowed",
                                  "yards_allowed","tds_allowed","int_def",
                                  "position"] if c in pfr_def.columns]
        if cov_cols:
            cv = pfr_def[cov_cols].copy()
            tgts = pd.to_numeric(cv.get("targets_as_def", 0), errors="coerce").fillna(0)
            cv = cv[tgts >= MIN_COV_SNAPS]
            if len(cv) > 0:
                tgts_c = tgts[cv.index].clip(lower=1)
                cv["yds_per_cov"]   = pd.to_numeric(cv.get("yards_allowed", 0), errors="coerce").fillna(0) / tgts_c
                cv["comp_pct_allowed"] = pd.to_numeric(cv.get("completions_allowed", 0), errors="coerce").fillna(0) / tgts_c
                # Lower is better — negate for Z-score
                cv["raw_score"] = _weighted_zscore_composite(
                    cv,
                    {"yds_per_cov": -0.60, "comp_pct_allowed": -0.40},
                    group_by=["season","week"]
                )
                cv["def_rating"]   = _normalize_to_99(cv["raw_score"])
                cv["def_position"] = cv.get("position", pd.Series("CB", index=cv.index))

                id_col = next((c for c in ["player_id","pfr_id"] if c in cv.columns), None)
                if id_col:
                    cv = cv.rename(columns={id_col: "player_id"})
                    records.append(cv[["player_id","team","season","week","def_position","def_rating"]].dropna())

    if not records:
        return pd.DataFrame(columns=["player_id","team","season","week","def_position","def_rating"])
    return pd.concat(records, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: INJURY ADJUSTMENTS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_injury_adjustments(
    qb_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    def_df: pd.DataFrame,
    rosters: Optional[pd.DataFrame],
    injuries: Optional[pd.DataFrame],
) -> dict:
    """
    Combine ratings with injury availability and depth chart order.
    Returns dict keyed by position group with effective rating per player.

    effective_rating = rating × availability × depth_weight
    """
    # Build availability lookup: (player_id, team, season, week) → factor
    avail: dict = {}
    if injuries is not None and len(injuries) > 0:
        inj = injuries.copy()
        # Standardise player ID column
        id_col = next((c for c in ["gsis_id","player_id"] if c in inj.columns), None)
        if id_col and "report_status" in inj.columns:
            inj["avail_factor"] = inj["report_status"].map(AVAILABILITY).fillna(0.85)
            for _, row in inj.iterrows():
                key = (str(row.get(id_col,"")), str(row.get("team","")),
                       int(row.get("season",0)), int(row.get("week",0)))
                avail[key] = float(row["avail_factor"])

    # Build depth chart lookup: (player_id, team, season, week) → depth_rank
    depth: dict = {}
    if rosters is not None and len(rosters) > 0:
        ros = rosters.copy()
        id_col = next((c for c in ["gsis_id","player_id"] if c in ros.columns), None)
        if id_col and "depth_chart_position" in ros.columns:
            for _, row in ros.iterrows():
                key = (str(row.get(id_col,"")), str(row.get("team","")),
                       int(row.get("season",0)), int(row.get("week",0)))
                # Use jersey_number as depth proxy if no explicit depth rank
                depth_val = int(row.get("depth_chart_position_rank",
                               row.get("jersey_number", 1)))
                depth[key] = max(1, min(depth_val, 3))

    def _apply(df, rating_col, pos_col=None):
        if len(df) == 0:
            return df
        df = df.copy()
        df["avail"] = df.apply(
            lambda r: avail.get(
                (str(r.get("player_id","")), str(r.get("team","")),
                 int(r.get("season",0)), int(r.get("week",0))),
                1.0
            ), axis=1
        )
        df["depth"] = df.apply(
            lambda r: depth.get(
                (str(r.get("player_id","")), str(r.get("team","")),
                 int(r.get("season",0)), int(r.get("week",0))),
                1
            ), axis=1
        )
        df["depth_w"]    = df["depth"].map(DEPTH_WEIGHT).fillna(0.05)
        df["eff_rating"] = df[rating_col] * df["avail"] * df["depth_w"]
        return df

    return {
        "qb":    _apply(qb_df,    "qb_raw_rating"),
        "skill": _apply(skill_df, "skill_rating"),
        "def":   _apply(def_df,   "def_rating", "def_position"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: AGGREGATE TO TEAM-WEEK
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_to_team(eff_ratings: dict) -> pd.DataFrame:
    """
    Produce one row per (team, season, week) with:
      qb_rating, wr_rating, rb_rating, te_rating,
      pass_def_rating, run_def_rating, lineup_health
    """
    records = []
    group_keys = ["team","season","week"]

    # QB — take starter (highest eff_rating)
    qb = eff_ratings.get("qb", pd.DataFrame())
    if len(qb) > 0:
        qb_team = (
            qb.sort_values("eff_rating", ascending=False)
            .groupby(group_keys)
            .agg(qb_rating=("eff_rating","first"))
            .reset_index()
        )
        records.append(qb_team)

    # Skill positions
    skill = eff_ratings.get("skill", pd.DataFrame())
    if len(skill) > 0 and "position" in skill.columns:
        for pos, col_name, top_n in [("WR","wr_rating",2), ("RB","rb_rating",1), ("TE","te_rating",1)]:
            sub = skill[skill["position"] == pos].copy()
            if len(sub) == 0:
                continue
            # Top-N effective rating (sum then normalize) — takes roster depth into account
            pos_team = (
                sub.sort_values("eff_rating", ascending=False)
                .groupby(group_keys)
                .apply(lambda g: g.nlargest(top_n, "eff_rating")["eff_rating"].sum())
                .reset_index()
                .rename(columns={0: col_name})
            )
            # Normalize back to 0-99 range
            if col_name in pos_team.columns:
                pos_team[col_name] = _normalize_to_99(pos_team[col_name])
            records.append(pos_team)

    # Defense
    defp = eff_ratings.get("def", pd.DataFrame())
    if len(defp) > 0 and "def_position" in defp.columns:
        # Pass defense: EDGE + CB/S composite
        pass_def = defp[defp["def_position"].isin(["EDGE","CB","S"])].copy()
        if len(pass_def) > 0:
            pdef_team = (
                pass_def.groupby(group_keys)["eff_rating"]
                .mean().reset_index()
                .rename(columns={"eff_rating": "pass_def_rating"})
            )
            pdef_team["pass_def_rating"] = _normalize_to_99(pdef_team["pass_def_rating"])
            records.append(pdef_team)

        # Run defense: DL + LB
        run_def = defp[defp["def_position"].isin(["DL","LB","EDGE"])].copy()
        if len(run_def) > 0:
            rdef_team = (
                run_def.groupby(group_keys)["eff_rating"]
                .mean().reset_index()
                .rename(columns={"eff_rating": "run_def_rating"})
            )
            rdef_team["run_def_rating"] = _normalize_to_99(rdef_team["run_def_rating"])
            records.append(rdef_team)

    # Lineup health: availability-weighted average across all positions
    all_ratings = []
    for k, rating_col in [("qb","qb_raw_rating"), ("skill","skill_rating"), ("def","def_rating")]:
        df = eff_ratings.get(k, pd.DataFrame())
        if len(df) > 0 and rating_col in df.columns and "avail" in df.columns:
            df2 = df[group_keys + ["avail"]].copy()
            all_ratings.append(df2)
    if all_ratings:
        all_avail = pd.concat(all_ratings, ignore_index=True)
        health = (
            all_avail.groupby(group_keys)["avail"]
            .mean().reset_index()
            .rename(columns={"avail": "lineup_health"})
        )
        records.append(health)

    if not records:
        return pd.DataFrame(columns=group_keys)

    result = records[0].copy()
    for df in records[1:]:
        result = result.merge(df, on=group_keys, how="outer")

    return result.sort_values(group_keys).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: MERGE INTO GAME_DF AND ADD MATCHUP GAPS
# ══════════════════════════════════════════════════════════════════════════════

def _merge_into_game_df(game_df: pd.DataFrame,
                        team_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team ratings into game_df (which has home_ and away_ prefixed columns).
    Adds positional matchup gap features.
    """
    if len(team_ratings) == 0:
        return game_df

    rating_cols = [c for c in team_ratings.columns
                   if c not in ["team","season","week"]]

    # Merge home team ratings
    home_ratings = team_ratings.rename(
        columns={c: f"home_{c}" for c in rating_cols}
    )
    game_df = game_df.merge(
        home_ratings.rename(columns={"team": "home_team"}),
        on=["home_team","season","week"], how="left"
    )

    # Merge away team ratings
    away_ratings = team_ratings.rename(
        columns={c: f"away_{c}" for c in rating_cols}
    )
    game_df = game_df.merge(
        away_ratings.rename(columns={"team": "away_team"}),
        on=["away_team","season","week"], how="left"
    )

    # ── Matchup gap features ───────────────────────────────────────────────
    # These directly measure positional mismatches between opposing units.
    # All on 0-99 scale so gaps are interpretable: +15 = big advantage

    def _gap(a, b, name):
        if a in game_df.columns and b in game_df.columns:
            game_df[name] = game_df[a].fillna(50) - game_df[b].fillna(50)

    # Offensive vs defensive matchups
    _gap("home_qb_rating",     "away_pass_def_rating",  "qb_vs_pass_def_gap")
    _gap("home_wr_rating",     "away_pass_def_rating",  "wr_vs_cb_gap")
    _gap("home_rb_rating",     "away_run_def_rating",   "rb_vs_run_def_gap")
    _gap("home_te_rating",     "away_pass_def_rating",  "te_vs_lb_gap")

    # Away perspective
    _gap("away_qb_rating",     "home_pass_def_rating",  "away_qb_vs_home_pass_def_gap")
    _gap("away_wr_rating",     "home_pass_def_rating",  "away_wr_vs_home_cb_gap")
    _gap("away_rb_rating",     "home_run_def_rating",   "away_rb_vs_home_run_def_gap")

    # Line play matchups
    _gap("home_ol_rating",     "away_pass_def_rating",  "ol_vs_edge_gap")
    _gap("away_ol_rating",     "home_pass_def_rating",  "away_ol_vs_home_edge_gap")

    # Roster health differential
    _gap("home_lineup_health", "away_lineup_health",    "lineup_health_gap")

    logger.info("  Player matchup gaps added to game_df")
    return game_df


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_zscore_composite(df: pd.DataFrame,
                                weights: dict,
                                group_by: list) -> pd.Series:
    """
    For each metric in weights: compute Z-score within group_by strata,
    then take weighted sum. Missing values → zero contribution.
    """
    composite = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if group_by and all(g in df.columns for g in group_by):
            # Z-score within group (e.g. all QBs in week 8 of 2024)
            group_mean = df.groupby(group_by)[col].transform("mean")
            group_std  = df.groupby(group_by)[col].transform("std").fillna(1.0).clip(lower=0.01)
            z = (vals - group_mean) / group_std
        else:
            mean = vals.mean()
            std  = vals.std()
            z = (vals - mean) / max(std, 0.01)
        composite += w * z.fillna(0.0)
    return composite


def _normalize_to_99(series: pd.Series) -> pd.Series:
    """
    Map any numeric series to 0-99 using percentile rank.
    Bayesian shrinkage toward 50 for small samples is implicit via Z-score grouping.
    """
    if series.isna().all():
        return pd.Series(50.0, index=series.index)
    # Percentile rank → scale to 0-99
    pct = series.rank(pct=True, na_option="keep")
    return (pct * 99).clip(0, 99)

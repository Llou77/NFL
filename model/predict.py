"""
predict.py
==========
Generates score predictions for all upcoming (unplayed) games.
Uses the trained ensemble and confidence scoring.
Outputs predictions_latest.json to data/predictions/.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent
PRED_DIR     = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR  = PRED_DIR / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def generate_predictions(
    game_df: pd.DataFrame,
    models: dict,
    current_season: int = 2026,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for all games in game_df that have no result yet
    (target_home_score is NaN).

    Parameters
    ----------
    game_df : full feature matrix (from feature_engineering + h2h)
    models  : dict returned by train.load_models()
    current_season : season to predict
    save    : if True, save predictions_latest.json

    Returns
    -------
    DataFrame of predictions with confidence scores
    """
    from confidence import compute_confidence_batch

    # Select unplayed games for current season
    upcoming = game_df[
        (game_df["season"] == current_season) &
        game_df["target_home_score"].isna()
    ].copy()

    if len(upcoming) == 0:
        logger.info("No upcoming games found.")
        return pd.DataFrame()

    logger.info(f"Generating predictions for {len(upcoming)} upcoming games …")

    feature_cols = models["feature_cols"]
    imputer      = models["imputer"]
    scaler       = models["scaler"]

    # Preprocess — two versions:
    # X_sc:   imputed + scaled (for Ridge and NN)
    # X_tree: imputed only for Ridge/NN columns, NaN preserved for tree features
    X_raw = upcoming[feature_cols].values
    nan_mask = np.isnan(X_raw)

    X_imp  = imputer.transform(X_raw)
    X_sc   = scaler.transform(X_imp)

    # Re-introduce NaN for tree models (they handle it natively via learned splits)
    X_tree = X_imp.copy()
    X_tree[nan_mask] = np.nan

    # ── Sub-model predictions ─────────────────────────────────────────────
    sub_preds_home = {}
    sub_preds_away = {}

    # Ridge — imputed + scaled
    sub_preds_home["ridge"] = models["ridge_home"].predict(X_sc)
    sub_preds_away["ridge"] = models["ridge_away"].predict(X_sc)

    # XGBoost — NaN-preserved (native NaN handling)
    if "xgb_home" in models:
        sub_preds_home["xgb"] = models["xgb_home"].predict(X_tree)
        sub_preds_away["xgb"] = models["xgb_away"].predict(X_tree)

    # LightGBM — NaN-preserved
    if "lgbm_home" in models:
        try:
            sub_preds_home["lgbm"] = models["lgbm_home"].predict(X_tree)
            sub_preds_away["lgbm"] = models["lgbm_away"].predict(X_tree)
        except Exception:
            pass

    # Neural Network — imputed + scaled (cannot handle NaN)
    if "nn" in models:
        from train import _nn_predict
        nn_h, nn_a = _nn_predict(models["nn"], X_sc)
        sub_preds_home["nn"] = nn_h
        sub_preds_away["nn"] = nn_a
    elif "nn_home_mlp" in models:
        sub_preds_home["nn"] = models["nn_home_mlp"].predict(X_sc)
        sub_preds_away["nn"] = models["nn_away_mlp"].predict(X_sc)

    # ── Meta-learner final predictions ────────────────────────────────────
    meta_X_home = np.column_stack(list(sub_preds_home.values()))
    meta_X_away = np.column_stack(list(sub_preds_away.values()))

    raw_home = models["meta_home"].predict(meta_X_home)
    raw_away = models["meta_away"].predict(meta_X_away)

    # ── Sanity check ─────────────────────────────────────────────────────
    avg_total = float(np.mean(raw_home + raw_away))
    avg_home  = float(np.mean(raw_home))
    if avg_total > 65 or avg_home < 10 or avg_home > 40:
        logger.error(
            "PREDICTION SANITY FAIL: avg_home=%.1f avg_total=%.1f — "
            "expected home ~24pts, total ~45pts. "
            "Model may be stale or trained on wrong targets. "
            "Re-run --mode full to retrain.",
            avg_home, avg_total
        )
        # Clamp to plausible range rather than publish garbage
        scale_h = 24.0 / max(avg_home, 1.0)
        scale_a = 21.8 / max(float(np.mean(raw_away)), 1.0)
        raw_home = raw_home * scale_h
        raw_away = raw_away * scale_a
        logger.warning("Applied emergency rescaling: home×%.3f away×%.3f", scale_h, scale_a)

    # ── Variance calibration ──────────────────────────────────────────────
    calib_path = ROOT / "model" / "saved" / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)

        spread_scale = calib.get("spread_scale", 1.0)
        total_scale  = calib.get("total_scale",  1.0)

        if spread_scale > 1.0 or total_scale > 1.0:
            mean_h       = np.mean(raw_home)
            mean_a       = np.mean(raw_away)
            mean_total   = mean_h + mean_a
            mean_spread  = mean_h - mean_a
            raw_spread   = raw_home - raw_away
            raw_total    = raw_home + raw_away

            # 1. Calibrate spread (expand around mean)
            cal_spread = mean_spread + (raw_spread - mean_spread) * spread_scale

            # 2. Calibrate total independently (expand around mean)
            cal_total  = mean_total  + (raw_total  - mean_total)  * total_scale

            # 3. Reconstruct home/away from calibrated spread + total
            raw_home = (cal_total + cal_spread) / 2.0
            raw_away = (cal_total - cal_spread) / 2.0

        logger.info("  Applied variance calibration (spread_scale=%.3f total_scale=%.3f)",
                    spread_scale, total_scale)

    # Clip to realistic range and round
    final_home = np.clip(np.round(raw_home).astype(int), 0, 65)
    final_away = np.clip(np.round(raw_away).astype(int), 0, 65)

    # ── Confidence intervals (bootstrap-style from sub-model spread) ──────
    home_preds_matrix = np.column_stack(list(sub_preds_home.values()))
    away_preds_matrix = np.column_stack(list(sub_preds_away.values()))

    ci_home_lo = np.percentile(home_preds_matrix, 20, axis=1).astype(int)
    ci_home_hi = np.percentile(home_preds_matrix, 80, axis=1).astype(int)
    ci_away_lo = np.percentile(away_preds_matrix, 20, axis=1).astype(int)
    ci_away_hi = np.percentile(away_preds_matrix, 80, axis=1).astype(int)

    # ── Assemble predictions DataFrame ───────────────────────────────────
    upcoming = upcoming.copy()
    upcoming["predicted_home_score"] = final_home
    upcoming["predicted_away_score"] = final_away
    upcoming["predicted_total"]      = final_home + final_away
    upcoming["predicted_spread"]     = final_home - final_away
    upcoming["ci_home_lo"] = ci_home_lo
    upcoming["ci_home_hi"] = ci_home_hi
    upcoming["ci_away_lo"] = ci_away_lo
    upcoming["ci_away_hi"] = ci_away_hi

    # Sub-model prediction dict per game (for confidence)
    all_sub = {}
    for i, gid in enumerate(upcoming["game_id"].values):
        all_sub[gid] = {
            model_name: (
                float(sub_preds_home[model_name][i]),
                float(sub_preds_away[model_name][i]),
            )
            for model_name in sub_preds_home
        }

    # Add confidence scores
    gen_ts = datetime.now(timezone.utc)
    upcoming = compute_confidence_batch(upcoming, all_sub, gen_ts)

    # Edge vs Vegas lines
    upcoming = _add_edge_signals(upcoming)

    if save:
        _save_predictions(upcoming, gen_ts)

    return upcoming


def _add_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model vs. book deltas for spread and total."""
    # Load lean thresholds dynamically
    from bayesian_optimizer import load_weights as _lw
    _lw_vals = _lw()
    SPREAD_THR = float(_lw_vals.get("spread_lean_thr", 1.5))
    TOTAL_THR  = float(_lw_vals.get("total_lean_thr",  1.5))

    if "spread_line" in df.columns:
        df["model_spread"] = df["predicted_spread"]
        df["book_spread"]  = df["spread_line"]
        df["spread_edge"]  = df["model_spread"] - (-df["book_spread"])
        df["spread_lean"]  = df["spread_edge"].apply(
            lambda x: "HOME" if x > SPREAD_THR else ("AWAY" if x < -SPREAD_THR else "PUSH")
        )

        # Opening spread divergence — model vs. pre-public-money line
        # This is the cleaner betting signal: model disagrees with oddsmaker's pure estimate
        if "opening_spread" in df.columns:
            df["opening_spread_edge"] = df["model_spread"] - (-df["opening_spread"].fillna(df["book_spread"]))

    if "total_line" in df.columns:
        df["model_total"] = df["predicted_total"]
        df["book_total"]  = df["total_line"]
        df["total_edge"]  = df["model_total"] - df["book_total"]
        df["total_lean"]  = df["total_edge"].apply(
            lambda x: "OVER" if x > TOTAL_THR else ("UNDER" if x < -TOTAL_THR else "PUSH")
        )

    # Suppress edges for LOW and WEAK confidence
    # (LOW = insufficient data to trust spread signal; WEAK = almost no data)
    if "confidence_label" in df.columns:
        for col in ["spread_lean", "total_lean"]:
            if col in df.columns:
                df.loc[df["confidence_label"].isin(["WEAK", "LOW"]), col] = "SUPPRESSED"

    return df


def _save_predictions(df: pd.DataFrame, gen_ts: datetime) -> None:
    """Save predictions as JSON in the format expected by the frontend."""
    records = []
    for _, row in df.iterrows():
        # Extract team names — try multiple column name variants produced by the pivot,
        # then fall back to parsing the game_id (format: YEAR_WK_AWAY_HOME)
        game_id = str(row.get("game_id", ""))
        home_team, away_team = _extract_teams(row, game_id)

        rec = {
            "game_id":              game_id,
            "season":               int(row.get("season", 2026)),
            "week":                 int(row.get("week", 0)),
            "game_type":            str(row.get("game_type", "REG")),
            "game_date":            str(row.get("game_date", ""))[:10],
            "home_team":            home_team,
            "away_team":            away_team,
            "predicted_home_score": int(row.get("predicted_home_score", 0)),
            "predicted_away_score": int(row.get("predicted_away_score", 0)),
            "predicted_total":      int(row.get("predicted_total", 0)),
            "predicted_spread":     int(row.get("predicted_spread", 0)),
            "ci_home":              [int(row.get("ci_home_lo", 0)), int(row.get("ci_home_hi", 0))],
            "ci_away":              [int(row.get("ci_away_lo", 0)), int(row.get("ci_away_hi", 0))],
            "confidence_score":     float(row.get("confidence_score", 0.5)),
            "confidence_label":     str(row.get("confidence_label", "LOW")),
            "confidence_breakdown": {
                "model_agreement":       float(row.get("conf_model_agreement", 0.5)),
                "feature_completeness":  float(row.get("conf_feature_completeness", 0.5)),
                "h2h_data_quality":      float(row.get("conf_h2h_data_quality", 0.5)),
                "injury_data_freshness": float(row.get("conf_injury_data_freshness", 0.5)),
            },
            "book_spread":          _safe_float(row.get("book_spread")),
            "book_total":           _safe_float(row.get("book_total")),
            "opening_spread":       _safe_float(row.get("opening_spread")),
            "opening_total":        _safe_float(row.get("opening_total")),
            "opening_spread_edge":  _safe_float(row.get("opening_spread_edge")),
            "market_win_prob":      _safe_float(row.get("market_win_prob_vigfree")),
            "spread_edge":   _safe_float(row.get("spread_edge")),
            "total_edge":    _safe_float(row.get("total_edge")),
            "spread_lean":   str(row.get("spread_lean", "PUSH")),
            "total_lean":    str(row.get("total_lean", "PUSH")),
            "is_dome":       int(row.get("is_dome", 0)),
            "is_primetime":  int(row.get("is_primetime", 0)),
            "is_international": int(row.get("is_international", 0)),
            "is_division_game": int(row.get("is_division_game", 0)),
            "temp":          _safe_float(row.get("temp")),
            "wind":          _safe_float(row.get("wind")),
            "top_features":  _get_top_features(row),
            "generated_at":  gen_ts.isoformat(),
        }
        records.append(rec)

    output = {
        "generated_at":   gen_ts.isoformat(),
        "season":         2026,
        "n_games":        len(records),
        "predictions":    records,
    }

    # Save latest
    latest_path = PRED_DIR / "predictions_latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved {len(records)} predictions → {latest_path}")

    # Archive
    ts_str = gen_ts.strftime("%Y%m%d_%H%M")
    archive_path = ARCHIVE_DIR / f"predictions_{ts_str}.json"
    with open(archive_path, "w") as f:
        json.dump(output, f, indent=2)


def _extract_teams(row: pd.Series, game_id: str) -> tuple:
    """
    Extract home and away team abbreviations from the prediction row.
    Tries multiple column name variants produced by the pivot, then
    falls back to parsing game_id (nflverse format: YEAR_WK_AWAY_HOME).
    """
    # Try every column variant the pivot might produce
    candidate_home = [
        "home_team", "home_team_x", "home_home_team",
        "home_team_y", "home_away_team",
    ]
    candidate_away = [
        "away_team", "away_team_x", "away_away_team",
        "away_team_y", "away_home_team",
    ]

    home = ""
    for col in candidate_home:
        val = row.get(col, "")
        if val and str(val).strip() and str(val) != "nan":
            home = str(val).strip()
            break

    away = ""
    for col in candidate_away:
        val = row.get(col, "")
        if val and str(val).strip() and str(val) != "nan":
            away = str(val).strip()
            break

    # Fallback: parse game_id — nflverse format is YEAR_WK_AWAY_HOME
    if (not home or not away) and game_id:
        parts = game_id.split("_")
        if len(parts) >= 4:
            # parts: [2026, 01, AWAY, HOME]
            away = away or parts[2]
            home = home or parts[3]

    return home, away


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 2)
    except Exception:
        return None


def _get_top_features(row: pd.Series, n: int = 6) -> dict:
    """Extract the top N most informative feature values for display."""
    display_features = [
        "home_off_epa_per_play_r8", "away_off_epa_per_play_r8",
        "home_def_epa_per_play_r8", "away_def_epa_per_play_r8",
        "elo_gap", "cpoe_gap_r8",
        "rest_diff", "home_off_cpoe_r8", "away_off_cpoe_r8",
        "h2h_avg_margin", "h2h_win_rate",
        "spread_edge", "total_edge",
        "is_dome", "high_wind", "cold_game",
    ]
    result = {}
    for feat in display_features:
        val = row.get(feat)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            result[feat] = round(float(val), 4) if isinstance(val, float) else val
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("predict.py — run via pipeline.py")

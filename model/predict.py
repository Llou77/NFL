"""
predict.py — predictions for every upcoming (unplayed) game of the season.

ensemble_predict() is the single prediction path: live predictions, the
backtest and the walk-forward all call it, so what is evaluated is exactly
what is published.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import settings

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Meta-learner input order — must match train._train_meta_learner.
META_ORDER = ("ridge", "xgb", "lgbm", "nn")


def ensemble_predict(models: dict, X_raw: np.ndarray) -> tuple:
    """Run the ensemble on a raw feature matrix (columns = models['feature_cols']).
    Returns (sub_home, sub_away, home, away); sub_* map model name → array."""
    from train import market_scores, nn_predict

    X_raw = np.asarray(X_raw, dtype=np.float64)
    X_sc = models["scaler"].transform(models["imputer"].transform(X_raw))
    sub_home = {
        "ridge": models["ridge_home"].predict(X_sc),
        "xgb":   models["xgb_home"].predict(X_raw),
        "lgbm":  models["lgbm_home"].predict(X_raw),
    }
    sub_away = {
        "ridge": models["ridge_away"].predict(X_sc),
        "xgb":   models["xgb_away"].predict(X_raw),
        "lgbm":  models["lgbm_away"].predict(X_raw),
    }
    sub_home["nn"], sub_away["nn"] = nn_predict(models["nn"], X_sc)
    S_h = np.column_stack([sub_home[m] for m in META_ORDER])
    S_a = np.column_stack([sub_away[m] for m in META_ORDER])
    home = models["meta_home"].predict(S_h)
    away = models["meta_away"].predict(S_a)
    # Games with a published line use the blender that also sees the market.
    mh, ma = market_scores(X_raw, models["feature_cols"])
    has = ~np.isnan(mh) & ~np.isnan(ma)
    if has.any():
        home[has] = models["meta_home_market"].predict(np.column_stack([S_h[has], mh[has]]))
        away[has] = models["meta_away_market"].predict(np.column_stack([S_a[has], ma[has]]))
    return sub_home, sub_away, home, away


def generate_predictions(game_df: pd.DataFrame, models: dict,
                         current_season: Optional[int] = None,
                         save: bool = True) -> pd.DataFrame:
    """Predict every game of `current_season` that has no result yet."""
    from confidence import compute_confidence_batch

    if current_season is None:
        from data_loader import CURRENT_SEASON as current_season

    upcoming = game_df[(game_df["season"] == current_season)
                       & game_df["target_home_score"].isna()].copy()
    if upcoming.empty:
        logger.info("No upcoming games found.")
        return upcoming
    logger.info("Generating predictions for %d upcoming games …", len(upcoming))

    cols = models["feature_cols"]
    missing = [c for c in cols if c not in upcoming.columns]
    if missing:
        raise RuntimeError(f"{len(missing)} model features are missing from the feature "
                           f"matrix (e.g. {missing[:5]}) — the saved model is stale; "
                           "retrain with --mode full.")
    sub_h, sub_a, home, away = ensemble_predict(models, upcoming[cols].to_numpy())

    upcoming["predicted_home_score"] = np.clip(np.round(home), 0, 70).astype(int)
    upcoming["predicted_away_score"] = np.clip(np.round(away), 0, 70).astype(int)
    upcoming["predicted_total"]  = upcoming["predicted_home_score"] + upcoming["predicted_away_score"]
    upcoming["predicted_spread"] = upcoming["predicted_home_score"] - upcoming["predicted_away_score"]

    # Range = spread of the four sub-models' predictions, widened to always
    # contain the published (blended) number.
    for side, sub, pred in (("home", sub_h, home), ("away", sub_a, away)):
        m = np.column_stack(list(sub.values()))
        upcoming[f"range_{side}_lo"] = np.floor(np.minimum(m.min(axis=1), pred)).astype(int)
        upcoming[f"range_{side}_hi"] = np.ceil(np.maximum(m.max(axis=1), pred)).astype(int)

    sub_by_game = {gid: {k: (float(sub_h[k][i]), float(sub_a[k][i])) for k in sub_h}
                   for i, gid in enumerate(upcoming["game_id"])}
    upcoming = compute_confidence_batch(upcoming, sub_by_game)
    upcoming = add_edge_signals(upcoming)

    if save:
        _save_predictions(upcoming, models.get("version", "unknown"))
    return upcoming


def add_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Model-minus-Vegas edges and the resulting lean labels.
    spread_line is home-oriented and POSITIVE when the home team is favoured,
    i.e. it is the book's expected home margin, so edge = model margin − line."""
    df["book_spread"] = df["spread_line"]
    df["book_total"]  = df["total_line"]
    df["spread_edge"] = df["predicted_spread"] - df["book_spread"]
    df["total_edge"]  = df["predicted_total"] - df["book_total"]
    df["spread_lean"] = np.select(
        [df["spread_edge"] >= settings.SPREAD_EDGE_PTS, df["spread_edge"] <= -settings.SPREAD_EDGE_PTS],
        ["HOME", "AWAY"], "NONE")
    df["total_lean"] = np.select(
        [df["total_edge"] >= settings.TOTAL_EDGE_PTS, df["total_edge"] <= -settings.TOTAL_EDGE_PTS],
        ["OVER", "UNDER"], "NONE")
    weak = df["confidence_label"].isin(["LOW", "WEAK"]) if "confidence_label" in df else False
    for col in ("spread_lean", "total_lean"):
        df.loc[weak & (df[col] != "NONE"), col] = "SUPPRESSED"
    return df


def _num(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), 2)


_TOP_FEATURES = [
    "elo_gap", "elo_vs_spread",
    "home_off_epa_per_play_r8", "away_off_epa_per_play_r8",
    "home_def_epa_per_play_r8", "away_def_epa_per_play_r8",
    "home_qb_epa_per_att_r8", "away_qb_epa_per_att_r8",
    "home_qb_start_share", "away_qb_start_share",
    "rest_diff", "h2h_avg_margin", "h2h_n_games",
]


def _save_predictions(df: pd.DataFrame, version: str) -> None:
    gen = datetime.now(timezone.utc).isoformat()
    records = []
    for _, r in df.iterrows():
        records.append({
            "game_id": r["game_id"], "season": int(r["season"]), "week": int(r["week"]),
            "game_type": r["game_type"], "game_date": str(r["game_date"])[:10],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "predicted_home_score": int(r["predicted_home_score"]),
            "predicted_away_score": int(r["predicted_away_score"]),
            "predicted_total": int(r["predicted_total"]),
            "predicted_spread": int(r["predicted_spread"]),
            "range_home": [int(r["range_home_lo"]), int(r["range_home_hi"])],
            "range_away": [int(r["range_away_lo"]), int(r["range_away_hi"])],
            "confidence_score": float(r["confidence_score"]),
            "confidence_label": r["confidence_label"],
            "confidence_breakdown": {
                "model_agreement":      float(r["conf_model_agreement"]),
                "feature_completeness": float(r["conf_feature_completeness"]),
                "h2h_data_quality":     float(r["conf_h2h_data_quality"]),
            },
            "book_spread": _num(r["book_spread"]), "book_total": _num(r["book_total"]),
            "spread_edge": _num(r["spread_edge"]), "total_edge": _num(r["total_edge"]),
            "spread_lean": r["spread_lean"], "total_lean": r["total_lean"],
            "is_dome": int(r["is_dome"]), "is_primetime": int(r["is_primetime"]),
            "is_international": int(r["is_international"]),
            "is_division_game": int(r["is_division_game"]),
            "top_features": {k: _num(r[k]) for k in _TOP_FEATURES
                             if k in r and _num(r[k]) is not None},
            "model_version": version,
            "generated_at": gen,
        })
    out = {"generated_at": gen, "model_version": version,
           "season": records[0]["season"] if records else None,
           "n_games": len(records), "predictions": records}
    path = PRED_DIR / "predictions_latest.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("  Saved %d predictions → %s", len(records), path)

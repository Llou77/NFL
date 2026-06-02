"""
evaluate.py
===========
Backtesting framework and ongoing performance tracking.

Backtest design:
  Train: 2022+2023+2024  → Test: 2025
  Train: 2021+2022+2023  → Test: 2024
  Train: 2020+2021+2022  → Test: 2023

ATS logic (corrected):
  A model ATS "WIN" means: the model picked the correct side of the spread.
  i.e. model predicted home margin M, book spread S (home perspective, neg = home favored):
    - Model pick: HOME if M > -S, AWAY if M < -S
    - Outcome: home covered if actual_margin > -S
    - WIN if model_pick == actual_cover_side

Metrics tracked:
  MAE (score, total, spread)
  ATS record (corrected definition)
  Over/Under record
  Calibration (R², predicted vs actual distribution)
  Performance by confidence tier, game_type, weather, division
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PERF_PATH = ROOT / "data" / "predictions" / "performance.json"


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtests(
    game_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> dict:
    from feature_engineering import get_feature_columns
    from train import train_all, load_models
    from train import _nn_predict

    feature_cols = get_feature_columns(game_df)

    backtest_configs = [
        {"train_seasons": [2022, 2023, 2024], "test_season": 2025},
        {"train_seasons": [2021, 2022, 2023], "test_season": 2024},
        {"train_seasons": [2020, 2021, 2022], "test_season": 2023},
    ]

    all_results = {}

    for config in backtest_configs:
        train_s = config["train_seasons"]
        test_s  = config["test_season"]

        df_train = game_df[
            game_df["season"].isin(train_s) &
            game_df["target_home_score"].notna()
        ].copy()
        df_test = game_df[
            (game_df["season"] == test_s) &
            game_df["target_home_score"].notna()
        ].copy()

        if len(df_train) < 50 or len(df_test) < 10:
            logger.warning("  Skipping backtest for %s — insufficient data", test_s)
            continue

        logger.info("  Backtest: train %s → test %s (%d / %d games)",
                    train_s, test_s, len(df_train), len(df_test))

        train_all(df_train, feature_cols, weights, current_season=test_s, run_cv=False)
        models = load_models()

        # Align features — fill missing cols with 0
        feature_cols_exist = [c for c in feature_cols if c in df_test.columns]
        X_test = df_test.reindex(columns=feature_cols, fill_value=np.nan).values
        X_imp  = models["imputer"].transform(X_test)
        X_sc   = models["scaler"].transform(X_imp)

        # Sub-model predictions
        ph = {"ridge": models["ridge_home"].predict(X_sc)}
        pa = {"ridge": models["ridge_away"].predict(X_sc)}
        if "xgb_home" in models:
            ph["xgb"] = models["xgb_home"].predict(X_sc)
            pa["xgb"] = models["xgb_away"].predict(X_sc)
        if "lgbm_home" in models:
            try:
                ph["lgbm"] = models["lgbm_home"].predict(X_sc)
                pa["lgbm"] = models["lgbm_away"].predict(X_sc)
            except Exception:
                pass
        if "nn" in models:
            nh, na = _nn_predict(models["nn"], X_sc)
            ph["nn"], pa["nn"] = nh, na

        meta_X_h = np.column_stack(list(ph.values()))
        meta_X_a = np.column_stack(list(pa.values()))

        pred_home = np.round(models["meta_home"].predict(meta_X_h)).astype(float)
        pred_away = np.round(models["meta_away"].predict(meta_X_a)).astype(float)

        actual_home = df_test["target_home_score"].values.astype(float)
        actual_away = df_test["target_away_score"].values.astype(float)

        result = compute_metrics(pred_home, pred_away, actual_home, actual_away, df_test)
        result["train_seasons"] = train_s
        result["test_season"]   = test_s
        all_results[str(test_s)] = result

        logger.info("    MAE home=%.2f away=%.2f total=%.2f | ATS=%.1f%% | OU=%.1f%%",
                    result["mae_home"], result["mae_away"], result["mae_total"],
                    result["ats_pct"] * 100, result["ou_pct"] * 100)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute all evaluation metrics.

    ATS definition (corrected):
      WIN  = model picked the CORRECT side of the Vegas spread
      LOSS = model picked the WRONG side
      PUSH = actual margin exactly equals spread (rare)

    This is the standard betting ATS definition.
    ~52% is break-even with -110 vig. >55% is meaningful edge.
    """
    pred_total    = pred_home  + pred_away
    pred_spread   = pred_home  - pred_away
    actual_total  = actual_home + actual_away
    actual_spread = actual_home - actual_away

    mae_home   = float(np.mean(np.abs(pred_home   - actual_home)))
    mae_away   = float(np.mean(np.abs(pred_away   - actual_away)))
    mae_total  = float(np.mean(np.abs(pred_total  - actual_total)))
    mae_spread = float(np.mean(np.abs(pred_spread - actual_spread)))

    # R² on total
    ss_res   = float(np.sum((pred_total - actual_total) ** 2))
    ss_tot   = float(np.sum((actual_total - np.mean(actual_total)) ** 2))
    r2_total = 1.0 - ss_res / max(ss_tot, 1e-9)

    # Variance of predictions vs. actuals (measures regression-to-mean)
    pred_spread_std  = float(np.std(pred_spread))
    actual_spread_std = float(np.std(actual_spread))

    # ── ATS (corrected logic using vectorized pandas) ──────────────────────
    ats_wins = ats_losses = ats_pushes = 0

    if df is not None and "spread_line" in df.columns:
        df_aligned = df.reset_index(drop=True)
        book_spread = pd.to_numeric(df_aligned["spread_line"], errors="coerce")
        valid = book_spread.notna()

        if valid.any():
            bs  = book_spread[valid].values   # book spread (home perspective)
            ph  = pred_home[valid]
            pa  = pred_away[valid]
            ah  = actual_home[valid]
            aa  = actual_away[valid]

            actual_margin = ah - aa
            model_margin  = ph - pa

            # threshold: home covers if actual_margin > -book_spread
            threshold = -bs

            # Model pick: which side does the model favor?
            model_pick_home = model_margin > threshold     # True = model takes home
            model_pick_away = model_margin < threshold     # True = model takes away

            # Actual outcome
            home_covered  = actual_margin > threshold
            away_covered  = actual_margin < threshold
            push          = actual_margin == threshold

            # ATS result
            ats_win_arr  = ((model_pick_home & home_covered) |
                            (model_pick_away & away_covered))
            ats_loss_arr = ((model_pick_home & away_covered) |
                            (model_pick_away & home_covered))
            ats_push_arr = push

            ats_wins   = int(ats_win_arr.sum())
            ats_losses = int(ats_loss_arr.sum())
            ats_pushes = int(ats_push_arr.sum())

    ats_total = ats_wins + ats_losses + ats_pushes
    ats_pct   = ats_wins / max(ats_wins + ats_losses, 1)

    # ── Over/Under ─────────────────────────────────────────────────────────
    ou_wins = ou_losses = ou_pushes = 0

    if df is not None and "total_line" in df.columns:
        df_aligned = df.reset_index(drop=True)
        book_total = pd.to_numeric(df_aligned["total_line"], errors="coerce")
        valid = book_total.notna()

        if valid.any():
            bt  = book_total[valid].values
            pt  = pred_total[valid]
            at  = actual_total[valid]

            model_over  = pt > bt
            actual_over = at > bt
            actual_under = at < bt
            actual_push = at == bt

            ou_win_arr  = ((model_over & actual_over) | (~model_over & actual_under))
            ou_loss_arr = ((model_over & actual_under) | (~model_over & actual_over))

            ou_wins   = int((ou_win_arr  & ~actual_push).sum())
            ou_losses = int((ou_loss_arr & ~actual_push).sum())
            ou_pushes = int(actual_push.sum())

    ou_pct = ou_wins / max(ou_wins + ou_losses, 1)

    result = {
        "mae_home":          round(mae_home,   3),
        "mae_away":          round(mae_away,   3),
        "mae_total":         round(mae_total,  3),
        "mae_spread":        round(mae_spread, 3),
        "ats_wins":          ats_wins,
        "ats_losses":        ats_losses,
        "ats_pushes":        ats_pushes,
        "ats_pct":           round(ats_pct,    4),
        "ou_wins":           ou_wins,
        "ou_losses":         ou_losses,
        "ou_pushes":         ou_pushes,
        "ou_pct":            round(ou_pct,     4),
        "r2_total":          round(r2_total,   4),
        "pred_spread_std":   round(pred_spread_std,   2),
        "actual_spread_std": round(actual_spread_std, 2),
        "n_games":           len(pred_home),
    }

    if df is not None:
        for col in ["game_type", "is_division_game", "is_dome", "is_primetime"]:
            result[f"by_{col.replace('is_','')}"] = _breakdown_by(
                pred_home, pred_away, actual_home, actual_away, df, col
            )

    return result


def _breakdown_by(ph, pa, ah, aa, df, col):
    if col not in df.columns:
        return {}
    df2 = df.reset_index(drop=True)
    result = {}
    for val in df2[col].dropna().unique():
        mask = (df2[col] == val).values
        if mask.sum() < 5:
            continue
        mae = float(np.mean(np.abs(ph[mask] + pa[mask] - ah[mask] - aa[mask])))
        result[str(val)] = {"mae_total": round(mae, 3), "n": int(mask.sum())}
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE SEASON PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def update_season_performance(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> dict:
    if PERF_PATH.exists():
        with open(PERF_PATH) as f:
            perf = json.load(f)
    else:
        perf = {"games": [], "summary": {}}

    merged = predictions_df.merge(
        results_df[["game_id", "home_score", "away_score"]],
        on="game_id", how="inner",
    )

    existing_ids = {g["game_id"] for g in perf["games"]}

    for _, row in merged.iterrows():
        pred_h = int(row.get("predicted_home_score", 0))
        pred_a = int(row.get("predicted_away_score", 0))
        act_h  = int(row.get("home_score", 0))
        act_a  = int(row.get("away_score", 0))

        book_spread = row.get("book_spread")
        book_total  = row.get("book_total")

        # ATS result (corrected)
        ats_result = None
        if book_spread is not None and not (isinstance(book_spread, float) and np.isnan(book_spread)):
            threshold    = -float(book_spread)
            actual_margin = act_h - act_a
            model_margin  = pred_h - pred_a
            if actual_margin == threshold:
                ats_result = "PUSH"
            elif ((model_margin > threshold) == (actual_margin > threshold)):
                ats_result = "WIN"
            else:
                ats_result = "LOSS"

        # OU result
        ou_result = None
        total_lean = row.get("total_lean", "PUSH")
        if book_total is not None and not (isinstance(book_total, float) and np.isnan(book_total)) \
                and total_lean not in ("SUPPRESSED", "PUSH"):
            actual_total = act_h + act_a
            if actual_total == float(book_total):
                ou_result = "PUSH"
            elif (actual_total > float(book_total) and total_lean == "OVER") or \
                 (actual_total < float(book_total) and total_lean == "UNDER"):
                ou_result = "WIN"
            else:
                ou_result = "LOSS"

        record = {
            "game_id":          str(row["game_id"]),
            "week":             int(row.get("week", 0)),
            "game_type":        str(row.get("game_type", "REG")),
            "home_team":        str(row.get("home_team", "")),
            "away_team":        str(row.get("away_team", "")),
            "predicted_home":   pred_h,
            "predicted_away":   pred_a,
            "actual_home":      act_h,
            "actual_away":      act_a,
            "error_home":       pred_h - act_h,
            "error_away":       pred_a - act_a,
            "error_total":      (pred_h + pred_a) - (act_h + act_a),
            "error_spread":     (pred_h - pred_a) - (act_h - act_a),
            "confidence_label": str(row.get("confidence_label", "LOW")),
            "confidence_score": float(row.get("confidence_score", 0.5)),
            "spread_lean":      str(row.get("spread_lean", "PUSH")),
            "total_lean":       total_lean,
            "book_spread":      book_spread,
            "book_total":       book_total,
            "ats_result":       ats_result,
            "ou_result":        ou_result,
        }

        if record["game_id"] in existing_ids:
            perf["games"] = [g if g["game_id"] != record["game_id"] else record
                             for g in perf["games"]]
        else:
            perf["games"].append(record)

    perf["summary"]    = _compute_summary(perf["games"])
    perf["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    with open(PERF_PATH, "w") as f:
        json.dump(perf, f, indent=2)

    logger.info("Performance updated: %d games tracked", len(perf["games"]))
    return perf


def _compute_summary(games: list) -> dict:
    if not games:
        return {}

    errors_t = [g["error_total"]  for g in games]
    errors_s = [g["error_spread"] for g in games]
    errors_h = [g["error_home"]   for g in games]
    errors_a = [g["error_away"]   for g in games]

    ats_games = [g for g in games if g.get("ats_result") in ("WIN", "LOSS")]
    ats_wins  = sum(1 for g in ats_games if g["ats_result"] == "WIN")
    ou_games  = [g for g in games if g.get("ou_result") in ("WIN", "LOSS")]
    ou_wins   = sum(1 for g in ou_games  if g["ou_result"]  == "WIN")

    conf_breakdown = {}
    for label in ("HIGH", "MEDIUM", "LOW", "WEAK"):
        tier = [g for g in games if g.get("confidence_label") == label]
        if tier:
            t_err = [g["error_total"] for g in tier]
            t_ats = [g for g in tier if g.get("ats_result") in ("WIN","LOSS")]
            conf_breakdown[label] = {
                "n":         len(tier),
                "mae_total": round(float(np.mean(np.abs(t_err))), 3),
                "ats_pct":   round(sum(1 for g in t_ats if g["ats_result"]=="WIN") / max(len(t_ats),1), 4),
            }

    return {
        "n_games":    len(games),
        "mae_home":   round(float(np.mean(np.abs(errors_h))), 3),
        "mae_away":   round(float(np.mean(np.abs(errors_a))), 3),
        "mae_total":  round(float(np.mean(np.abs(errors_t))), 3),
        "mae_spread": round(float(np.mean(np.abs(errors_s))), 3),
        "ats_record": f"{ats_wins}-{len(ats_games)-ats_wins}",
        "ats_pct":    round(ats_wins / max(len(ats_games), 1), 4),
        "ou_record":  f"{ou_wins}-{len(ou_games)-ou_wins}",
        "ou_pct":     round(ou_wins  / max(len(ou_games),  1), 4),
        "confidence_breakdown": conf_breakdown,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("evaluate.py — run via pipeline.py --mode backtest")

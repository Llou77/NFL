"""
evaluate.py
===========
Backtesting framework and ongoing performance tracking.

Backtest design:
  Train: 2022+2023+2024  → Test: 2025
  Train: 2021+2022+2023  → Test: 2024
  Train: 2020+2021+2022  → Test: 2023

Metrics tracked:
  MAE (score, total, spread)
  ATS record (wins, losses, pushes)
  Over/Under record
  Performance by confidence tier
  Performance by game_type, weather, division, home/away
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
PERF_PATH  = ROOT / "data" / "predictions" / "performance.json"


def run_backtests(
    game_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> dict:
    """
    Run 3-season rolling backtests.
    Returns dict of backtest metrics per test season.
    """
    from feature_engineering import get_feature_columns
    from bayesian_optimizer import compute_sample_weights
    from train import train_all
    from train import _nn_predict
    import pickle
    from pathlib import Path

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

        # Filter data
        df_train = game_df[
            game_df["season"].isin(train_s) &
            game_df["target_home_score"].notna()
        ].copy()
        df_test = game_df[
            (game_df["season"] == test_s) &
            game_df["target_home_score"].notna()
        ].copy()

        if len(df_train) < 50 or len(df_test) < 10:
            logger.warning(f"  Skipping backtest for {test_s} — insufficient data")
            continue

        logger.info(f"  Backtest: train {train_s} → test {test_s} "
                    f"({len(df_train)} train / {len(df_test)} test games)")

        # Train
        metrics = train_all(
            df_train, feature_cols, weights,
            current_season=test_s, run_cv=False
        )

        # Load and predict
        from train import load_models
        models = load_models()

        X_test  = df_test[feature_cols].values
        X_imp   = models["imputer"].transform(X_test)
        X_sc    = models["scaler"].transform(X_imp)

        # Sub-model predictions
        preds_home = {"ridge": models["ridge_home"].predict(X_sc)}
        preds_away = {"ridge": models["ridge_away"].predict(X_sc)}

        if "xgb_home" in models:
            preds_home["xgb"] = models["xgb_home"].predict(X_sc)
            preds_away["xgb"] = models["xgb_away"].predict(X_sc)
        if "lgbm_home" in models:
            try:
                preds_home["lgbm"] = models["lgbm_home"].predict(X_sc)
                preds_away["lgbm"] = models["lgbm_away"].predict(X_sc)
            except Exception:
                pass
        if "nn" in models:
            nh, na = _nn_predict(models["nn"], X_sc)
            preds_home["nn"] = nh
            preds_away["nn"] = na

        meta_X_h = np.column_stack(list(preds_home.values()))
        meta_X_a = np.column_stack(list(preds_away.values()))

        pred_home = np.round(models["meta_home"].predict(meta_X_h)).astype(int)
        pred_away = np.round(models["meta_away"].predict(meta_X_a)).astype(int)

        actual_home = df_test["target_home_score"].values
        actual_away = df_test["target_away_score"].values

        result = compute_metrics(
            pred_home, pred_away, actual_home, actual_away, df_test
        )
        result["train_seasons"] = train_s
        result["test_season"]   = test_s
        all_results[str(test_s)] = result

        logger.info(f"    MAE home: {result['mae_home']:.2f} | "
                    f"away: {result['mae_away']:.2f} | "
                    f"total: {result['mae_total']:.2f} | "
                    f"ATS: {result['ats_pct']:.1%} | "
                    f"OU: {result['ou_pct']:.1%}")

    return all_results


def compute_metrics(
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute all evaluation metrics for a set of predictions."""

    pred_total  = pred_home  + pred_away
    pred_spread = pred_home  - pred_away
    actual_total  = actual_home + actual_away
    actual_spread = actual_home - actual_away

    # ── Core MAE ──────────────────────────────────────────────────────────
    mae_home   = float(np.mean(np.abs(pred_home  - actual_home)))
    mae_away   = float(np.mean(np.abs(pred_away  - actual_away)))
    mae_total  = float(np.mean(np.abs(pred_total  - actual_total)))
    mae_spread = float(np.mean(np.abs(pred_spread - actual_spread)))

    # ── ATS performance ──────────────────────────────────────────────────
    # Model predicts home margin; actual margin vs. book spread
    ats_wins = ats_losses = ats_pushes = 0
    if df is not None and "spread_line" in df.columns:
        for i, (ph, pa, ah, aa) in enumerate(
            zip(pred_home, pred_away, actual_home, actual_away)
        ):
            book_spread = df.iloc[i].get("spread_line", np.nan)
            if np.isnan(book_spread):
                continue
            actual_margin = ah - aa
            # Home covered if actual_margin > -book_spread
            # (book_spread negative means home favored, e.g. -6.5)
            if actual_margin > -book_spread:
                # Model said home wins by more than spread? → win
                if (ph - pa) > -book_spread:
                    ats_wins += 1
                else:
                    ats_losses += 1
            elif actual_margin < -book_spread:
                if (ph - pa) < -book_spread:
                    ats_wins += 1
                else:
                    ats_losses += 1
            else:
                ats_pushes += 1

    ats_total = ats_wins + ats_losses + ats_pushes
    ats_pct   = ats_wins / max(ats_total - ats_pushes, 1)

    # ── Over/Under performance ────────────────────────────────────────────
    ou_wins = ou_losses = ou_pushes = 0
    if df is not None and "total_line" in df.columns:
        for i, (pt, at) in enumerate(zip(pred_total, actual_total)):
            book_total = df.iloc[i].get("total_line", np.nan)
            if np.isnan(book_total):
                continue
            model_lean = "OVER" if pt > book_total else "UNDER"
            if actual_total[i] > book_total:
                if model_lean == "OVER":
                    ou_wins += 1
                else:
                    ou_losses += 1
            elif actual_total[i] < book_total:
                if model_lean == "UNDER":
                    ou_wins += 1
                else:
                    ou_losses += 1
            else:
                ou_pushes += 1

    ou_total = ou_wins + ou_losses + ou_pushes
    ou_pct   = ou_wins / max(ou_total - ou_pushes, 1)

    # ── Calibration ───────────────────────────────────────────────────────
    # R² of predicted vs actual totals
    ss_res = np.sum((pred_total - actual_total) ** 2)
    ss_tot = np.sum((actual_total - actual_total.mean()) ** 2)
    r2_total = 1.0 - ss_res / max(ss_tot, 1e-9)

    result = {
        "mae_home":   round(mae_home,   3),
        "mae_away":   round(mae_away,   3),
        "mae_total":  round(mae_total,  3),
        "mae_spread": round(mae_spread, 3),
        "ats_wins":   ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_pct":    round(ats_pct,    4),
        "ou_wins":    ou_wins,
        "ou_losses":  ou_losses,
        "ou_pushes":  ou_pushes,
        "ou_pct":     round(ou_pct,     4),
        "r2_total":   round(r2_total,   4),
        "n_games":    len(pred_home),
    }

    # ── Breakdowns ─────────────────────────────────────────────────────────
    if df is not None:
        result["by_game_type"] = _breakdown_by(
            pred_home, pred_away, actual_home, actual_away, df, "game_type"
        )
        result["by_division"]  = _breakdown_by(
            pred_home, pred_away, actual_home, actual_away, df, "is_division_game"
        )
        result["by_dome"]      = _breakdown_by(
            pred_home, pred_away, actual_home, actual_away, df, "is_dome"
        )
        result["by_primetime"] = _breakdown_by(
            pred_home, pred_away, actual_home, actual_away, df, "is_primetime"
        )

    return result


def _breakdown_by(ph, pa, ah, aa, df, col):
    if col not in df.columns:
        return {}
    result = {}
    for val in df[col].unique():
        mask = (df[col] == val).values
        if mask.sum() < 5:
            continue
        mae = float(np.mean(np.abs(ph[mask] + pa[mask] - ah[mask] - aa[mask])))
        result[str(val)] = {"mae_total": round(mae, 3), "n": int(mask.sum())}
    return result


def update_season_performance(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> dict:
    """
    Compare predictions against actual results for completed games.
    Returns updated performance dict saved to data/predictions/performance.json.
    """
    # Load existing performance
    if PERF_PATH.exists():
        with open(PERF_PATH) as f:
            perf = json.load(f)
    else:
        perf = {"games": [], "summary": {}}

    # Match predictions to results
    merged = predictions_df.merge(
        results_df[["game_id", "home_score", "away_score"]],
        on="game_id", how="inner",
    )

    for _, row in merged.iterrows():
        game_record = {
            "game_id":              row["game_id"],
            "week":                 int(row.get("week", 0)),
            "game_type":            row.get("game_type", "REG"),
            "home_team":            row.get("home_team", ""),
            "away_team":            row.get("away_team", ""),
            "predicted_home":       int(row.get("predicted_home_score", 0)),
            "predicted_away":       int(row.get("predicted_away_score", 0)),
            "actual_home":          int(row.get("home_score", 0)),
            "actual_away":          int(row.get("away_score", 0)),
            "confidence_label":     row.get("confidence_label", "LOW"),
            "confidence_score":     float(row.get("confidence_score", 0.5)),
            "spread_lean":          row.get("spread_lean", "PUSH"),
            "total_lean":           row.get("total_lean", "PUSH"),
            "book_spread":          row.get("book_spread"),
            "book_total":           row.get("book_total"),
        }
        # Compute errors
        pred_h = game_record["predicted_home"]
        pred_a = game_record["predicted_away"]
        act_h  = game_record["actual_home"]
        act_a  = game_record["actual_away"]
        game_record["error_home"]   = pred_h - act_h
        game_record["error_away"]   = pred_a - act_a
        game_record["error_total"]  = (pred_h + pred_a) - (act_h + act_a)
        game_record["error_spread"] = (pred_h - pred_a) - (act_h - act_a)

        # ATS result
        book_spread = game_record.get("book_spread")
        if book_spread is not None:
            actual_margin = act_h - act_a
            model_pick_home = (pred_h - pred_a) > -book_spread
            home_covered = actual_margin > -book_spread
            if (pred_h - pred_a) - (-book_spread) == 0:
                game_record["ats_result"] = "PUSH"
            elif model_pick_home == home_covered:
                game_record["ats_result"] = "WIN"
            else:
                game_record["ats_result"] = "LOSS"
        else:
            game_record["ats_result"] = None

        # OU result
        book_total = game_record.get("book_total")
        if book_total is not None:
            actual_total_score = act_h + act_a
            model_lean = game_record["total_lean"]
            if actual_total_score == book_total:
                game_record["ou_result"] = "PUSH"
            elif (actual_total_score > book_total and model_lean == "OVER") or \
                 (actual_total_score < book_total and model_lean == "UNDER"):
                game_record["ou_result"] = "WIN"
            elif model_lean == "SUPPRESSED":
                game_record["ou_result"] = None
            else:
                game_record["ou_result"] = "LOSS"
        else:
            game_record["ou_result"] = None

        # Upsert
        existing = [g for g in perf["games"] if g["game_id"] == row["game_id"]]
        if existing:
            perf["games"][perf["games"].index(existing[0])] = game_record
        else:
            perf["games"].append(game_record)

    # Recompute summary
    perf["summary"] = _compute_summary(perf["games"])
    perf["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    with open(PERF_PATH, "w") as f:
        json.dump(perf, f, indent=2)

    logger.info(f"Performance updated: {len(perf['games'])} games tracked")
    return perf


def _compute_summary(games: list) -> dict:
    if not games:
        return {}

    errors_h = [g["error_home"]   for g in games]
    errors_a = [g["error_away"]   for g in games]
    errors_t = [g["error_total"]  for g in games]
    errors_s = [g["error_spread"] for g in games]

    ats_games  = [g for g in games if g.get("ats_result") in ("WIN", "LOSS")]
    ats_wins   = sum(1 for g in ats_games if g["ats_result"] == "WIN")
    ou_games   = [g for g in games if g.get("ou_result") in ("WIN", "LOSS")]
    ou_wins    = sum(1 for g in ou_games if g["ou_result"] == "WIN")

    # Confidence tier breakdown
    conf_breakdown = {}
    for label in ("HIGH", "MEDIUM", "LOW", "WEAK"):
        tier = [g for g in games if g.get("confidence_label") == label]
        if tier:
            t_err = [g["error_total"] for g in tier]
            conf_breakdown[label] = {
                "n": len(tier),
                "mae_total": round(float(np.mean(np.abs(t_err))), 3),
            }

    return {
        "n_games":        len(games),
        "mae_home":       round(float(np.mean(np.abs(errors_h))), 3),
        "mae_away":       round(float(np.mean(np.abs(errors_a))), 3),
        "mae_total":      round(float(np.mean(np.abs(errors_t))), 3),
        "mae_spread":     round(float(np.mean(np.abs(errors_s))), 3),
        "ats_record":     f"{ats_wins}-{len(ats_games)-ats_wins}",
        "ats_pct":        round(ats_wins / max(len(ats_games), 1), 4),
        "ou_record":      f"{ou_wins}-{len(ou_games)-ou_wins}",
        "ou_pct":         round(ou_wins / max(len(ou_games), 1), 4),
        "confidence_breakdown": conf_breakdown,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("evaluate.py — run backtests via pipeline.py --mode backtest")

"""
pipeline.py — NFL Score Predictor: Main Orchestration Script

Modes:
  full          — fetch data → engineer features → train → predict
  predict_only  — skip training, regenerate predictions from saved model
  backtest      — run historical backtesting (requires extended season data)
  optimize      — Bayesian weight optimisation (run pre-season)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure data dir exists before setting up log handler
(ROOT / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "model" / "saved").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "predictions" / "archive").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "model"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "data" / "pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# Single source of truth for the season — computed from the clock in
# data_loader (no more hand-edited season constants scattered across files).
from data_loader import CURRENT_SEASON

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "model" / "saved"
PRED_DIR  = DATA_DIR / "predictions"


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: build feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def _build_features(force_data: bool = False):
    """Fetch data + engineer features. Returns game_df."""
    log.info("Fetching NFL data …")
    from data_loader import load_all
    load_all(force_refresh=force_data)

    log.info("Engineering features …")
    from feature_engineering import build_all_features
    game_df = build_all_features()

    log.info("Building H2H features …")
    from feature_h2h import build_h2h_features
    game_df = build_h2h_features(game_df)

    return game_df


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: FULL
# ══════════════════════════════════════════════════════════════════════════════

def _train_models(game_df):
    """Train the full ensemble and persist artifacts. Shared by full mode and
    the predict_only fallback."""
    log.info("Loading season weights …")
    from bayesian_optimizer import load_weights
    weights = load_weights()
    log.info("  Weights: %s", weights)

    log.info("Training ensemble …")
    from feature_engineering import get_feature_columns
    from train import train_all
    feature_cols = get_feature_columns(game_df)
    metrics = train_all(game_df, feature_cols, weights, CURRENT_SEASON)
    log.info("  Training metrics: %s", metrics)
    return metrics


def _reconcile_performance(game_df):
    """Match the previously published predictions against games completed
    since, and refresh performance.json — BEFORE the file is overwritten."""
    try:
        from evaluate import update_performance_from_latest
        update_performance_from_latest(game_df)
    except Exception as e:
        log.warning("Performance reconciliation failed (non-fatal): %s", e)


def run_full(args):
    log.info("=" * 60)
    log.info("FULL PIPELINE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 60)

    game_df = _build_features(force_data=getattr(args, "force_data", False))

    _reconcile_performance(game_df)
    _train_models(game_df)

    log.info("Generating predictions …")
    from train import load_models
    from predict import generate_predictions
    models   = load_models()
    preds_df = generate_predictions(game_df, models, CURRENT_SEASON, save=True)
    log.info("  Generated %d predictions", len(preds_df))

    _copy_to_docs()
    _write_status("success")
    log.info("FULL PIPELINE COMPLETE")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: PREDICT ONLY
# ══════════════════════════════════════════════════════════════════════════════

def run_predict_only(args):
    log.info("PREDICT-ONLY MODE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    game_df = _build_features(force_data=False)

    _reconcile_performance(game_df)

    from train import load_models
    from predict import generate_predictions
    try:
        models = load_models()
    except FileNotFoundError as e:
        # CRASH FIX: model artifacts are gitignored, so a clean CI checkout
        # has none — this previously killed the Thursday workflow every week.
        # Fall back to training (the workflow-level cache makes this rare).
        log.warning("Saved model artifacts unavailable (%s) — "
                    "falling back to full training.", e)
        _train_models(game_df)
        models = load_models()

    preds_df = generate_predictions(game_df, models, CURRENT_SEASON, save=True)
    log.info("  Generated %d predictions", len(preds_df))

    _copy_to_docs()
    _write_status("success")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: OPTIMIZE
# ══════════════════════════════════════════════════════════════════════════════

def run_optimize(args):
    log.info("BAYESIAN OPTIMISATION MODE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    game_df = _build_features(force_data=True)

    from feature_engineering import get_feature_columns
    feature_cols = get_feature_columns(game_df)

    labeled = game_df[game_df["target_home_score"].notna()].copy()
    all_seasons = sorted(labeled["season"].unique())

    if len(all_seasons) < 2:
        log.warning("Not enough seasons for optimisation — using default weights.")
        from bayesian_optimizer import DEFAULT_WEIGHTS, save_weights
        save_weights(DEFAULT_WEIGHTS)
        _write_status("success")
        return

    val_season    = all_seasons[-1]
    train_seasons = all_seasons[:-1]
    df_train = labeled[labeled["season"].isin(train_seasons)].copy()
    df_val   = labeled[labeled["season"] == val_season].copy()

    log.info("  Train seasons: %s  (n=%d)", train_seasons, len(df_train))
    log.info("  Val season:    %s  (n=%d)", val_season, len(df_val))

    if len(df_val) < 20:
        log.warning("Validation set too small (%d games) — using default weights.", len(df_val))
        from bayesian_optimizer import DEFAULT_WEIGHTS, save_weights
        save_weights(DEFAULT_WEIGHTS)
        _write_status("success")
        return

    from bayesian_optimizer import run_bayesian_optimization, save_weights
    best = run_bayesian_optimization(df_train, df_val, feature_cols, n_trials=60)
    save_weights(best)
    log.info("Optimal weights saved: %s", best)
    _write_status("success")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(args):
    log.info("BACKTEST MODE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    game_df = _build_features(force_data=False)

    from bayesian_optimizer import load_weights
    from evaluate import run_backtests
    weights = load_weights()
    results = run_backtests(game_df, weights)

    out = PRED_DIR / "backtest_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Backtest results → %s", out)

    for season, r in results.items():
        log.info("  Season %s: MAE total=%.2f | ATS=%.1f%% | OU=%.1f%%",
                 season,
                 r.get("mae_total", 0),
                 r.get("ats_pct", 0) * 100,
                 r.get("ou_pct", 0) * 100)

    _copy_to_docs()
    _write_status("success")


def run_analyze_seasons(args):
    log.info("PER-SEASON ANALYSIS MODE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    game_df = _build_features(force_data=False)

    from season_analysis import run_season_analysis
    results = run_season_analysis(game_df)

    log.info("Season analysis complete: %d seasons analysed", results.get("n_seasons", 0))
    trends = results.get("trends", {})
    interp = trends.get("interpretation", {})
    log.info("  Total scoring: %s | Home advantage: %s | Model difficulty: %s",
             interp.get("total_scoring", "?"),
             interp.get("home_advantage", "?"),
             interp.get("model_difficulty", "?"))

    _copy_to_docs()
    _write_status("success")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _copy_to_docs():
    """Copy prediction outputs to docs/assets/ for GitHub Pages."""
    docs_assets = ROOT / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    for fname in ["predictions_latest.json", "performance.json",
                  "backtest_results.json", "season_analysis.json"]:
        src = PRED_DIR / fname
        if src.exists():
            import shutil
            shutil.copy2(src, docs_assets / fname)
            log.info("  Copied %s → docs/assets/", fname)


def _write_status(status: str):
    with open(DATA_DIR / "pipeline_status.json", "w") as f:
        json.dump({"status": status,
                   "timestamp": datetime.now(timezone.utc).isoformat()}, f)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NFL Score Predictor Pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "predict_only", "backtest", "optimize", "analyze_seasons"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        default=False,
        help="Force re-download all data even if cached",
    )
    args = parser.parse_args()
    # Make force_data accessible as attribute
    args.force_data = args.force_data

    dispatch = {
        "full":             run_full,
        "predict_only":     run_predict_only,
        "backtest":         run_backtest,
        "optimize":         run_optimize,
        "analyze_seasons":  run_analyze_seasons,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

"""
pipeline.py
===========
Main orchestration script. Called by GitHub Actions and manual runs.

Modes:
  --mode full          fetch → features → h2h → train → predict
  --mode predict_only  skip training, regenerate predictions only
  --mode backtest      run full historical backtesting validation
  --mode optimize      run Bayesian weight optimisation (pre-season)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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

CURRENT_SEASON = 2026
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "model" / "saved"
PRED_DIR  = DATA_DIR / "predictions"
for d in [DATA_DIR, MODEL_DIR, PRED_DIR, PRED_DIR / "archive"]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: FULL
# ══════════════════════════════════════════════════════════════════════════════

def run_full(args):
    log.info("=" * 60)
    log.info("FULL PIPELINE START  —  %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 60)

    # 1. Data
    log.info("[1/6] Fetching NFL data …")
    from data_loader import load_all
    load_all(force_refresh=False)

    # 2. Features
    log.info("[2/6] Engineering features …")
    from feature_engineering import build_all_features, get_feature_columns
    game_df = build_all_features()

    # 3. H2H features
    log.info("[3/6] Building head-to-head features …")
    from feature_h2h import build_h2h_features
    game_df = build_h2h_features(game_df)

    # 4. Load weights
    log.info("[4/6] Loading season weights …")
    from bayesian_optimizer import load_weights
    weights = load_weights()
    log.info("  Weights: %s", weights)

    # 5. Train
    log.info("[5/6] Training ensemble …")
    from train import train_all
    feature_cols = get_feature_columns(game_df)
    metrics = train_all(game_df, feature_cols, weights, CURRENT_SEASON)
    log.info("  Training metrics: %s", metrics)

    # 6. Predict
    log.info("[6/6] Generating predictions …")
    from train import load_models
    from predict import generate_predictions
    models = load_models()
    preds_df = generate_predictions(game_df, models, CURRENT_SEASON, save=True)
    log.info("  Generated %d predictions", len(preds_df))

    _write_status("success")
    log.info("FULL PIPELINE COMPLETE")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: PREDICT ONLY
# ══════════════════════════════════════════════════════════════════════════════

def run_predict_only(args):
    log.info("PREDICT-ONLY MODE")
    from data_loader import load_all
    load_all(force_refresh=False)

    from feature_engineering import build_all_features, get_feature_columns
    game_df = build_all_features()

    from feature_h2h import build_h2h_features
    game_df = build_h2h_features(game_df)

    from train import load_models
    from predict import generate_predictions
    models = load_models()
    preds_df = generate_predictions(game_df, models, CURRENT_SEASON, save=True)
    log.info("  Generated %d predictions", len(preds_df))
    _write_status("success")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(args):
    log.info("BACKTEST MODE")
    from data_loader import load_all
    load_all(force_refresh=False)

    from feature_engineering import build_all_features
    game_df = build_all_features()

    from feature_h2h import build_h2h_features
    game_df = build_h2h_features(game_df)

    from bayesian_optimizer import load_weights
    weights = load_weights()

    from evaluate import run_backtests
    results = run_backtests(game_df, weights)

    out = DATA_DIR / "predictions" / "backtest_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Backtest results saved → %s", out)

    for season, r in results.items():
        log.info("  Season %s: MAE total=%.2f | ATS=%.1f%% | OU=%.1f%%",
                 season, r.get("mae_total", 0),
                 r.get("ats_pct", 0) * 100,
                 r.get("ou_pct", 0) * 100)
    _write_status("success")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE: OPTIMIZE
# ══════════════════════════════════════════════════════════════════════════════

def run_optimize(args):
    log.info("BAYESIAN OPTIMISATION MODE")
    from data_loader import load_all
    load_all(force_refresh=False)

    from feature_engineering import build_all_features, get_feature_columns
    game_df = build_all_features()

    from feature_h2h import build_h2h_features
    game_df = build_h2h_features(game_df)

    feature_cols = get_feature_columns(game_df)

    # Use last available season as validation, rest as train
    all_seasons = sorted(game_df["season"].unique())
    val_season  = all_seasons[-1]
    train_seasons = all_seasons[:-1]

    df_train = game_df[game_df["season"].isin(train_seasons) & game_df["target_home_score"].notna()].copy()
    df_val   = game_df[(game_df["season"] == val_season) & game_df["target_home_score"].notna()].copy()

    log.info("  Train seasons: %s  (n=%d)", train_seasons, len(df_train))
    log.info("  Val season:    %s  (n=%d)", val_season, len(df_val))

    from bayesian_optimizer import run_bayesian_optimization, save_weights
    best = run_bayesian_optimization(df_train, df_val, feature_cols, n_trials=60)
    save_weights(best)

    log.info("Optimal weights: %s", best)
    _write_status("success")


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_status(status: str):
    p = DATA_DIR / "pipeline_status.json"
    with open(p, "w") as f:
        json.dump({"status": status, "timestamp": datetime.utcnow().isoformat() + "Z"}, f)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NFL Score Predictor Pipeline")
    parser.add_argument("--mode",
                        choices=["full", "predict_only", "backtest", "optimize"],
                        default="full")
    args = parser.parse_args()
    {"full": run_full, "predict_only": run_predict_only,
     "backtest": run_backtest, "optimize": run_optimize}[args.mode](args)


if __name__ == "__main__":
    main()

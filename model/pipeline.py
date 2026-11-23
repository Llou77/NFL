"""
pipeline.py — entry point.

  --mode full          fetch data → features → reconcile the live record →
                       train → predict → publish to docs/assets/
  --mode predict_only  same, but reuse the saved model (trains if missing)
  --mode backtest      train on the 4 seasons before --backtest-season and
                       score that season (writes a fragment for the CI merge)
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
(ROOT / "data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(ROOT / "data" / "pipeline.log", mode="a")],
)
log = logging.getLogger("pipeline")

from data_loader import CURRENT_SEASON  # noqa: E402

PRED_DIR = ROOT / "data" / "predictions"
PUBLISHED = ["predictions_latest.json", "performance.json", "backtest_results.json"]


def _build_features(force_data: bool = False, seasons=None):
    from data_loader import load_all
    from feature_engineering import build_all_features
    from feature_h2h import build_h2h_features
    log.info("Fetching NFL data …")
    load_all(force_refresh=force_data, seasons=seasons)
    log.info("Engineering features …")
    return build_h2h_features(build_all_features(seasons=seasons))


def _train(game_df):
    from feature_engineering import get_feature_columns
    from train import train_all
    log.info("Training ensemble …")
    return train_all(game_df, get_feature_columns(game_df), current_season=CURRENT_SEASON)


def _reconcile(game_df):
    """Score the previously published predictions BEFORE they are replaced."""
    from evaluate import update_performance_from_latest
    try:
        update_performance_from_latest(game_df)
    except Exception as e:  # noqa: BLE001 — bookkeeping must not block predictions
        log.warning("Performance reconciliation failed (non-fatal): %s", e)


def _predict_and_publish(game_df, models):
    from predict import generate_predictions
    preds = generate_predictions(game_df, models, CURRENT_SEASON, save=True)
    log.info("  Generated %d predictions", len(preds))
    docs = ROOT / "docs" / "assets"
    docs.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED:
        if (PRED_DIR / name).exists():
            shutil.copy2(PRED_DIR / name, docs / name)


def run_full(args):
    log.info("FULL PIPELINE  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    game_df = _build_features(force_data=args.force_data)
    _reconcile(game_df)
    _train(game_df)
    from train import load_models
    _predict_and_publish(game_df, load_models())
    log.info("FULL PIPELINE COMPLETE")


def run_predict_only(args):
    log.info("PREDICT-ONLY  —  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    from feature_engineering import get_feature_columns
    from train import load_models
    game_df = _build_features()
    _reconcile(game_df)
    try:
        models = load_models()
        if models["feature_cols"] != get_feature_columns(game_df):
            raise FileNotFoundError("saved model was trained on a different feature set")
    except FileNotFoundError as e:
        log.warning("Saved model unusable (%s) — training instead.", e)
        _train(game_df)
        models = load_models()
    _predict_and_publish(game_df, models)


def run_backtest(args):
    from evaluate import run_backtest as backtest
    seasons = [args.backtest_season] if args.backtest_season else [CURRENT_SEASON - 2, CURRENT_SEASON - 1]
    results = {}
    for s in seasons:
        log.info("BACKTEST %d  —  %s", s, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        game_df = _build_features(seasons=list(range(s - 4, s + 1)))
        results[str(s)] = backtest(game_df, s)
    if args.backtest_season:
        out = PRED_DIR / "backtest_fragments" / f"backtest_{args.backtest_season}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = PRED_DIR / "backtest_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log.info("Backtest results → %s", out)


def main():
    ap = argparse.ArgumentParser(description="NFL score predictor")
    ap.add_argument("--mode", choices=["full", "predict_only", "backtest"], default="full")
    ap.add_argument("--force-data", action="store_true",
                    help="re-download live data (frozen closed-season data is never re-downloaded)")
    ap.add_argument("--backtest-season", type=int, default=None,
                    help="backtest only this season and write a fragment for the CI merge")
    args = ap.parse_args()
    {"full": run_full, "predict_only": run_predict_only, "backtest": run_backtest}[args.mode](args)


if __name__ == "__main__":
    main()

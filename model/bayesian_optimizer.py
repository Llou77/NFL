"""
bayesian_optimizer.py
=====================
Bayesian hyperparameter optimisation via Optuna.

WHAT IS ACTUALLY SEARCHED (and why only this):
  The optimisation objective is a fast GBR proxy trained on a FIXED,
  pre-built feature matrix. The only parameters that can influence that
  objective are the SAMPLE WEIGHTS:

    w_oldest, w_middle, w_recent, w_current   (season recency weights)
    wt_wc, wt_div, wt_con, wt_sb              (playoff game-type weights)

  The previous version also "searched" 18 further parameters (elo_k,
  variance_scale, confidence weights, NN loss weights, lean thresholds, …).
  None of them touched the proxy objective — the features were already built
  and the proxy uses neither the confidence module nor the NN — so Optuna was
  sampling pure noise in 18 of 26 dimensions, and a sum-constraint on the
  confidence weights silently discarded most trials with a 999 penalty.
  That noise diluted the TPE sampler and made the found "optimum" unstable.

EVERYTHING ELSE stays a research-backed constant in DEFAULT_WEIGHTS
(consumed at runtime by feature_engineering / train / confidence / predict
via load_weights). Tuning those honestly would require rebuilding features
and retraining the full ensemble inside the objective — deliberately out of
scope for a weekly CI job.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)  # module logger

ROOT       = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "model" / "saved" / "optimal_weights.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    # Season recency
    "w_oldest":  0.60,
    "w_middle":  0.80,
    "w_recent":  1.00,
    "w_current": 1.20,
    # Game type
    "wt_wc":  0.70,
    "wt_div": 0.75,
    "wt_con": 0.75,
    "wt_sb":  0.60,
    # Model calibration
    "variance_scale":  1.8,
    "elo_k":           20.0,
    "elo_regression":  0.67,
    "elo_mov_mult":    2.2,
    "off_epa_weight":  1.6,    # research-backed: nfelo WEPA study
    "turnover_weight": 0.60,   # research-backed: ~54% luck, downweight
    # Confidence
    "conf_w_model":    0.55,
    "conf_w_feat":     0.30,
    "conf_w_h2h":      0.15,
    "conf_model_div":  10.0,
    "conf_high_thr":   0.74,
    "conf_med_thr":    0.58,
    # NN loss
    "nn_w_home":   0.40,
    "nn_w_away":   0.40,
    "nn_w_total":  0.35,
    "nn_w_spread": 0.25,
    # Lean thresholds
    "spread_lean_thr": 1.5,
    "total_lean_thr":  1.5,
}


def load_weights() -> dict:
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH) as f:
            saved = json.load(f)
        # Merge: saved values override defaults (handles new params in new versions)
        merged = {**DEFAULT_WEIGHTS, **saved}
        return merged
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Always persist a full set including defaults for missing keys
    full = {**DEFAULT_WEIGHTS, **weights}
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(full, f, indent=2)
    logger.info("Weights saved → %s", WEIGHTS_PATH)


def compute_sample_weights(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
    current_season: Optional[int] = None,
) -> np.ndarray:
    if weights is None:
        weights = load_weights()
    if current_season is None:
        from data_loader import CURRENT_SEASON as current_season

    season_weights = {
        current_season - 3: weights["w_oldest"],
        current_season - 2: weights["w_middle"],
        current_season - 1: weights["w_recent"],
        current_season:     weights["w_current"],
    }
    game_type_weights = {
        "REG": 1.0,
        "WC":  weights["wt_wc"],
        "DIV": weights["wt_div"],
        "CON": weights["wt_con"],
        "SB":  weights["wt_sb"],
    }

    sw = df["season"].map(season_weights).fillna(weights["w_oldest"]).values
    gw = df["game_type"].map(game_type_weights).fillna(1.0).values

    # COVID 2020: no fans → home field advantage near zero
    # Downweight heavily to avoid learning aberrant HFA patterns
    COVID_PENALTY = 0.15
    covid_mask = (df["season"] == 2020).values
    sw = np.where(covid_mask, sw * COVID_PENALTY, sw)

    return (sw * gw).astype(np.float32)


def run_bayesian_optimization(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list,
    n_trials: int = 80,
    timeout: int = 900,
) -> dict:
    """
    Run Optuna Bayesian optimisation across ALL tunable parameters.
    Uses a fast GBR proxy for speed (full ensemble would take hours).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — using default weights")
        save_weights(DEFAULT_WEIGHTS)
        return DEFAULT_WEIGHTS

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    feat_exist = [c for c in feature_cols if c in df_train.columns]

    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(df_train[feat_exist].values)
    X_val   = imp.transform(df_val[feat_exist].values)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_val_sc   = sc.transform(X_val)

    y_val_home = df_val["target_home_score"].values
    y_val_away = df_val["target_away_score"].values

    # Recency weights must be interpreted relative to the VALIDATION season,
    # not the wall-clock current season (they may differ during backfills).
    ref_season = int(df_val["season"].max())

    def _trial_mae(w: dict) -> float:
        sw = compute_sample_weights(df_train, w, current_season=ref_season)
        sw = sw / sw.mean()
        mae_total = 0.0
        for y_tr, y_val in [
            (df_train["target_home_score"].values, y_val_home),
            (df_train["target_away_score"].values, y_val_away),
        ]:
            m = GradientBoostingRegressor(
                n_estimators=80, max_depth=4,
                learning_rate=0.1, subsample=0.8, random_state=42,
            )
            m.fit(X_train_sc, y_tr, sample_weight=sw)
            mae_total += np.mean(np.abs(m.predict(X_val_sc) - y_val))
        return mae_total / 2.0

    # ONLY the parameters that actually influence the proxy objective are
    # searched — see the module docstring for why the other 18 were removed.
    SEARCHED_PARAMS = (
        "w_oldest", "w_middle", "w_recent", "w_current",
        "wt_wc", "wt_div", "wt_con", "wt_sb",
    )

    def objective(trial):
        w = {
            # ── Season recency ────────────────────────────────────────
            "w_oldest":  trial.suggest_float("w_oldest",  0.10, 0.90),
            "w_middle":  trial.suggest_float("w_middle",  0.40, 1.10),
            "w_recent":  trial.suggest_float("w_recent",  0.65, 1.20),
            "w_current": trial.suggest_float("w_current", 0.85, 1.50),
            # ── Game type ─────────────────────────────────────────────
            "wt_wc":  trial.suggest_float("wt_wc",  0.35, 0.90),
            "wt_div": trial.suggest_float("wt_div", 0.35, 0.90),
            "wt_con": trial.suggest_float("wt_con", 0.35, 0.90),
            "wt_sb":  trial.suggest_float("wt_sb",  0.20, 0.80),
        }
        # Non-searched params come from research-backed defaults so the
        # proxy sees the exact configuration production will use.
        return _trial_mae({**DEFAULT_WEIGHTS, **w})

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )
    # Seed with research-backed defaults (searched subset only)
    study.enqueue_trial({k: DEFAULT_WEIGHTS[k] for k in SEARCHED_PARAMS})

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best     = {**DEFAULT_WEIGHTS, **study.best_params}
    best_mae = study.best_value
    logger.info("Optimisation complete. Best proxy MAE: %.4f", best_mae)
    logger.info("Best sample weights: %s",
                {k: round(v, 3) for k, v in study.best_params.items()})

    save_weights(best)
    return best

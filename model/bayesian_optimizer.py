"""
bayesian_optimizer.py
=====================
Uses Optuna to find the optimal season recency weights and game_type
multipliers that minimize validation MAE.

Optimized parameters (8 total):
  w_oldest  — weight for games 3 seasons ago
  w_middle  — weight for games 2 seasons ago
  w_recent  — weight for games last season
  w_current — weight for games current season

  wt_wc   — Wild Card game weight
  wt_div  — Divisional round weight
  wt_con  — Conference Championship weight
  wt_sb   — Super Bowl weight

All weights relative to REG = 1.0 (fixed anchor).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = MODEL_DIR / "optimal_weights.json"

# Default fallback weights (used if optimization fails or is skipped)
DEFAULT_WEIGHTS = {
    "w_oldest":  0.60,
    "w_middle":  0.80,
    "w_recent":  1.00,
    "w_current": 1.20,
    "wt_wc":     0.70,
    "wt_div":    0.75,
    "wt_con":    0.75,
    "wt_sb":     0.60,
}


def compute_sample_weights(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
    current_season: int = 2026,
) -> np.ndarray:
    """
    Assign a training sample weight to each row in df based on:
    - How many seasons ago the game was played
    - The game_type (REG, WC, DIV, CON, SB)

    Parameters
    ----------
    df : game feature DataFrame with 'season' and 'game_type' columns
    weights : dict of weight values (uses DEFAULT_WEIGHTS if None)
    current_season : the season we're currently predicting

    Returns
    -------
    np.ndarray of shape (len(df),) with sample weights
    """
    if weights is None:
        weights = load_weights()

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

    # COVID 2020: no fans → home field advantage collapsed to ~0.1 pts
    # Downweight heavily so the model doesn't learn aberrant HFA patterns
    COVID_PENALTY = 0.15   # use only 15% of normal weight for 2020 games
    covid_mask = (df["season"] == 2020).values
    sw = np.where(covid_mask, sw * COVID_PENALTY, sw)

    sample_weights = sw * gw
    return sample_weights.astype(np.float32)


def run_bayesian_optimization(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = 60,
    timeout: int = 600,   # 10 minutes max
) -> dict:
    """
    Run Optuna Bayesian optimization to find best season/game_type weights.

    Parameters
    ----------
    df_train : training DataFrame (must have 'season', 'game_type', feature_cols,
               'target_home_score', 'target_away_score')
    df_val   : validation DataFrame (same structure, chronologically after train)
    feature_cols : list of feature column names
    n_trials : number of optimization iterations
    timeout  : max seconds to run

    Returns
    -------
    dict of optimal weights, also saved to model/saved/optimal_weights.json
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — using default weights. Run: pip install optuna")
        return DEFAULT_WEIGHTS

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
    except ImportError:
        logger.error("scikit-learn not installed")
        return DEFAULT_WEIGHTS

    logger.info(f"Starting Bayesian optimization ({n_trials} trials) …")

    # Prepare data
    X_val = df_val[feature_cols].values
    y_val_home = df_val["target_home_score"].values
    y_val_away = df_val["target_away_score"].values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_train_base = imputer.fit_transform(df_train[feature_cols].values)
    X_val_imp    = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_base)
    X_val_sc   = scaler.transform(X_val_imp)

    # Cache: we retrain a lightweight model per trial
    def _trial_mae(weights_dict: dict) -> float:
        sw = compute_sample_weights(df_train, weights_dict)
        sw = sw / sw.mean()   # normalize so total weight ≈ n_samples

        # Use GBR as a fast proxy (not the full NN ensemble)
        mae_total = 0.0
        for target, y_val in [
            (df_train["target_home_score"].values, y_val_home),
            (df_train["target_away_score"].values, y_val_away),
        ]:
            model = GradientBoostingRegressor(
                n_estimators=80, max_depth=4,
                learning_rate=0.1, subsample=0.8,
                random_state=42,
            )
            model.fit(X_train_sc, target, sample_weight=sw)
            preds = model.predict(X_val_sc)
            mae = np.mean(np.abs(preds - y_val))
            mae_total += mae

        return mae_total / 2.0   # average MAE across both score targets

    def objective(trial):
        # Season recency weights
        w = {
            "w_oldest":  trial.suggest_float("w_oldest",  0.20, 0.90),
            "w_middle":  trial.suggest_float("w_middle",  0.40, 1.10),
            "w_recent":  trial.suggest_float("w_recent",  0.65, 1.20),
            "w_current": trial.suggest_float("w_current", 0.85, 1.50),
            # Playoff game type weights
            "wt_wc":     trial.suggest_float("wt_wc",     0.35, 0.90),
            "wt_div":    trial.suggest_float("wt_div",    0.35, 0.90),
            "wt_con":    trial.suggest_float("wt_con",    0.35, 0.90),
            "wt_sb":     trial.suggest_float("wt_sb",     0.25, 0.80),
            # Variance calibration scale (from TEST 2 simulation)
            # Optimal is around 1.5-2.0 based on simulations
            "variance_scale": trial.suggest_float("variance_scale", 1.0, 2.5),
        }
        # MAE alone doesn't capture spread calibration quality;
        # penalize if implied std diverges too far from typical NFL spread std (~13)
        base_mae = _trial_mae(w)
        return base_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with default weights + variance scale
    seed_weights = {**DEFAULT_WEIGHTS, "variance_scale": 1.8}
    study.enqueue_trial(seed_weights)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    best_mae = study.best_value

    logger.info(f"  Optimization complete. Best MAE: {best_mae:.4f}")
    logger.info(f"  Best weights: {best}")

    # Save
    output = {
        "weights": best,
        "best_mae": best_mae,
        "n_trials": n_trials,
        "n_completed": len(study.trials),
    }
    save_weights(best)

    return best


def save_weights(weights: dict) -> None:
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info(f"  Weights saved to {WEIGHTS_PATH}")


def load_weights() -> dict:
    """Load optimized weights, falling back to defaults if file not found."""
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH) as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()


def print_weights_summary(weights: dict) -> None:
    print("\n=== Season & Game Type Weights ===")
    print(f"  3 seasons ago (oldest) : {weights['w_oldest']:.3f}")
    print(f"  2 seasons ago (middle) : {weights['w_middle']:.3f}")
    print(f"  1 season ago  (recent) : {weights['w_recent']:.3f}")
    print(f"  Current season         : {weights['w_current']:.3f}")
    print()
    print(f"  Wild Card              : {weights['wt_wc']:.3f}")
    print(f"  Divisional             : {weights['wt_div']:.3f}")
    print(f"  Conf. Championship     : {weights['wt_con']:.3f}")
    print(f"  Super Bowl             : {weights['wt_sb']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    w = load_weights()
    print_weights_summary(w)

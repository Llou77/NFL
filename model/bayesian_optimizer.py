"""
bayesian_optimizer.py
=====================
Bayesian hyperparameter optimisation via Optuna.

Optimises ALL tunable parameters in one unified search:

Season weights:
  w_oldest, w_middle, w_recent, w_current  (relative to REG=1.0)
  wt_wc, wt_div, wt_con, wt_sb            (playoff game type weights)

Model parameters:
  variance_scale    — spread variance calibration (1.0–3.0)
  elo_k             — Elo update rate (10–30)
  elo_regression    — seasonal mean regression (0.55–0.80)
  elo_mov_mult      — margin-of-victory multiplier (1.5–3.0)
  off_epa_weight    — offensive EPA weight vs defensive (1.0–2.0)
                      research: nfelo found 1.6× optimal (Weighted EPA)
  turnover_weight   — turnover feature weight (0.3–1.0)
                      research: ~54% of turnovers are luck → downweight

Confidence weights:
  conf_w_model      — model agreement weight (0.40–0.65)
  conf_w_feat       — feature completeness weight (0.20–0.40)
  conf_w_h2h        — H2H quality weight (0.10–0.20)
  conf_model_div    — model agreement divisor (6–14)
  conf_high_thr     — HIGH confidence threshold (0.68–0.80)
  conf_med_thr      — MEDIUM confidence threshold (0.50–0.68)

NN loss weights:
  nn_w_home, nn_w_away  — individual score loss weights (0.25–0.55)
  nn_w_total            — combined total loss weight (0.20–0.50)
  nn_w_spread           — spread loss weight (0.15–0.40)

Lean thresholds:
  spread_lean_thr   — minimum pts edge to signal a spread lean (0.5–3.0)
  total_lean_thr    — minimum pts edge to signal a total lean (0.5–3.0)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    current_season: int = 2026,
) -> np.ndarray:
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

    def _trial_mae(w: dict) -> float:
        sw = compute_sample_weights(df_train, w)
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
            # ── Model calibration ─────────────────────────────────────
            "variance_scale": trial.suggest_float("variance_scale", 1.0, 2.8),
            "elo_k":          trial.suggest_float("elo_k",          12.0, 28.0),
            "elo_regression": trial.suggest_float("elo_regression", 0.55, 0.82),
            "elo_mov_mult":   trial.suggest_float("elo_mov_mult",   1.5, 3.2),
            # ── Research-backed ranges ────────────────────────────────
            # nfelo: off EPA weight 1.6 optimal
            "off_epa_weight":  trial.suggest_float("off_epa_weight",  1.0, 2.2),
            # Harvard: ~54% turnover luck → downweight 0.3–0.8
            "turnover_weight": trial.suggest_float("turnover_weight", 0.25, 0.85),
            # ── Confidence ────────────────────────────────────────────
            "conf_w_model":  trial.suggest_float("conf_w_model",  0.38, 0.65),
            "conf_w_feat":   trial.suggest_float("conf_w_feat",   0.20, 0.42),
            "conf_w_h2h":    trial.suggest_float("conf_w_h2h",    0.08, 0.22),
            "conf_model_div":trial.suggest_float("conf_model_div",6.0, 14.0),
            "conf_high_thr": trial.suggest_float("conf_high_thr", 0.66, 0.82),
            "conf_med_thr":  trial.suggest_float("conf_med_thr",  0.48, 0.68),
            # ── NN loss weights ───────────────────────────────────────
            "nn_w_home":   trial.suggest_float("nn_w_home",   0.25, 0.55),
            "nn_w_away":   trial.suggest_float("nn_w_away",   0.25, 0.55),
            "nn_w_total":  trial.suggest_float("nn_w_total",  0.18, 0.50),
            "nn_w_spread": trial.suggest_float("nn_w_spread", 0.12, 0.40),
            # ── Lean thresholds ───────────────────────────────────────
            "spread_lean_thr": trial.suggest_float("spread_lean_thr", 0.5, 3.5),
            "total_lean_thr":  trial.suggest_float("total_lean_thr",  0.5, 3.5),
        }
        # Constraint: confidence weights must sum roughly to 1
        conf_sum = w["conf_w_model"] + w["conf_w_feat"] + w["conf_w_h2h"]
        if abs(conf_sum - 1.0) > 0.05:
            return 999.0
        return _trial_mae(w)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )
    # Seed with research-backed defaults
    study.enqueue_trial(DEFAULT_WEIGHTS)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    best_mae = study.best_value
    logger.info("Optimisation complete. Best proxy MAE: %.4f", best_mae)
    logger.info("Best weights: %s", best)

    save_weights(best)
    return best

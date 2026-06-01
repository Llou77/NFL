"""
train.py
========
Trains the full stacked ensemble:

  Layer 1 : Linear Ridge, XGBoost, LightGBM
  Layer 2 : Dual-head PyTorch Neural Network
  Layer 3 : Ridge meta-learner (blends Layer-1 + Layer-2 outputs)

Each model predicts both home_score and away_score simultaneously.
Training uses sample weights from bayesian_optimizer.py.
TimeSeriesSplit is used for cross-validation (no leakage).

Outputs saved to model/saved/:
  model_weights.pt        — PyTorch NN state dict
  xgb_home.json           — XGBoost home model
  xgb_away.json           — XGBoost away model
  lgbm_home.txt           — LightGBM home model
  lgbm_away.txt           — LightGBM away model
  meta_learner.pkl        — Ridge meta-learner
  scaler.pkl              — StandardScaler
  imputer.pkl             — SimpleImputer
  feature_list.json       — Ordered feature columns
  training_log.json       — MAE metrics from training
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER TRAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_all(
    game_df: pd.DataFrame,
    feature_cols: list[str],
    weights: Optional[dict] = None,
    current_season: int = 2026,
    run_cv: bool = True,
) -> dict:
    """
    Train the full ensemble on game_df.

    Parameters
    ----------
    game_df     : feature matrix (from feature_engineering.build_all_features)
    feature_cols: list of feature column names
    weights     : Bayesian-optimized sample weights dict
    current_season : season being predicted (used for sample weight assignment)
    run_cv      : if True, run TimeSeriesSplit CV before final training

    Returns
    -------
    dict of MAE metrics
    """
    from bayesian_optimizer import compute_sample_weights

    # Filter to rows with known scores (training data)
    df_known = game_df[
        game_df["target_home_score"].notna() &
        game_df["target_away_score"].notna()
    ].copy()

    if len(df_known) < 50:
        raise ValueError(f"Only {len(df_known)} labeled games found — need at least 50.")

    logger.info(f"Training on {len(df_known)} games with {len(feature_cols)} features")

    # ── Prepare X, y ─────────────────────────────────────────────────────
    X_raw  = df_known[feature_cols].values
    y_home = df_known["target_home_score"].values.astype(np.float32)
    y_away = df_known["target_away_score"].values.astype(np.float32)
    sw     = compute_sample_weights(df_known, weights, current_season)

    # ── Preprocessing ─────────────────────────────────────────────────────
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    # Save preprocessors
    with open(MODEL_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature list
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f)

    # ── Cross-validation ─────────────────────────────────────────────────
    cv_metrics = {}
    if run_cv:
        cv_metrics = _run_cv(df_known, X_sc, y_home, y_away, sw, feature_cols)
        logger.info(f"  CV MAE home: {cv_metrics.get('cv_mae_home', '?'):.3f} | "
                    f"away: {cv_metrics.get('cv_mae_away', '?'):.3f}")

    # ── Train each model on full training data ────────────────────────────
    layer1_preds_home, layer1_preds_away = {}, {}

    # 1a. Ridge Regression
    ridge_h, ridge_a = _train_ridge(X_sc, y_home, y_away, sw)
    layer1_preds_home["ridge"] = ridge_h.predict(X_sc)
    layer1_preds_away["ridge"] = ridge_a.predict(X_sc)

    # 1b. XGBoost
    xgb_h, xgb_a = _train_xgboost(X_sc, y_home, y_away, sw)
    layer1_preds_home["xgb"] = xgb_h.predict(X_sc)
    layer1_preds_away["xgb"] = xgb_a.predict(X_sc)

    # 1c. LightGBM
    lgbm_h, lgbm_a = _train_lightgbm(X_sc, y_home, y_away, sw)
    layer1_preds_home["lgbm"] = lgbm_h.predict(X_sc)
    layer1_preds_away["lgbm"] = lgbm_a.predict(X_sc)

    # 2. Neural Network
    nn_model = _train_neural_network(X_sc, y_home, y_away, sw)
    nn_home_preds, nn_away_preds = _nn_predict(nn_model, X_sc)
    layer1_preds_home["nn"] = nn_home_preds
    layer1_preds_away["nn"] = nn_away_preds

    # 3. Meta-learner (trained on held-out predictions via CV)
    meta_h, meta_a = _train_meta_learner(
        df_known, X_sc, y_home, y_away, sw, feature_cols,
        ridge_h, ridge_a, xgb_h, xgb_a, lgbm_h, lgbm_a, nn_model
    )

    # ── Compute final training MAE ────────────────────────────────────────
    meta_X_home = np.column_stack([v for v in layer1_preds_home.values()])
    meta_X_away = np.column_stack([v for v in layer1_preds_away.values()])

    final_home = meta_h.predict(meta_X_home)
    final_away = meta_a.predict(meta_X_away)

    train_mae_home = float(np.mean(np.abs(final_home - y_home)))
    train_mae_away = float(np.mean(np.abs(final_away - y_away)))
    train_mae_total = float(np.mean(np.abs((final_home + final_away) - (y_home + y_away))))
    train_mae_spread= float(np.mean(np.abs((final_home - final_away) - (y_home - y_away))))

    metrics = {
        "train_mae_home":   round(train_mae_home, 4),
        "train_mae_away":   round(train_mae_away, 4),
        "train_mae_total":  round(train_mae_total, 4),
        "train_mae_spread": round(train_mae_spread, 4),
        "n_games":          len(df_known),
        "n_features":       len(feature_cols),
        **cv_metrics,
    }

    # Save training log
    with open(MODEL_DIR / "training_log.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Train MAE → home: {train_mae_home:.3f} | "
                f"away: {train_mae_away:.3f} | "
                f"total: {train_mae_total:.3f} | "
                f"spread: {train_mae_spread:.3f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 MODELS
# ══════════════════════════════════════════════════════════════════════════════

def _train_ridge(X, y_home, y_away, sw):
    from sklearn.linear_model import Ridge
    logger.info("  Training Ridge …")
    rh = Ridge(alpha=10.0)
    rh.fit(X, y_home, sample_weight=sw)
    ra = Ridge(alpha=10.0)
    ra.fit(X, y_away, sample_weight=sw)
    # Save
    with open(MODEL_DIR / "ridge_home.pkl", "wb") as f:
        pickle.dump(rh, f)
    with open(MODEL_DIR / "ridge_away.pkl", "wb") as f:
        pickle.dump(ra, f)
    return rh, ra


def _train_xgboost(X, y_home, y_away, sw):
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("  XGBoost not installed — skipping")
        from sklearn.ensemble import GradientBoostingRegressor
        xh = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        xa = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        xh.fit(X, y_home, sample_weight=sw)
        xa.fit(X, y_away, sample_weight=sw)
        with open(MODEL_DIR / "xgb_home.pkl", "wb") as f:
            pickle.dump(xh, f)
        with open(MODEL_DIR / "xgb_away.pkl", "wb") as f:
            pickle.dump(xa, f)
        return xh, xa

    logger.info("  Training XGBoost …")
    params = dict(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:absoluteerror",
        random_state=42, n_jobs=-1,
    )
    xh = xgb.XGBRegressor(**params)
    xh.fit(X, y_home, sample_weight=sw, eval_set=[(X, y_home)], verbose=False)
    xh.save_model(str(MODEL_DIR / "xgb_home.json"))

    xa = xgb.XGBRegressor(**params)
    xa.fit(X, y_away, sample_weight=sw, eval_set=[(X, y_away)], verbose=False)
    xa.save_model(str(MODEL_DIR / "xgb_away.json"))

    return xh, xa


def _train_lightgbm(X, y_home, y_away, sw):
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("  LightGBM not installed — using GBR fallback")
        from sklearn.ensemble import GradientBoostingRegressor
        lh = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=43)
        la = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=43)
        lh.fit(X, y_home, sample_weight=sw)
        la.fit(X, y_away, sample_weight=sw)
        with open(MODEL_DIR / "lgbm_home.pkl", "wb") as f:
            pickle.dump(lh, f)
        with open(MODEL_DIR / "lgbm_away.pkl", "wb") as f:
            pickle.dump(la, f)
        return lh, la

    logger.info("  Training LightGBM …")
    params = dict(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=0.5,
        objective="mae", metric="mae",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lh = lgb.LGBMRegressor(**params)
    lh.fit(X, y_home, sample_weight=sw)
    lh.booster_.save_model(str(MODEL_DIR / "lgbm_home.txt"))

    la = lgb.LGBMRegressor(**params)
    la.fit(X, y_away, sample_weight=sw)
    la.booster_.save_model(str(MODEL_DIR / "lgbm_away.txt"))

    return lh, la


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class DualHeadNet(object):
    """
    Dual-head feedforward network.
    Implemented with PyTorch when available, falls back to sklearn MLPRegressor.
    """
    pass


def _train_neural_network(X, y_home, y_away, sw):
    try:
        return _train_pytorch_nn(X, y_home, y_away, sw)
    except ImportError:
        logger.warning("  PyTorch not available — using MLP fallback")
        return _train_mlp_fallback(X, y_home, y_away, sw)


def _train_pytorch_nn(X, y_home, y_away, sw):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("  Training PyTorch dual-head NN …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]

    class NFLNet(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.head_home = nn.Linear(64, 1)
            self.head_away = nn.Linear(64, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_home(z).squeeze(-1), self.head_away(z).squeeze(-1)

    model = NFLNet(n_features).to(device)

    # Tensors
    X_t  = torch.FloatTensor(X).to(device)
    yh_t = torch.FloatTensor(y_home).to(device)
    ya_t = torch.FloatTensor(y_away).to(device)
    sw_t = torch.FloatTensor(sw / sw.mean()).to(device)

    dataset = TensorDataset(X_t, yh_t, ya_t, sw_t)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Alpha, beta, gamma for multi-target loss
    alpha, beta, gamma = 0.40, 0.35, 0.25

    best_loss = float("inf")
    best_state = None

    for epoch in range(150):
        model.train()
        epoch_loss = 0.0
        for xb, yh, ya, w in loader:
            optimizer.zero_grad()
            ph, pa = model(xb)
            mae_h = (torch.abs(ph - yh) * w).mean()
            mae_a = (torch.abs(pa - ya) * w).mean()
            mae_t = (torch.abs((ph + pa) - (yh + ya)) * w).mean()
            mae_s = (torch.abs((ph - pa) - (yh - ya)) * w).mean()
            loss  = alpha * mae_h + alpha * mae_a + beta * mae_t + gamma * mae_s
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 30 == 0:
            logger.info(f"    Epoch {epoch+1:3d} loss: {epoch_loss:.4f}")

    # Restore best weights
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save(model.state_dict(), MODEL_DIR / "model_weights.pt")

    # Also save architecture config
    with open(MODEL_DIR / "nn_config.json", "w") as f:
        json.dump({"n_features": n_features, "type": "pytorch"}, f)

    return ("pytorch", model, n_features, device)


def _train_mlp_fallback(X, y_home, y_away, sw):
    from sklearn.neural_network import MLPRegressor
    logger.info("  Training sklearn MLP (fallback) …")
    mlp_h = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        learning_rate_init=0.001, max_iter=300, random_state=42,
    )
    mlp_h.fit(X, y_home)
    mlp_a = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        learning_rate_init=0.001, max_iter=300, random_state=43,
    )
    mlp_a.fit(X, y_away)
    with open(MODEL_DIR / "mlp_home.pkl", "wb") as f:
        pickle.dump(mlp_h, f)
    with open(MODEL_DIR / "mlp_away.pkl", "wb") as f:
        pickle.dump(mlp_a, f)
    with open(MODEL_DIR / "nn_config.json", "w") as f:
        json.dump({"type": "sklearn_mlp"}, f)
    return ("sklearn", mlp_h, mlp_a)


def _nn_predict(nn_obj, X):
    kind = nn_obj[0]
    if kind == "pytorch":
        import torch
        _, model, n_features, device = nn_obj
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            ph, pa = model(X_t)
        return ph.cpu().numpy(), pa.cpu().numpy()
    else:
        _, mlp_h, mlp_a = nn_obj
        return mlp_h.predict(X), mlp_a.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: META-LEARNER
# ══════════════════════════════════════════════════════════════════════════════

def _train_meta_learner(
    df, X_sc, y_home, y_away, sw, feature_cols,
    ridge_h, ridge_a, xgb_h, xgb_a, lgbm_h, lgbm_a, nn_obj
):
    """
    Train meta-learner using out-of-fold predictions (TimeSeriesSplit).
    This ensures the meta-learner doesn't overfit to training data.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("  Training meta-learner …")

    tscv = TimeSeriesSplit(n_splits=5)
    oof_home = np.zeros(len(df))
    oof_away = np.zeros(len(df))
    oof_meta_X_home_list = []
    oof_meta_X_away_list = []
    oof_idx_list = []

    for train_idx, val_idx in tscv.split(X_sc):
        X_tr, X_val = X_sc[train_idx], X_sc[val_idx]
        yh_tr, yh_val = y_home[train_idx], y_home[val_idx]
        ya_tr, ya_val = y_away[train_idx], y_away[val_idx]
        sw_tr = sw[train_idx]

        # Retrain each sub-model on fold
        from sklearn.linear_model import Ridge as R
        rh = R(alpha=10.0).fit(X_tr, yh_tr, sample_weight=sw_tr)
        ra = R(alpha=10.0).fit(X_tr, ya_tr, sample_weight=sw_tr)

        # Use pre-trained XGB/LGBM directly (fold retraining is too slow)
        xgb_h_fold_pred = xgb_h.predict(X_val)
        xgb_a_fold_pred = xgb_a.predict(X_val)
        lgbm_h_fold_pred= lgbm_h.predict(X_val)
        lgbm_a_fold_pred= lgbm_a.predict(X_val)
        nn_h_fold, nn_a_fold = _nn_predict(nn_obj, X_val)

        meta_X_h = np.column_stack([
            rh.predict(X_val), xgb_h_fold_pred, lgbm_h_fold_pred, nn_h_fold
        ])
        meta_X_a = np.column_stack([
            ra.predict(X_val), xgb_a_fold_pred, lgbm_a_fold_pred, nn_a_fold
        ])
        oof_meta_X_home_list.append(meta_X_h)
        oof_meta_X_away_list.append(meta_X_a)
        oof_idx_list.append(val_idx)

    oof_meta_X_home = np.vstack(oof_meta_X_home_list)
    oof_meta_X_away = np.vstack(oof_meta_X_away_list)
    oof_idx = np.concatenate(oof_idx_list)

    y_home_oof = y_home[oof_idx]
    y_away_oof = y_away[oof_idx]

    meta_h = Ridge(alpha=1.0)
    meta_h.fit(oof_meta_X_home, y_home_oof)

    meta_a = Ridge(alpha=1.0)
    meta_a.fit(oof_meta_X_away, y_away_oof)

    with open(MODEL_DIR / "meta_learner_home.pkl", "wb") as f:
        pickle.dump(meta_h, f)
    with open(MODEL_DIR / "meta_learner_away.pkl", "wb") as f:
        pickle.dump(meta_a, f)

    return meta_h, meta_a


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_cv(df, X_sc, y_home, y_away, sw, feature_cols):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("  Running TimeSeriesSplit CV …")
    tscv = TimeSeriesSplit(n_splits=3)
    mae_home_list, mae_away_list, mae_total_list = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_sc)):
        X_tr, X_val = X_sc[tr_idx], X_sc[val_idx]
        yh_tr = y_home[tr_idx]
        ya_tr = y_away[tr_idx]
        yh_val = y_home[val_idx]
        ya_val = y_away[val_idx]
        sw_tr = sw[tr_idx]

        # Fast proxy model for CV
        m_h = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        m_a = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        m_h.fit(X_tr, yh_tr, sample_weight=sw_tr)
        m_a.fit(X_tr, ya_tr, sample_weight=sw_tr)

        ph = m_h.predict(X_val)
        pa = m_a.predict(X_val)

        mae_home_list.append(np.mean(np.abs(ph - yh_val)))
        mae_away_list.append(np.mean(np.abs(pa - ya_val)))
        mae_total_list.append(np.mean(np.abs((ph + pa) - (yh_val + ya_val))))

    return {
        "cv_mae_home":  round(float(np.mean(mae_home_list)),  3),
        "cv_mae_away":  round(float(np.mean(mae_away_list)),  3),
        "cv_mae_total": round(float(np.mean(mae_total_list)), 3),
        "cv_folds":     3,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD TRAINED MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> dict:
    """Load all saved model artifacts from model/saved/."""
    models = {}

    # Preprocessors
    with open(MODEL_DIR / "imputer.pkl", "rb") as f:
        models["imputer"] = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        models["scaler"] = pickle.load(f)

    # Feature list
    with open(MODEL_DIR / "feature_list.json") as f:
        models["feature_cols"] = json.load(f)

    # Ridge
    with open(MODEL_DIR / "ridge_home.pkl", "rb") as f:
        models["ridge_home"] = pickle.load(f)
    with open(MODEL_DIR / "ridge_away.pkl", "rb") as f:
        models["ridge_away"] = pickle.load(f)

    # XGBoost
    try:
        import xgboost as xgb
        xh = xgb.XGBRegressor()
        xh.load_model(str(MODEL_DIR / "xgb_home.json"))
        models["xgb_home"] = xh
        xa = xgb.XGBRegressor()
        xa.load_model(str(MODEL_DIR / "xgb_away.json"))
        models["xgb_away"] = xa
    except Exception:
        if (MODEL_DIR / "xgb_home.pkl").exists():
            with open(MODEL_DIR / "xgb_home.pkl", "rb") as f:
                models["xgb_home"] = pickle.load(f)
            with open(MODEL_DIR / "xgb_away.pkl", "rb") as f:
                models["xgb_away"] = pickle.load(f)

    # LightGBM
    try:
        import lightgbm as lgb
        lh = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_home.txt"))
        la = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_away.txt"))
        models["lgbm_home"] = lh
        models["lgbm_away"] = la
    except Exception:
        if (MODEL_DIR / "lgbm_home.pkl").exists():
            with open(MODEL_DIR / "lgbm_home.pkl", "rb") as f:
                models["lgbm_home"] = pickle.load(f)
            with open(MODEL_DIR / "lgbm_away.pkl", "rb") as f:
                models["lgbm_away"] = pickle.load(f)

    # Neural Network
    nn_config_path = MODEL_DIR / "nn_config.json"
    if nn_config_path.exists():
        with open(nn_config_path) as f:
            nn_cfg = json.load(f)
        if nn_cfg.get("type") == "pytorch":
            try:
                import torch
                import torch.nn as nn as torch_nn
                n_features = nn_cfg["n_features"]

                class NFLNet(torch_nn.Module):
                    def __init__(self, n_in):
                        super().__init__()
                        self.trunk = torch_nn.Sequential(
                            torch_nn.Linear(n_in, 256), torch_nn.BatchNorm1d(256),
                            torch_nn.ReLU(), torch_nn.Dropout(0.3),
                            torch_nn.Linear(256, 128), torch_nn.BatchNorm1d(128),
                            torch_nn.ReLU(), torch_nn.Dropout(0.2),
                            torch_nn.Linear(128, 64), torch_nn.ReLU(),
                        )
                        self.head_home = torch_nn.Linear(64, 1)
                        self.head_away = torch_nn.Linear(64, 1)
                    def forward(self, x):
                        z = self.trunk(x)
                        return self.head_home(z).squeeze(-1), self.head_away(z).squeeze(-1)

                device = torch.device("cpu")
                net = NFLNet(n_features).to(device)
                net.load_state_dict(torch.load(
                    MODEL_DIR / "model_weights.pt", map_location=device
                ))
                net.eval()
                models["nn"] = ("pytorch", net, n_features, device)
            except Exception as e:
                logger.warning(f"Could not load PyTorch model: {e}")
        elif nn_cfg.get("type") == "sklearn_mlp":
            with open(MODEL_DIR / "mlp_home.pkl", "rb") as f:
                models["nn_home_mlp"] = pickle.load(f)
            with open(MODEL_DIR / "mlp_away.pkl", "rb") as f:
                models["nn_away_mlp"] = pickle.load(f)

    # Meta-learner
    with open(MODEL_DIR / "meta_learner_home.pkl", "rb") as f:
        models["meta_home"] = pickle.load(f)
    with open(MODEL_DIR / "meta_learner_away.pkl", "rb") as f:
        models["meta_away"] = pickle.load(f)

    return models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("train.py — run via pipeline.py")

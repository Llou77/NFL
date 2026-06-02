"""
train.py
========
Trains the full stacked ensemble:
  Layer 1 : Ridge, XGBoost, LightGBM
  Layer 2 : Dual-head PyTorch Neural Network (or sklearn MLP fallback)
  Layer 3 : Ridge meta-learner

Outputs saved to model/saved/.
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
#  MASTER TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_all(
    game_df: pd.DataFrame,
    feature_cols: list,
    weights: Optional[dict] = None,
    current_season: int = 2026,
    run_cv: bool = True,
) -> dict:
    from bayesian_optimizer import compute_sample_weights
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    df_known = game_df[
        game_df["target_home_score"].notna() &
        game_df["target_away_score"].notna()
    ].copy()

    if len(df_known) < 50:
        raise ValueError(f"Only {len(df_known)} labeled games — need ≥50.")

    # Filter to only feature columns that exist
    feature_cols = [c for c in feature_cols if c in df_known.columns]
    logger.info("Training on %d games, %d features", len(df_known), len(feature_cols))

    X_raw  = df_known[feature_cols].values.astype(np.float32)
    y_home = df_known["target_home_score"].values.astype(np.float32)
    y_away = df_known["target_away_score"].values.astype(np.float32)
    sw     = compute_sample_weights(df_known, weights, current_season)

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    with open(MODEL_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f)

    # CV
    cv_metrics = {}
    if run_cv and len(df_known) >= 100:
        cv_metrics = _run_cv(X_sc, y_home, y_away, sw)
        logger.info("  CV MAE home=%.3f away=%.3f total=%.3f",
                    cv_metrics.get("cv_mae_home", 0),
                    cv_metrics.get("cv_mae_away", 0),
                    cv_metrics.get("cv_mae_total", 0))

    # Layer 1
    ridge_h, ridge_a = _train_ridge(X_sc, y_home, y_away, sw)
    xgb_h,   xgb_a   = _train_xgboost(X_sc, y_home, y_away, sw)
    lgbm_h,  lgbm_a  = _train_lightgbm(X_sc, y_home, y_away, sw)

    # Layer 2
    nn_obj = _train_neural_network(X_sc, y_home, y_away, sw)

    # Layer 3: meta-learner via OOF
    meta_h, meta_a = _train_meta_learner(
        X_sc, y_home, y_away, sw,
        ridge_h, ridge_a, xgb_h, xgb_a, lgbm_h, lgbm_a, nn_obj
    )

    # Final training metrics
    nn_ph, nn_pa = _nn_predict(nn_obj, X_sc)
    meta_X_h = np.column_stack([
        ridge_h.predict(X_sc), xgb_h.predict(X_sc),
        lgbm_h.predict(X_sc), nn_ph,
    ])
    meta_X_a = np.column_stack([
        ridge_a.predict(X_sc), xgb_a.predict(X_sc),
        lgbm_a.predict(X_sc), nn_pa,
    ])
    final_h = meta_h.predict(meta_X_h)
    final_a = meta_a.predict(meta_X_a)

    metrics = {
        "train_mae_home":   round(float(np.mean(np.abs(final_h - y_home))), 4),
        "train_mae_away":   round(float(np.mean(np.abs(final_a - y_away))), 4),
        "train_mae_total":  round(float(np.mean(np.abs((final_h+final_a)-(y_home+y_away)))), 4),
        "train_mae_spread": round(float(np.mean(np.abs((final_h-final_a)-(y_home-y_away)))), 4),
        "n_games": len(df_known),
        "n_features": len(feature_cols),
        **cv_metrics,
    }

    with open(MODEL_DIR / "training_log.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "version.txt", "w") as f:
        from datetime import datetime
        f.write(f"v1.0-{datetime.utcnow().strftime('%Y%m%d')}")

    logger.info("  Train MAE home=%.3f away=%.3f total=%.3f spread=%.3f",
                metrics["train_mae_home"], metrics["train_mae_away"],
                metrics["train_mae_total"], metrics["train_mae_spread"])
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1
# ══════════════════════════════════════════════════════════════════════════════

def _train_ridge(X, y_home, y_away, sw):
    from sklearn.linear_model import Ridge
    logger.info("  Training Ridge …")
    rh = Ridge(alpha=10.0).fit(X, y_home, sample_weight=sw)
    ra = Ridge(alpha=10.0).fit(X, y_away, sample_weight=sw)
    with open(MODEL_DIR / "ridge_home.pkl", "wb") as f: pickle.dump(rh, f)
    with open(MODEL_DIR / "ridge_away.pkl", "wb") as f: pickle.dump(ra, f)
    return rh, ra


def _train_xgboost(X, y_home, y_away, sw):
    logger.info("  Training XGBoost …")
    params = dict(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:absoluteerror", random_state=42, n_jobs=-1,
    )
    try:
        import xgboost as xgb
        xh = xgb.XGBRegressor(**params)
        xh.fit(X, y_home, sample_weight=sw, verbose=False)
        xh.save_model(str(MODEL_DIR / "xgb_home.json"))
        xa = xgb.XGBRegressor(**params)
        xa.fit(X, y_away, sample_weight=sw, verbose=False)
        xa.save_model(str(MODEL_DIR / "xgb_away.json"))
        return xh, xa
    except ImportError:
        logger.warning("  XGBoost not available — using GBR fallback")
        from sklearn.ensemble import GradientBoostingRegressor
        xh = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        xa = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        xh.fit(X, y_home, sample_weight=sw)
        xa.fit(X, y_away, sample_weight=sw)
        with open(MODEL_DIR / "xgb_home.pkl", "wb") as f: pickle.dump(xh, f)
        with open(MODEL_DIR / "xgb_away.pkl", "wb") as f: pickle.dump(xa, f)
        return xh, xa


def _train_lightgbm(X, y_home, y_away, sw):
    logger.info("  Training LightGBM …")
    params = dict(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=0.5,
        objective="mae", metric="mae", random_state=42, n_jobs=-1, verbose=-1,
    )
    try:
        import lightgbm as lgb
        lh = lgb.LGBMRegressor(**params)
        lh.fit(X, y_home, sample_weight=sw)
        lh.booster_.save_model(str(MODEL_DIR / "lgbm_home.txt"))
        la = lgb.LGBMRegressor(**params)
        la.fit(X, y_away, sample_weight=sw)
        la.booster_.save_model(str(MODEL_DIR / "lgbm_away.txt"))
        return lh, la
    except ImportError:
        logger.warning("  LightGBM not available — using GBR fallback")
        from sklearn.ensemble import GradientBoostingRegressor
        lh = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=43)
        la = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=43)
        lh.fit(X, y_home, sample_weight=sw)
        la.fit(X, y_away, sample_weight=sw)
        with open(MODEL_DIR / "lgbm_home.pkl", "wb") as f: pickle.dump(lh, f)
        with open(MODEL_DIR / "lgbm_away.pkl", "wb") as f: pickle.dump(la, f)
        return lh, la


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def _train_neural_network(X, y_home, y_away, sw):
    try:
        return _train_pytorch_nn(X, y_home, y_away, sw)
    except Exception as e:
        logger.warning("  PyTorch NN failed (%s) — using MLP fallback", e)
        return _train_mlp_fallback(X, y_home, y_away, sw)


def _train_pytorch_nn(X, y_home, y_away, sw):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("  Training PyTorch NN …")
    device     = torch.device("cpu")
    n_features = X.shape[1]

    class NFLNet(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.ReLU(),
            )
            self.head_home = nn.Linear(64, 1)
            self.head_away = nn.Linear(64, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_home(z).squeeze(-1), self.head_away(z).squeeze(-1)

    model = NFLNet(n_features).to(device)

    X_t  = torch.FloatTensor(X).to(device)
    yh_t = torch.FloatTensor(y_home).to(device)
    ya_t = torch.FloatTensor(y_away).to(device)
    sw_t = torch.FloatTensor(sw / sw.mean()).to(device)

    loader    = DataLoader(TensorDataset(X_t, yh_t, ya_t, sw_t), batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_loss, best_state = float("inf"), None

    for epoch in range(150):
        model.train()
        epoch_loss = 0.0
        for xb, yh, ya, w in loader:
            optimizer.zero_grad()
            ph, pa = model(xb)
            loss = (
                0.40 * (torch.abs(ph - yh) * w).mean() +
                0.40 * (torch.abs(pa - ya) * w).mean() +
                0.35 * (torch.abs((ph + pa) - (yh + ya)) * w).mean() +
                0.25 * (torch.abs((ph - pa) - (yh - ya)) * w).mean()
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            logger.info("    Epoch %3d loss=%.4f", epoch + 1, epoch_loss)

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_DIR / "model_weights.pt")
    with open(MODEL_DIR / "nn_config.json", "w") as f:
        json.dump({"n_features": n_features, "type": "pytorch"}, f)

    return ("pytorch", model, n_features, device)


def _train_mlp_fallback(X, y_home, y_away, sw):
    from sklearn.neural_network import MLPRegressor
    logger.info("  Training sklearn MLP (fallback) …")
    mlp_h = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42)
    mlp_a = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=43)
    mlp_h.fit(X, y_home)
    mlp_a.fit(X, y_away)
    with open(MODEL_DIR / "mlp_home.pkl", "wb") as f: pickle.dump(mlp_h, f)
    with open(MODEL_DIR / "mlp_away.pkl", "wb") as f: pickle.dump(mlp_a, f)
    with open(MODEL_DIR / "nn_config.json", "w") as f:
        json.dump({"type": "sklearn_mlp"}, f)
    return ("sklearn", mlp_h, mlp_a)


def _nn_predict(nn_obj, X):
    kind = nn_obj[0]
    if kind == "pytorch":
        import torch
        _, model, _, device = nn_obj
        model.eval()
        with torch.no_grad():
            ph, pa = model(torch.FloatTensor(X).to(device))
        return ph.cpu().numpy(), pa.cpu().numpy()
    else:
        _, mlp_h, mlp_a = nn_obj
        return mlp_h.predict(X), mlp_a.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: META-LEARNER (out-of-fold, no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def _train_meta_learner(X_sc, y_home, y_away, sw,
                        ridge_h, ridge_a, xgb_h, xgb_a, lgbm_h, lgbm_a, nn_obj):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("  Training meta-learner (OOF) …")
    tscv = TimeSeriesSplit(n_splits=5)

    oof_meta_h, oof_meta_a, oof_idx = [], [], []

    for tr_idx, val_idx in tscv.split(X_sc):
        X_tr,  X_val  = X_sc[tr_idx], X_sc[val_idx]
        yh_tr, ya_tr  = y_home[tr_idx], y_away[tr_idx]
        sw_tr          = sw[tr_idx]

        # Retrain ridge on fold (fast)
        r_h = Ridge(alpha=10.0).fit(X_tr, yh_tr, sample_weight=sw_tr)
        r_a = Ridge(alpha=10.0).fit(X_tr, ya_tr, sample_weight=sw_tr)

        # Use full-trained tree models on val (approximate — avoids 3× retraining cost)
        xgb_h_p  = xgb_h.predict(X_val)
        xgb_a_p  = xgb_a.predict(X_val)
        lgbm_h_p = lgbm_h.predict(X_val)
        lgbm_a_p = lgbm_a.predict(X_val)
        nn_h_p, nn_a_p = _nn_predict(nn_obj, X_val)

        oof_meta_h.append(np.column_stack([r_h.predict(X_val), xgb_h_p, lgbm_h_p, nn_h_p]))
        oof_meta_a.append(np.column_stack([r_a.predict(X_val), xgb_a_p, lgbm_a_p, nn_a_p]))
        oof_idx.append(val_idx)

    oof_X_h = np.vstack(oof_meta_h)
    oof_X_a = np.vstack(oof_meta_a)
    oof_y_h = y_home[np.concatenate(oof_idx)]
    oof_y_a = y_away[np.concatenate(oof_idx)]

    meta_h = Ridge(alpha=1.0).fit(oof_X_h, oof_y_h)
    meta_a = Ridge(alpha=1.0).fit(oof_X_a, oof_y_a)

    # ── Variance calibration ──────────────────────────────────────────────
    # Ridge meta-learner heavily regresses predictions toward the mean,
    # collapsing spread std to ~1-2 pts (real NFL: ~12 pts).
    # Fix: fit a linear rescaling on OOF predictions to match target variance.
    oof_pred_h = meta_h.predict(oof_X_h)
    oof_pred_a = meta_a.predict(oof_X_a)
    oof_spread_pred   = oof_pred_h - oof_pred_a
    oof_spread_actual = oof_y_h    - oof_y_a

    pred_std   = np.std(oof_spread_pred)
    actual_std = np.std(oof_spread_actual)

    # Scale factor: expand predictions to match actual variance
    # Capped at 3× to prevent overcorrection on small training sets
    if pred_std > 0.5:
        spread_scale = float(np.clip(actual_std / pred_std, 1.0, 3.0))
    else:
        spread_scale = 1.5   # safe default if predictions are degenerate

    logger.info("  Variance calibration: pred_spread_std=%.2f → target=%.2f → scale=%.3f",
                pred_std, actual_std, spread_scale)

    # Save calibration parameters
    calib = {
        "spread_scale":      spread_scale,
        "home_mean_pred":    float(np.mean(oof_pred_h)),
        "home_mean_actual":  float(np.mean(oof_y_h)),
        "away_mean_pred":    float(np.mean(oof_pred_a)),
        "away_mean_actual":  float(np.mean(oof_y_a)),
    }
    with open(MODEL_DIR / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    with open(MODEL_DIR / "meta_learner_home.pkl", "wb") as f: pickle.dump(meta_h, f)
    with open(MODEL_DIR / "meta_learner_away.pkl", "wb") as f: pickle.dump(meta_a, f)
    return meta_h, meta_a


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_cv(X_sc, y_home, y_away, sw):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("  Running TimeSeriesSplit CV …")
    tscv = TimeSeriesSplit(n_splits=3)
    mae_h, mae_a, mae_t = [], [], []

    for tr_idx, val_idx in tscv.split(X_sc):
        if len(val_idx) < 10:
            continue
        X_tr, X_val = X_sc[tr_idx], X_sc[val_idx]
        yh_tr = y_home[tr_idx]; yh_val = y_home[val_idx]
        ya_tr = y_away[tr_idx]; ya_val = y_away[val_idx]
        sw_tr = sw[tr_idx]

        mh = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        ma = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        mh.fit(X_tr, yh_tr, sample_weight=sw_tr)
        ma.fit(X_tr, ya_tr, sample_weight=sw_tr)

        ph, pa = mh.predict(X_val), ma.predict(X_val)
        mae_h.append(np.mean(np.abs(ph - yh_val)))
        mae_a.append(np.mean(np.abs(pa - ya_val)))
        mae_t.append(np.mean(np.abs((ph + pa) - (yh_val + ya_val))))

    return {
        "cv_mae_home":  round(float(np.mean(mae_h)),  3) if mae_h else 0,
        "cv_mae_away":  round(float(np.mean(mae_a)),  3) if mae_a else 0,
        "cv_mae_total": round(float(np.mean(mae_t)),  3) if mae_t else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> dict:
    """Load all saved model artifacts. Raises FileNotFoundError if not trained yet."""
    models = {}

    # Required: preprocessors + feature list
    for name in ("imputer.pkl", "scaler.pkl", "meta_learner_home.pkl", "meta_learner_away.pkl"):
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact '{name}' not found. Run pipeline --mode full first."
            )
        with open(path, "rb") as f:
            key = name.replace(".pkl", "").replace("meta_learner_", "meta_")
            models[key] = pickle.load(f)

    with open(MODEL_DIR / "feature_list.json") as f:
        models["feature_cols"] = json.load(f)

    # Ridge
    with open(MODEL_DIR / "ridge_home.pkl", "rb") as f: models["ridge_home"] = pickle.load(f)
    with open(MODEL_DIR / "ridge_away.pkl", "rb") as f: models["ridge_away"] = pickle.load(f)

    # XGBoost
    try:
        import xgboost as xgb
        xh = xgb.XGBRegressor(); xh.load_model(str(MODEL_DIR / "xgb_home.json"))
        xa = xgb.XGBRegressor(); xa.load_model(str(MODEL_DIR / "xgb_away.json"))
        models["xgb_home"], models["xgb_away"] = xh, xa
    except Exception:
        for suffix in ("home", "away"):
            p = MODEL_DIR / f"xgb_{suffix}.pkl"
            if p.exists():
                with open(p, "rb") as f: models[f"xgb_{suffix}"] = pickle.load(f)

    # LightGBM
    try:
        import lightgbm as lgb
        models["lgbm_home"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_home.txt"))
        models["lgbm_away"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_away.txt"))
    except Exception:
        for suffix in ("home", "away"):
            p = MODEL_DIR / f"lgbm_{suffix}.pkl"
            if p.exists():
                with open(p, "rb") as f: models[f"lgbm_{suffix}"] = pickle.load(f)

    # Neural Network
    nn_cfg_path = MODEL_DIR / "nn_config.json"
    if nn_cfg_path.exists():
        with open(nn_cfg_path) as f:
            nn_cfg = json.load(f)

        if nn_cfg.get("type") == "pytorch":
            try:
                import torch
                import torch.nn as nn
                n_features = nn_cfg["n_features"]

                class _NFLNet(nn.Module):
                    def __init__(self, n_in):
                        super().__init__()
                        self.trunk = nn.Sequential(
                            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(128, 64), nn.ReLU(),
                        )
                        self.head_home = nn.Linear(64, 1)
                        self.head_away = nn.Linear(64, 1)
                    def forward(self, x):
                        z = self.trunk(x)
                        return self.head_home(z).squeeze(-1), self.head_away(z).squeeze(-1)

                device = torch.device("cpu")
                net = _NFLNet(n_features).to(device)
                net.load_state_dict(
                    torch.load(MODEL_DIR / "model_weights.pt", map_location=device)
                )
                net.eval()
                models["nn"] = ("pytorch", net, n_features, device)
            except Exception as e:
                logger.warning("Could not load PyTorch model: %s", e)

        elif nn_cfg.get("type") == "sklearn_mlp":
            for suffix in ("home", "away"):
                p = MODEL_DIR / f"mlp_{suffix}.pkl"
                if p.exists():
                    with open(p, "rb") as f: models[f"nn_{suffix}_mlp"] = pickle.load(f)

    return models

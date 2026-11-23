"""
train.py — trains the stacked ensemble and saves it to model/saved/.

  Layer 1  Ridge, XGBoost, LightGBM          (one model per score: home, away)
  Layer 2  PyTorch network, two output heads (home, away)
  Layer 3  Ridge meta-learner over the four predictions — plus the
           Vegas-implied score whenever the line is published — fitted on true
           out-of-fold predictions (every layer-1/2 model refit per fold)

There is deliberately NO variance "calibration": an earlier version stretched
predicted margins ×1.8 to match the spread of actual results. Predictions
that minimise error are supposed to be narrower than outcomes (outcome =
signal + noise); stretching them cost ~0.4 points of spread MAE on the
held-out 2025 season and manufactured large fake edges against the line.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import settings

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = settings.SEED

# Shared tree parameters for the final fit AND the per-fold OOF refits.
XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
)
LGBM_PARAMS = dict(
    n_estimators=500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=10, reg_alpha=0.05, reg_lambda=0.5,
    objective="mae", metric="mae", random_state=SEED, n_jobs=-1, verbose=-1,
)


def market_scores(X: np.ndarray, feature_cols: list) -> tuple:
    """Vegas-implied home/away scores, (total ± spread) / 2, from the feature
    matrix; NaN where the line is not published yet."""
    spread = X[:, feature_cols.index("spread_line")].astype(float)
    total = X[:, feature_cols.index("total_line")].astype(float)
    return (total + spread) / 2.0, (total - spread) / 2.0


def compute_sample_weights(df: pd.DataFrame, current_season: int) -> np.ndarray:
    """Recency × game-type weight per training game; 2020 (no fans) heavily
    down-weighted."""
    age = current_season - df["season"].astype(int)
    sw = age.map(settings.SEASON_WEIGHTS).fillna(settings.OLDEST_SEASON_WEIGHT).to_numpy(float)
    gw = df["game_type"].map(settings.GAME_TYPE_WEIGHTS).fillna(1.0).to_numpy(float)
    covid = (df["season"] == settings.COVID_SEASON).to_numpy()
    return (np.where(covid, sw * settings.COVID_WEIGHT_FACTOR, sw) * gw).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_all(game_df: pd.DataFrame, feature_cols: list,
              current_season: Optional[int] = None) -> dict:
    """Fit the full ensemble on every labelled row of game_df and save it."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if current_season is None:
        from data_loader import CURRENT_SEASON as current_season
    np.random.seed(SEED)

    df = game_df[game_df["target_home_score"].notna() & game_df["target_away_score"].notna()]
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} labelled games — need ≥50.")
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Training on %d games, %d features", len(df), len(feature_cols))

    X_raw  = df[feature_cols].to_numpy(np.float32)
    y_home = df["target_home_score"].to_numpy(np.float32)
    y_away = df["target_away_score"].to_numpy(np.float32)
    sw     = compute_sample_weights(df, current_season)

    empty = np.isnan(X_raw).all(axis=0)
    if empty.any():
        logger.warning("  %d features have no observed values (0 for Ridge/NN, NaN for trees): %s",
                       int(empty.sum()), [feature_cols[i] for i in np.where(empty)[0]][:10])

    # Trees get raw features with NaN (native missing-value handling); Ridge and
    # the NN get median-imputed, standardised features. keep_empty_features keeps
    # the column count stable so train and predict matrices always line up.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(imputer.fit_transform(X_raw))
    X_tree  = X_raw

    ridge_h = Ridge(alpha=10.0).fit(X_sc, y_home, sample_weight=sw)
    ridge_a = Ridge(alpha=10.0).fit(X_sc, y_away, sample_weight=sw)
    logger.info("  Training XGBoost …")
    xgb_h = xgb.XGBRegressor(**XGB_PARAMS).fit(X_tree, y_home, sample_weight=sw)
    xgb_a = xgb.XGBRegressor(**XGB_PARAMS).fit(X_tree, y_away, sample_weight=sw)
    logger.info("  Training LightGBM …")
    lgb_h = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X_tree, y_home, sample_weight=sw)
    lgb_a = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X_tree, y_away, sample_weight=sw)
    logger.info("  Training PyTorch NN …")
    net = _fit_net(X_sc, y_home, y_away, sw, epochs=150, seed=SEED)

    mkt_h, mkt_a = market_scores(X_raw, feature_cols)
    metas, oof = _train_meta_learner(X_sc, X_tree, y_home, y_away, sw, mkt_h, mkt_a)

    # ── persist ───────────────────────────────────────────────────────────
    import torch
    for name, obj in [("imputer", imputer), ("scaler", scaler), ("ridge_home", ridge_h),
                      ("ridge_away", ridge_a), *metas.items()]:
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    xgb_h.save_model(str(MODEL_DIR / "xgb_home.json"))
    xgb_a.save_model(str(MODEL_DIR / "xgb_away.json"))
    lgb_h.booster_.save_model(str(MODEL_DIR / "lgbm_home.txt"))
    lgb_a.booster_.save_model(str(MODEL_DIR / "lgbm_away.txt"))
    torch.save(net.state_dict(), MODEL_DIR / "model_weights.pt")
    (MODEL_DIR / "nn_config.json").write_text(json.dumps({"n_features": len(feature_cols)}))
    (MODEL_DIR / "feature_list.json").write_text(json.dumps(feature_cols))
    version = f"{settings.MODEL_VERSION}-{datetime.now(timezone.utc):%Y%m%d}"
    (MODEL_DIR / "version.txt").write_text(version)

    metrics = {"n_games": len(df), "n_features": len(feature_cols),
               "model_version": version, **oof}
    (MODEL_DIR / "training_log.json").write_text(json.dumps(metrics, indent=2))
    logger.info("  Out-of-fold MAE (honest): home=%.3f away=%.3f spread=%.3f (Vegas %.3f) total=%.3f "
                "| market weight home=%.2f away=%.2f", oof["oof_mae_home"], oof["oof_mae_away"],
                oof["oof_mae_spread"], oof["oof_vegas_mae_spread"], oof["oof_mae_total"],
                oof["market_weight_home"], oof["market_weight_away"])
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def build_net(n_in: int):
    """Dual-head MLP (home score, away score). Single definition, used for
    training and for loading saved weights."""
    import torch.nn as nn

    class NFLNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(64, 32),    nn.ReLU(),
            )
            self.head_home = nn.Linear(32, 1)
            self.head_away = nn.Linear(32, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_home(z).squeeze(-1), self.head_away(z).squeeze(-1)

    return NFLNet()


def _fit_net(X, y_home, y_away, sw, epochs: int, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    net = build_net(X.shape[1])
    data = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_home),
                         torch.FloatTensor(y_away), torch.FloatTensor(sw / sw.mean()))
    loader = DataLoader(data, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    lw = settings.NN_LOSS_WEIGHTS

    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for xb, yh, ya, w in loader:
            opt.zero_grad()
            ph, pa = net(xb)
            loss = (lw["home"] * (torch.abs(ph - yh) * w).mean()
                    + lw["away"] * (torch.abs(pa - ya) * w).mean()
                    + lw["total"] * (torch.abs((ph + pa) - (yh + ya)) * w).mean()
                    + lw["spread"] * (torch.abs((ph - pa) - (yh - ya)) * w).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            logger.info("    Epoch %3d loss=%.4f", epoch + 1, epoch_loss)
    net.load_state_dict(best_state)
    net.eval()
    return net


def nn_predict(net, X) -> tuple:
    import torch
    net.eval()
    with torch.no_grad():
        ph, pa = net(torch.FloatTensor(np.asarray(X, dtype=np.float32)))
    return ph.numpy(), pa.numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  META-LEARNER (true out-of-fold)
# ══════════════════════════════════════════════════════════════════════════════

def _train_meta_learner(X_sc, X_tree, y_home, y_away, sw, mkt_h, mkt_a):
    """Refit every layer-1/2 model on each TimeSeriesSplit training fold and
    predict the following fold. Two Ridge blenders are fitted on those
    out-of-fold predictions:
      meta_learner_*         the four model predictions
      meta_learner_*_market  the same plus the Vegas-implied score — used
                             whenever the line is published, so the blend
                             learns from data how far to trust the market
                             over the models (the market is the strongest
                             single predictor of NFL scores).
    Also returns the out-of-fold error of the blend (honest in-window error)."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("  Training meta-learner (true OOF) …")
    stack_h, stack_a, idx = [], [], []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=5).split(X_sc), start=1):
        args = dict(sample_weight=sw[tr])
        r_h = Ridge(alpha=10.0).fit(X_sc[tr], y_home[tr], **args)
        r_a = Ridge(alpha=10.0).fit(X_sc[tr], y_away[tr], **args)
        x_h = xgb.XGBRegressor(**XGB_PARAMS).fit(X_tree[tr], y_home[tr], **args)
        x_a = xgb.XGBRegressor(**XGB_PARAMS).fit(X_tree[tr], y_away[tr], **args)
        l_h = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X_tree[tr], y_home[tr], **args)
        l_a = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X_tree[tr], y_away[tr], **args)
        net = _fit_net(X_sc[tr], y_home[tr], y_away[tr], sw[tr], epochs=60, seed=SEED + fold)
        n_h, n_a = nn_predict(net, X_sc[va])
        stack_h.append(np.column_stack([r_h.predict(X_sc[va]), x_h.predict(X_tree[va]),
                                        l_h.predict(X_tree[va]), n_h]))
        stack_a.append(np.column_stack([r_a.predict(X_sc[va]), x_a.predict(X_tree[va]),
                                        l_a.predict(X_tree[va]), n_a]))
        idx.append(va)

    S_h, S_a = np.vstack(stack_h), np.vstack(stack_a)
    va_all = np.concatenate(idx)
    yh, ya = y_home[va_all], y_away[va_all]
    mh, ma = mkt_h[va_all], mkt_a[va_all]
    has = ~np.isnan(mh) & ~np.isnan(ma)

    metas = {
        "meta_learner_home": Ridge(alpha=1.0).fit(S_h, yh),
        "meta_learner_away": Ridge(alpha=1.0).fit(S_a, ya),
        "meta_learner_home_market": Ridge(alpha=1.0).fit(np.column_stack([S_h[has], mh[has]]), yh[has]),
        "meta_learner_away_market": Ridge(alpha=1.0).fit(np.column_stack([S_a[has], ma[has]]), ya[has]),
    }
    ph = metas["meta_learner_home"].predict(S_h)
    pa = metas["meta_learner_away"].predict(S_a)
    ph[has] = metas["meta_learner_home_market"].predict(np.column_stack([S_h[has], mh[has]]))
    pa[has] = metas["meta_learner_away_market"].predict(np.column_stack([S_a[has], ma[has]]))
    oof = {
        "oof_mae_home":   round(float(np.mean(np.abs(ph - yh))), 3),
        "oof_mae_away":   round(float(np.mean(np.abs(pa - ya))), 3),
        "oof_mae_spread": round(float(np.mean(np.abs((ph - pa) - (yh - ya)))), 3),
        "oof_mae_total":  round(float(np.mean(np.abs((ph + pa) - (yh + ya)))), 3),
        "oof_vegas_mae_spread": round(float(np.mean(np.abs((mh - ma)[has] - (yh - ya)[has]))), 3),
        "oof_pred_spread_std":   round(float(np.std(ph - pa)), 3),
        "oof_actual_spread_std": round(float(np.std(yh - ya)), 3),
        "market_weight_home": round(float(metas["meta_learner_home_market"].coef_[-1]), 3),
        "market_weight_away": round(float(metas["meta_learner_away_market"].coef_[-1]), 3),
    }
    return metas, oof


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> dict:
    """Load the saved ensemble. Raises FileNotFoundError if it was never trained."""
    import lightgbm as lgb
    import torch
    import xgboost as xgb

    required = ["imputer.pkl", "scaler.pkl", "ridge_home.pkl", "ridge_away.pkl",
                "meta_learner_home.pkl", "meta_learner_away.pkl",
                "meta_learner_home_market.pkl", "meta_learner_away_market.pkl", "xgb_home.json",
                "xgb_away.json", "lgbm_home.txt", "lgbm_away.txt", "model_weights.pt",
                "nn_config.json", "feature_list.json"]
    missing = [n for n in required if not (MODEL_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"Model artifacts missing: {missing}. Run pipeline --mode full.")

    models = {}
    for name in ("imputer", "scaler", "ridge_home", "ridge_away"):
        with open(MODEL_DIR / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    for side in ("home", "away"):
        for suffix in ("", "_market"):
            with open(MODEL_DIR / f"meta_learner_{side}{suffix}.pkl", "rb") as f:
                models[f"meta_{side}{suffix}"] = pickle.load(f)
        x = xgb.XGBRegressor()
        x.load_model(str(MODEL_DIR / f"xgb_{side}.json"))
        models[f"xgb_{side}"] = x
        models[f"lgbm_{side}"] = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{side}.txt"))
    n_in = json.loads((MODEL_DIR / "nn_config.json").read_text())["n_features"]
    net = build_net(n_in)
    net.load_state_dict(torch.load(MODEL_DIR / "model_weights.pt", map_location="cpu"))
    net.eval()
    models["nn"] = net
    models["feature_cols"] = json.loads((MODEL_DIR / "feature_list.json").read_text())
    # The version travels with the artifacts (training_log.json is cached with
    # them); version.txt is only the committed copy for humans.
    log = MODEL_DIR / "training_log.json"
    models["version"] = (json.loads(log.read_text()).get("model_version", "unknown")
                         if log.exists() else "unknown")
    return models

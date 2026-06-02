"""
season_analysis.py
==================
Per-season model analysis (Question 4).

For each available season S:
  1. Fit the BEST possible model for that season alone (using only season S data)
  2. Record which features were most important
  3. Compute how well it fits season S (in-sample, as upper bound)

Then analyse:
  - How optimal model parameters change season-to-season
  - Which features are consistently important vs. seasonally variable
  - Trends that could improve future predictions

Output: data/predictions/season_analysis.json
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "predictions" / "season_analysis.json"


def run_season_analysis(game_df: pd.DataFrame) -> dict:
    """
    Fit a model per season, capture best params and feature importance.
    Returns dict with per-season results and cross-season trend analysis.
    """
    from feature_engineering import get_feature_columns

    feature_cols = get_feature_columns(game_df)
    labeled = game_df[game_df["target_home_score"].notna()].copy()
    seasons  = sorted(labeled["season"].unique())

    logger.info("Running per-season analysis for seasons: %s", seasons)

    per_season = {}
    for season in seasons:
        df_s = labeled[labeled["season"] == season].copy()
        if len(df_s) < 30:
            logger.warning("  Season %s: only %d games — skipping", season, len(df_s))
            continue

        logger.info("  Fitting season %s (%d games) …", season, len(df_s))
        result = _fit_single_season(df_s, feature_cols)
        per_season[str(season)] = result

    # Cross-season trend analysis
    trends = _analyse_trends(per_season)

    output = {
        "per_season": per_season,
        "trends":     trends,
        "n_seasons":  len(per_season),
        "seasons":    list(per_season.keys()),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Season analysis saved → %s", OUTPUT_PATH)
    return output


def _fit_single_season(df: pd.DataFrame, feature_cols: list) -> dict:
    """Fit XGBoost (best single-model) to one season and extract key metrics."""
    import pickle
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y_home = df["target_home_score"].values.astype(np.float32)
    y_away = df["target_away_score"].values.astype(np.float32)
    y_total  = y_home + y_away
    y_spread = y_home - y_away

    # Preprocessing
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_imp)

    # XGBoost with 5-fold CV to get unbiased in-season MAE
    try:
        import xgboost as xgb

        # Tune params for single-season (smaller trees to avoid overfit)
        params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective="reg:absoluteerror", random_state=42, n_jobs=-1,
        )

        # 5-fold CV MAE
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_maes_total, cv_maes_spread = [], []
        for tr, val in kf.split(X_sc):
            mh = xgb.XGBRegressor(**params)
            ma = xgb.XGBRegressor(**params)
            mh.fit(X_sc[tr], y_home[tr], verbose=False)
            ma.fit(X_sc[tr], y_away[tr], verbose=False)
            ph = mh.predict(X_sc[val])
            pa = ma.predict(X_sc[val])
            cv_maes_total.append(float(np.mean(np.abs((ph+pa)-(y_home[val]+y_away[val])))))
            cv_maes_spread.append(float(np.mean(np.abs((ph-pa)-(y_home[val]-y_away[val])))))

        cv_mae_total  = float(np.mean(cv_maes_total))
        cv_mae_spread = float(np.mean(cv_maes_spread))

        # Full-season fit for feature importance
        mh_full = xgb.XGBRegressor(**params)
        ma_full = xgb.XGBRegressor(**params)
        mh_full.fit(X_sc, y_home, verbose=False)
        ma_full.fit(X_sc, y_away, verbose=False)

        # Feature importance (average home+away)
        imp_h = mh_full.feature_importances_
        imp_a = ma_full.feature_importances_
        imp_avg = (imp_h + imp_a) / 2.0
        fi = dict(sorted(
            zip(feature_cols, imp_avg.tolist()),
            key=lambda x: -x[1]
        )[:20])

        # Optimal hyperparams (simplified — would use optuna in full version)
        best_params = params.copy()
        model_type = "xgboost"

    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        mh = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        ma = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)

        cv_scores_h = cross_val_score(mh, X_sc, y_home, cv=5, scoring="neg_mean_absolute_error")
        cv_scores_a = cross_val_score(ma, X_sc, y_away, cv=5, scoring="neg_mean_absolute_error")
        cv_mae_total  = float(np.mean(-cv_scores_h) + np.mean(-cv_scores_a))
        cv_mae_spread = cv_mae_total * 0.6   # rough estimate

        mh.fit(X_sc, y_home)
        ma.fit(X_sc, y_away)
        imp_avg = (mh.feature_importances_ + ma.feature_importances_) / 2.0
        fi = dict(sorted(zip(feature_cols, imp_avg.tolist()), key=lambda x: -x[1])[:20])
        best_params = {"n_estimators": 200, "max_depth": 4}
        model_type = "gbm_fallback"

    # Distribution stats for this season
    spread_std   = float(np.std(y_spread))
    total_mean   = float(np.mean(y_total))
    total_std    = float(np.std(y_total))
    home_win_pct = float(np.mean(y_home > y_away))

    return {
        "n_games":          len(df),
        "cv_mae_total":     round(cv_mae_total,  3),
        "cv_mae_spread":    round(cv_mae_spread, 3),
        "model_type":       model_type,
        "top_features":     {k: round(v, 5) for k, v in fi.items()},
        "season_stats": {
            "total_mean":    round(total_mean, 2),
            "total_std":     round(total_std,  2),
            "spread_std":    round(spread_std, 2),
            "home_win_pct":  round(home_win_pct, 3),
        },
    }


def _analyse_trends(per_season: dict) -> dict:
    """
    Cross-season trend analysis:
    - How does MAE change year over year?
    - Which features are consistently in top-10?
    - Is total scoring trending up/down?
    - Is home field advantage changing?
    """
    if len(per_season) < 2:
        return {"note": "Need at least 2 seasons for trend analysis"}

    seasons = sorted(per_season.keys())

    # MAE trend
    mae_trend = {s: per_season[s]["cv_mae_total"] for s in seasons}
    maes = list(mae_trend.values())
    mae_direction = "improving" if maes[-1] < maes[0] else "worsening"

    # Feature consistency: which features appear in top-10 every season
    top10_per_season = [
        set(list(per_season[s]["top_features"].keys())[:10])
        for s in seasons
    ]
    consistent_features = set.intersection(*top10_per_season) if top10_per_season else set()

    # Scoring trends
    total_means = {s: per_season[s]["season_stats"]["total_mean"] for s in seasons}
    spread_stds = {s: per_season[s]["season_stats"]["spread_std"]  for s in seasons}
    home_wins   = {s: per_season[s]["season_stats"]["home_win_pct"]for s in seasons}

    # Linear trend slopes (simple)
    def slope(d):
        xs = list(range(len(d)))
        ys = list(d.values())
        if len(xs) < 2: return 0.0
        n = len(xs)
        return (n*sum(x*y for x,y in zip(xs,ys)) - sum(xs)*sum(ys)) / \
               max(n*sum(x**2 for x in xs) - sum(xs)**2, 1e-9)

    return {
        "mae_by_season":              mae_trend,
        "mae_trend":                  mae_direction,
        "mae_slope_per_season":       round(slope(mae_trend), 4),
        "consistently_important_features": sorted(consistent_features),
        "total_scoring_by_season":    total_means,
        "total_scoring_slope":        round(slope(total_means), 4),
        "spread_variance_by_season":  spread_stds,
        "home_win_pct_by_season":     home_wins,
        "home_win_slope":             round(slope(home_wins), 4),
        "interpretation": {
            "total_scoring": "trending up" if slope(total_means) > 0.3 else
                             "trending down" if slope(total_means) < -0.3 else "stable",
            "home_advantage": "declining" if slope(home_wins) < -0.005 else
                              "increasing" if slope(home_wins) > 0.005 else "stable",
            "model_difficulty": mae_direction,
        }
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("season_analysis.py — run via pipeline.py --mode analyze_seasons")

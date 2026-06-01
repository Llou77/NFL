# 🏈 NFL Score Predictor 2026

A machine learning ensemble that predicts the final score of every NFL game — powering a live public dashboard at **[llou77.github.io/NFL](https://llou77.github.io/NFL/)**.

Every prediction simultaneously covers:
- **Moneyline** — predicted winner
- **ATS (Against The Spread)** — predicted margin vs. the book spread  
- **Over/Under** — predicted total vs. the book total
- **Confidence Score** — model certainty (0–100%) per game

---

## How It Works

The pipeline runs automatically every Tuesday via GitHub Actions:

1. **Data Fetch** — Downloads all available NFL data from nflverse (play-by-play, rosters, injuries, NGS, PFR advanced stats, draft picks, win totals, and more)
2. **Feature Engineering** — Computes ~150 features per game: rolling EPA metrics, QB efficiency, injury scores, Elo ratings, head-to-head history, weather, rest, and cross-season signals
3. **Training** — Fits a stacked ensemble (Ridge + XGBoost + LightGBM + PyTorch Neural Network) on a sliding 3-season window with Bayesian-optimized sample weights
4. **Prediction** — Generates predictions for all upcoming games with confidence scores and edge signals vs. the Vegas line
5. **Deploy** — Commits updated `predictions_latest.json` to the repo; GitHub Pages serves the dashboard automatically

## Model Architecture

| Layer | Model | Purpose |
|-------|-------|---------|
| 1A | Ridge Regression | Linear baseline, L2 regularization |
| 1B | XGBoost | Non-linear interactions, MAE objective |
| 1C | LightGBM | Faster gradient boosting alternative |
| 2  | Dual-Head PyTorch NN (256→128→64) | Jointly optimizes score, total, and spread |
| 3  | Ridge Meta-Learner | Optimal blend of all sub-model outputs |

## Data Sources

All free and publicly available:
- **[nflverse / nfl_data_py](https://nflverse.nflverse.com)** — PBP, schedules, rosters, injuries, NGS, PFR, FTN
- **[weatherapi.com](https://weatherapi.com)** — Game day weather for outdoor stadiums

## Sliding Window

The model always trains on exactly **816 regular season + 39 playoff games** (3 full seasons). After each game completes, it enters the window and the oldest game drops out. Season weights are automatically tuned by Bayesian optimization (Optuna).

## Confidence Score

Four components combined:
- **Model Agreement (40%)** — spread of sub-model predictions
- **Feature Completeness (30%)** — fraction of expected features available
- **H2H Data Quality (15%)** — historical matchup sample size
- **Injury Data Freshness (15%)** — hours before kickoff predictions are generated

Edge indicators are suppressed for WEAK confidence predictions (&lt;50%).

## Repository Structure

```
NFL/
├── .github/workflows/
│   ├── weekly_update.yml      # Runs every Tuesday 10:00 UTC
│   ├── thursday_update.yml    # Runs Wednesday night for TNF games
│   └── pre_season_train.yml   # Runs in August (Bayesian opt + full train)
├── data/
│   ├── raw/                   # Cached nflverse parquet files
│   ├── processed/             # Feature-engineered game tables
│   └── predictions/           # Output JSON files
├── model/
│   ├── pipeline.py            # Main orchestration
│   ├── data_loader.py         # nflverse data fetching (26 tables)
│   ├── feature_engineering.py # All feature computation (~150 features)
│   ├── feature_h2h.py         # Head-to-head history features
│   ├── train.py               # Ensemble training (Ridge/XGB/LGBM/NN)
│   ├── bayesian_optimizer.py  # Optuna weight optimization
│   ├── confidence.py          # Per-prediction confidence scoring
│   ├── predict.py             # Prediction generation + edge detection
│   ├── evaluate.py            # Backtesting + performance tracking
│   └── saved/                 # Trained model artifacts
└── docs/                      # GitHub Pages frontend
    ├── index.html             # Main prediction dashboard
    └── assets/
        ├── predictions_latest.json
        └── performance.json
```

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (fetch → features → train → predict)
cd model && python pipeline.py --mode full

# Bayesian weight optimization (run once pre-season)
cd model && python pipeline.py --mode optimize

# Historical backtesting
cd model && python pipeline.py --mode backtest
```

## Manual GitHub Actions Trigger

All three workflows support `workflow_dispatch` — you can run them manually from the **Actions** tab in GitHub.

---

*Predictions are for informational and entertainment purposes only. Not financial advice.*

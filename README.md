# NFL Score Predictor — 2026 Season

**🏈 Live site:** [llou77.github.io/NFL](https://llou77.github.io/NFL) · **Magyar leírás:** [README.hu.md](README.hu.md)

A machine learning model that predicts the final score of every NFL game before it is played. From the two predicted scores it derives the projected winner and margin, an over/under lean versus the Vegas total, and a confidence rating. Everything runs automatically on GitHub Actions and publishes to GitHub Pages.

Every term used below is defined in the [Glossary](#glossary). If you read nothing else, read [How a run works](#how-a-run-works-step-by-step) and [How accurate is it?](#how-accurate-is-it).

---

## How accurate is it?

**Honest status (July 2026): previously published accuracy figures were invalid and have been withdrawn.**

An internal audit found multiple *data leakage* channels — the model was seeing statistics from the very game it was predicting (same-game box scores, post-game Elo ratings, same-week tracking stats). That fully explained the implausibly strong old numbers. The leaks are closed (feature allowlist, two-phase Elo updates, lagged rolling stats, honest out-of-fold stacking, and a backtest that runs the exact production prediction path).

Realistic expectations, to be verified by each new backtest run:

| Metric | Honest expectation | Context |
|---|---|---|
| Average margin error (MAE) | ~10–11 points | Theoretical best for NFL is ~9.5–10; Vegas closing lines achieve ~10.2–10.5 |
| Edge ATS hit rate | 50–54% | 52.4% is break-even at standard -110 odds; anything sustained above that is real edge |
| ROI / CLV | tracked live | Flat-stake ROI and closing-line value are first-class backtest metrics |

> **On betting:** the site shows *edge signals* — games where the model meaningfully disagrees with the Vegas line. They are not betting advice. Whether there is genuine edge will only be known after a full season of live, timestamped predictions.

---

## How a run works, step by step

One full pipeline run (`model/pipeline.py --mode full`) does the following, in order:

1. **Load data** (`data_loader.py`). Data for *closed* seasons (everything before the season in progress) is read from the committed **frozen layer** in `data/frozen/` — zero network access, nothing to break. Only *live* data is downloaded: the schedule (upcoming games + final scores), betting lines, and current-season stat files once the season starts.
2. **Aggregate play-by-play** (`feature_engineering._aggregate_pbp`). Every play of every game is summarised into one row per team per game (offensive/defensive efficiency, pace, turnovers, red-zone rates, …). For closed seasons this aggregate is pre-computed and stored in the repo (`data/frozen/pbp_agg_*.parquet`); only current-season games are aggregated fresh.
3. **Build features** (`feature_engineering.build_all_features`). The team-game rows are expanded with rolling form windows (last 4/8/16 games, always **lagged** — a game never describes itself), Elo ratings (updated in two phases so both teams see strictly pre-game values), QB form, injuries, referee tendencies (prior games only), Next Gen Stats (lagged), rest days, weather from the schedule feed, travel/primetime context, and previous-season priors. Rows are then pivoted to **one row per game** (home vs away). A strict **allowlist** decides which ~370 columns the model may see — anything that could leak the game's own outcome is dropped and logged.
4. **Head-to-head features** (`feature_h2h.py`). Computed from historical game *scores* in the schedule table (never from play-by-play), with matchup-type-dependent lookbacks and roster-continuity damping.
5. **Reconcile performance** (`evaluate.update_performance_from_latest`). Before anything is overwritten, previously published predictions are matched against games completed since, and `performance.json` (the site's live track record) is updated.
6. **Train the ensemble** (`train.py`). See [Model architecture](#model-architecture). All artifacts are saved to `model/saved/`.
7. **Generate predictions** (`predict.py`). Upcoming games are run through the exact same `ensemble_predict()` path the backtest uses (no train/serve skew), calibrated, scored for confidence, decorated with edge signals, and written to `data/predictions/predictions_latest.json`.
8. **Publish**. Outputs are copied to `docs/assets/`, which GitHub Pages serves to the site.

The **backtest** (`--mode backtest`) is separate: for each test season it retrains the full ensemble on the seasons before it, predicts the held-out season, and reports MAE / ATS / over-under / ROI / CLV. The **optimizer** (`--mode optimize`) is also separate and pre-season only: Optuna tunes the 8 sample-weight parameters (how much each training season and game type counts).

---

## Model architecture

A three-layer *stacked ensemble* — several models whose outputs are blended:

```
Layer 1:  Ridge Regression  +  XGBoost  +  LightGBM     (three independent score predictors)
Layer 2:  PyTorch neural network (dual head: home score, away score)
Layer 3:  Ridge meta-learner — blends the four predictions (trained on true out-of-fold data)
          → variance calibration → final predicted score
```

- **Preprocessing:** tree models (XGBoost, LightGBM) receive raw features with missing values preserved — they handle NaN natively. Ridge and the neural network receive median-imputed, standardised features. Columns with no observed values are kept (zero-filled for Ridge/NN, NaN for trees) so the matrix width never changes between training and prediction.
- **Honest stacking:** the meta-learner is trained on out-of-fold (OOF) predictions where every layer-1 model — including the neural net — is *refit per fold*, so the blender never sees a model predicting its own training data.
- **Calibration:** the meta-learner compresses predicted spreads toward the mean; a variance scale factor (data-driven, clipped to [1, 3]) restores realistic spread width.
- **Training window:** the last 4 completed seasons + the season in progress, retrained weekly in season. Season/game-type sample weights come from the Bayesian optimizer. The current season year is derived from the clock (`data_loader.get_current_season`), never hand-edited.

---

## Automation — the workflows

All automation lives in `.github/workflows/`. Every heavy stage is a **separate job well under 20 minutes**, individually re-runnable from the Actions UI. Each stage commits a public log to `data/logs/*_last.txt`, so outcomes are inspectable without Actions access.

| Workflow | When it runs | What it does |
|---|---|---|
| `weekly_update.yml` | Tuesdays 10:00 UTC (cron) | Full run (steps 1–8 above). An **offseason guard** exits in ~30 s when no NFL games fall within [-3, +10] days. |
| `thursday_update.yml` | Wednesdays 22:00 UTC (cron) | Predict-only refresh before Thursday Night Football, using cached model artifacts (falls back to training on cache miss). Same guard. |
| `full_pipeline.yml` | push touching `.full-trigger`, or manual | Staged chain: optimize → train+predict → **parallel per-season backtests** → merge & publish. |
| `optimize_run.yml` | push touching `.opt-trigger`, or manual | Stage 1 alone: Bayesian weight optimisation (~8–10 min). |
| `train_run.yml` | push touching `.train-trigger`, or manual | Stage 2 alone: train + predict (~8–12 min). |
| `backtest_run.yml` | push touching `.backtest-trigger`, or manual | Stage 3 alone: backtest matrix (one season per parallel job) + merge. |
| `freeze_data.yml` | push touching `.freeze-trigger`, manual, or Apr 1 yearly | Rebuilds `data/frozen/` — downloads all closed-season data once and commits it (back to `FREEZE_FROM`, default 2009). |
| `walkforward.yml` | push touching `.walkforward-trigger`, or manual | Deep walk-forward simulation: one parallel job per (test season 2013–2025 × window 3/4), nested tuning per fold, merged results + weight-drift chart published to [docs/walkforward.html](https://llou77.github.io/NFL/walkforward.html). |
| `pre_season_train.yml` | first Tuesday of August | Dispatches `full_pipeline.yml` for the new season. |
| `season_analysis.yml` | manual | Per-season difficulty/trend analysis for the site. |

To start any push-triggered workflow without touching the Actions UI:

```bash
date > .full-trigger && git add .full-trigger && git commit -m "run full pipeline" && git push
```

---

## The frozen data layer

Between seasons — and for every finished season — historical statistics can never change again. So they are downloaded **once** by `scripts/freeze_data.py` and committed to `data/frozen/`:

- per-season stat tables (player stats, rosters, snap counts, injuries, depth charts, FTN charting, PFR advanced stats) for every closed season;
- full-history reference tables (Next Gen Stats, draft, combine, officials, betting lines, team/player IDs) as offline fallbacks;
- `pbp_agg_{season}.parquet` — the per-season team-game aggregate of raw play-by-play, built with the *same function* the live pipeline uses. Raw play-by-play (~25 MB/season) is deliberately **not** committed; the aggregate is ~100× smaller and is all the pipeline needs.

Normal runs therefore need the network only for the schedule, betting lines and current-season files. If nflverse renames or removes a historical file (it happened in 2025), nothing breaks. Re-run the freeze once a year after the Super Bowl — the April 1st cron does it automatically.

---

## Repository map

```
model/
  pipeline.py            entry point — modes: full / predict_only / backtest / optimize / analyze_seasons
  data_loader.py         downloads + caches all data; frozen-first loading; season constants
  feature_engineering.py PBP aggregation + all feature construction + the allowlist
  feature_h2h.py         head-to-head features from historical schedule scores
  train.py               the 3-layer ensemble training + artifact save/load
  predict.py             the single shared prediction path (live + backtest)
  evaluate.py            backtests, ATS/ROI/CLV metrics, live performance reconciliation
  bayesian_optimizer.py  Optuna search over the 8 sample-weight parameters
  confidence.py          confidence rating (model agreement / data completeness / H2H sample)
  season_analysis.py     per-season difficulty analysis
  player_ratings.py      DISABLED pending temporal-shift rework (leak risk)
  walkforward.py         one walk-forward fold: nested tuning → train → score a held-out season
scripts/
  freeze_data.py         builds data/frozen/ (run in CI, yearly)
  season_guard.py        offseason guard — "run"/"skip" for scheduled workflows
  merge_backtests.py     merges per-season backtest fragments into backtest_results.json
  merge_walkforward.py   aggregates walk-forward folds; weight-drift chart
data/
  frozen/                committed, sealed historical data (see above)
  raw/, processed/       gitignored scratch caches
  predictions/           committed model outputs (site data source)
  logs/                  committed last-run logs of every workflow stage
docs/                    GitHub Pages site (index.html + assets/*.json)
```

---

## Glossary

**Betting / NFL terms**

- **Spread (point spread):** the bookmaker's handicap for the favourite. A spread of -3.5 means the favourite is expected to win by more than 3.5.
- **ATS (against the spread):** a team "covers" if it beats the spread. *Edge ATS hit rate* = how often the model's disagreement with the spread turned out right.
- **Over/Under (total):** the bookmaker's line for the combined final score; you can bet the actual total goes over or under it.
- **-110 odds:** standard US pricing — bet 110 to win 100. Implies a 52.4% hit rate just to break even.
- **Closing line:** the final betting line before kickoff — the market's most informed price.
- **CLV (closing line value):** how much better the line you (or the model) acted on was than the closing line. Positive average CLV is the strongest known predictor of long-term betting profit.
- **ROI (flat-stake):** profit per unit staked assuming an identical bet on every edge signal at -110.
- **Moneyline win:** simply picking the game's winner (no spread involved).

**Data terms**

- **PBP (play-by-play):** one row per play of every game, ~370 columns (source: nflverse/nflfastR).
- **EPA (expected points added):** how much a single play shifted the team's expected points — the standard efficiency currency of NFL analytics.
- **WEPA (weighted EPA):** EPA with plays re-weighted by how *predictive* they are of future performance (garbage time down-weighted, normal passes up-weighted, per nfelo research).
- **Success rate:** share of plays with positive EPA.
- **NGS (Next Gen Stats):** NFL tracking data (receiver separation, time to throw, …). Used lagged by one week.
- **PFR / FTN:** Pro Football Reference advanced stats; FTN manual charting data.
- **Snap counts / depth charts:** who actually played and their roster position — inputs to injury/continuity features.
- **Frozen layer:** the committed `data/frozen/` snapshot of all closed-season data (never re-downloaded).

**Modeling terms**

- **Feature:** one input column the model sees (e.g. "home team's offensive EPA/play over its last 8 games").
- **Allowlist:** the explicit list of feature name patterns the model is *permitted* to see. Anything not on it is dropped — the safe inverse of trying to blocklist known leaks.
- **Data leakage:** any path by which information from the predicted game's own outcome reaches the model at training or prediction time. Leakage inflates test accuracy and destroys live accuracy.
- **Lagged / rolling window (r4/r8/r16):** averages over the previous 4/8/16 games, shifted by one game so the current game is never included in its own features.
- **Elo:** a running strength rating updated after each game from the result and margin. Two-phase update = both teams' *pre-game* ratings are recorded before either is updated.
- **Imputation:** filling missing values (median here) for models that cannot handle NaN (Ridge, NN). Trees receive genuine NaN instead.
- **Ensemble / stacking / meta-learner:** several models predict independently; a small final model (the meta-learner) learns the best blend of their outputs.
- **OOF (out-of-fold):** predictions made by a model on data it was not trained on, produced via cross-validation — the only honest inputs for training a meta-learner.
- **TimeSeriesSplit:** cross-validation that always trains on the past and validates on the future (no shuffling), matching how the model is used in reality.
- **Backtest:** simulating the past — train only on seasons before X, predict season X, compare with what happened.
- **Bayesian optimisation (Optuna):** a guided search that proposes promising parameter combinations instead of trying all of them. Here it tunes 8 sample-weight parameters on a train/validation season split.
- **Sample weights:** how much each training game counts (recent seasons count more; playoff game types count differently).
- **Variance calibration:** rescaling predicted spreads so their distribution width matches reality (blended models regress toward the mean).
- **MAE (mean absolute error):** average absolute miss, in points.
- **Confidence rating (High/Medium/Low/Weak):** *not* a win probability — a measure of how much the model trusts this prediction, from sub-model agreement (55%), input data completeness (30%) and H2H sample size (15%).
- **Edge signal:** shown only when confidence is Medium+ **and** the model disagrees with the Vegas line by 3+ points.

---

## Data sources

All free and public: [nflverse](https://github.com/nflverse) release files (play-by-play, player stats, rosters, snap counts, injuries, depth charts, FTN, PFR, Next Gen Stats, players, contracts, draft, combine), [Lee Sharpe / nfldata](https://github.com/nflverse/nfldata) (games & schedule incl. stadium/weather fields, win totals, officials, scoring lines), [mrcaseb/nfl-data](https://github.com/mrcaseb/nfl-data) (historical betting lines), [DynastyProcess](https://github.com/dynastyprocess/data) (player ID map).

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · Optuna · GitHub Actions · GitHub Pages

---

*Built by [@llou77](https://github.com/llou77) · Predictions update automatically every Tuesday (and Wednesday night) during the NFL season.*

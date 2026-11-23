# NFL Score Predictor

**🏈 Live site:** [llou77.github.io/NFL](https://llou77.github.io/NFL) · **Magyar leírás:** [README.hu.md](README.hu.md)

A machine-learning model that predicts the final score of every NFL game before it is played. From the two predicted scores it derives the winner, the margin and the total, compares them with the Vegas line, and shows a lean when the model and the line differ by 3+ points. Everything runs on GitHub Actions and publishes to GitHub Pages.

Terms are defined in the [Glossary](#glossary).

---

## How accurate is it?

**The model has not beaten the Vegas line.** Treat it as a statistics project, not a betting tool.

| | Margin (spread) error | Total error | Spread record, all games | Edge games (3+ pts) |
|---|---|---|---|---|
| **Live 2026, weeks 1–10** (model v1.1, before the audit) | 11.11 pts — Vegas 9.62 | 11.22 — Vegas 10.51 | 67–80 (45.6%) | 43–53 (44.8%), ROI −14.5% |
| **Held-out 2025, old model as it actually ran live** (v1.1) | 10.53 — Vegas 9.67 | 11.05 — Vegas 10.42 | 50.5% | 48.3% (176 games) |
| **Held-out 2025, model v2.0** (trained on 2022–24) | **9.74** — Vegas 9.67 | **10.46** — Vegas 10.42 | 51.9% | 50.0% (12 games) |
| **2026 weeks 7–11, model v2.0** (trained on games before Oct 15) | **9.28** — Vegas 9.10 | **10.15** — Vegas 10.20 | — | 2 games |

Multi-season numbers (every season since 2013, each trained only on the four seasons before it) are on the site's [walk-forward page](https://llou77.github.io/NFL/walkforward.html).

The **November 2026 audit** (see [WORKLOG.md](WORKLOG.md)) found why the live numbers were worse than the backtests had promised, and fixed it:

- **Predictions were stretched ×1.8** to match the spread of actual results. Good predictions are supposed to be narrower than outcomes (outcome = signal + luck); the stretch alone cost ~0.4 points of margin error and manufactured large fake "edges".
- **34 features existed for every historical game but were always missing for upcoming games**: rolling starting-QB form and Next Gen Stats (merged by the week of the game, which has no data until it is played), game-day weather and referee crews (published after the game). Backtests saw them; live predictions never did.
- **Head-to-head features were constant for every game** (the team columns they needed were dropped earlier in the pipeline), and several other features were dead or wrong: preseason win totals (source ended in 2020), "opening line" and market features (source ended in 2021 — they were copies of the current line), draft-class quality (always 0), the QB-change flag (right only 59% of the time), and the dome flag (wrong for 8 teams).
- **The Bayesian weight optimiser was tuning noise**: across 26 nested walk-forward folds, none of the 8 tuned weights could be told apart from a uniform random draw over its search range. It was removed; the weights are now fixed constants.

Model v2.0 uses only features that exist both for past games and for the next game, with no variance stretching, and its final blender sees the Vegas-implied score whenever the line is published. Fitted out-of-fold, that blender gives the market a weight of ≈1.0–1.1 — i.e. **the model essentially follows the market and adds small corrections**, which is the honest behaviour for a model without a proven edge, and it now flags a 3+ point "edge" in ~4% of games instead of ~65%. Its live record starts with week 12 of 2026 and is tracked separately from v1.1 (`performance.json → by_model_version`). Whether it has any real edge will only be known after a season of live, timestamped predictions — the git history of `predictions_latest.json` is that timestamp.

> **On betting:** leans are not betting advice. At standard −110 odds you need 52.4% to break even; one season's hit rate carries several points of noise.

---

## How a run works

`model/pipeline.py --mode full` (every Tuesday), in order:

1. **Load data** (`data_loader.py`). Closed seasons come from the committed **frozen layer** in `data/frozen/` (no network). Only the schedule, the season in progress and the Next Gen Stats files are downloaded.
2. **Aggregate play-by-play** (`feature_engineering._aggregate_pbp`): one row per team per game (efficiency, success rate, third downs, red zone, turnovers, pace, …). Closed seasons are pre-aggregated in `data/frozen/pbp_agg_*.parquet`.
3. **Build features** (`feature_engineering.build_all_features`): rolling form over the team's last 4/8/16 played games, starting-QB form, Next Gen Stats form, Elo (computed over every game since 1999), QB/coach changes, roster continuity, rest, calendar, home-field advantage, the Vegas spread and total; pivoted to one row per game. An **allowlist** decides which ~330 columns the model may see.
4. **Head-to-head features** (`feature_h2h.py`) from earlier meetings' final scores.
5. **Reconcile the live record** (`evaluate.update_performance_from_latest`): the predictions published last time are scored against the games completed since and stored in `performance.json`, before anything is overwritten.
6. **Train** (`train.py`) the ensemble — see [Model architecture](#model-architecture).
7. **Predict** (`predict.py`) every remaining game through `ensemble_predict()`, the same function the backtest uses; add the confidence score and the leans.
8. **Publish**: copy the JSON files to `docs/assets/`, which GitHub Pages serves.

The Wednesday-night run (`--mode predict_only`) reuses the Tuesday model with fresh lines before Thursday Night Football. `--mode backtest --backtest-season S` trains on the 4 seasons before S and scores S; `model/walkforward.py` does the same for every season since 2013 from frozen data only.

### Feature rules

A column becomes a feature only if it is (1) knowable before kickoff and (2) **available for upcoming games when the weekly job runs**. Rolling statistics are computed on each team's full game timeline: a game's value is the mean over the team's last N *played* games, so next week's game sees exactly the window a historical game saw in training. Game-day weather, referee assignments and final injury statuses fail rule 2 (they appear after the Tuesday/Wednesday runs) and are not used; the starting quarterback for next week *is* published in the schedule and is used.

---

## Model architecture

```
Layer 1:  Ridge  +  XGBoost  +  LightGBM          one model each for home and away score
Layer 2:  PyTorch network with two heads          (home score, away score)
Layer 3:  Ridge meta-learner over the four predictions (+ the Vegas-implied score when
          the line is published), fitted on true out-of-fold data
```

- Trees receive raw features with missing values; Ridge and the network receive median-imputed, standardised features.
- The meta-learner is trained on out-of-fold predictions: every layer-1/2 model is refit on each time-ordered fold. Two blenders are fitted: one with the Vegas-implied home/away score ((total ± spread) / 2) as an extra input, used whenever the line is out, and one without it for games further ahead. The data decides how much to trust the market; the fitted weight is logged with every training run.
- No variance calibration (see the audit notes above). The out-of-fold error is logged after every training run.
- Training window: the last 4 completed seasons + the season in progress, retrained weekly in season. Sample weights: 1.2 (current season), 1.0, 0.8, 0.6 (oldest); playoff games 0.6–0.75×; 2020 down-weighted (no fans). All constants live in `model/settings.py`.

---

## Automation — the workflows

| Workflow | When | What |
|---|---|---|
| `weekly_update.yml` | Tuesday 10:23 UTC | Full run. Exits in ~30 s in the offseason (no game within [−3, +10] days). |
| `thursday_update.yml` | Wednesday 21:41 UTC | Predict-only refresh before TNF with the Tuesday model (retrains if it is missing). |
| `full_pipeline.yml` | manual, or a push touching `.full-trigger` | Train + predict → backtest of the two last completed seasons in parallel → merge. |
| `train_run.yml` | manual, or `.train-trigger` | Train + predict only. |
| `backtest_run.yml` | manual, or `.backtest-trigger` | Backtest matrix + merge only. |
| `walkforward.yml` | manual, or `.walkforward-trigger` | One job per held-out season since 2013 (4-season window) + summary and chart for `docs/walkforward.html`. |
| `freeze_data.yml` | April 1 yearly, manual, or `.freeze-trigger` | Re-seals `data/frozen/` with the season that just ended. |
| `pre_season_train.yml` | first Tuesday of August | Starts `full_pipeline.yml` for the new season. |

Scheduled times are deliberately off the hour: GitHub delays (or drops) scheduled runs at :00, and the old 10:00 slot started 6–9 hours late all season. Every stage commits its console tail to `data/logs/*_last.txt`, and commits are pushed with `scripts/push_with_retry.sh`, which fails the job if the push never lands.

Push-trigger example, without the Actions UI:

```bash
date > .full-trigger && git add .full-trigger && git commit -m "run full pipeline" && git push
```

---

## The frozen data layer

Finished seasons never change, so `scripts/freeze_data.py` downloads them once and commits them to `data/frozen/` (~24 MB):

- weekly player stats and weekly rosters for every closed season since 2009;
- full-history Next Gen Stats files and a schedule snapshot;
- `pbp_agg_{season}.parquet`, the per-season team-game aggregate of play-by-play, built with the same function the live path uses (raw play-by-play is ~25 MB/season and is not committed).

---

## Repository map

```
model/
  pipeline.py            entry point — modes: full / predict_only / backtest
  settings.py            every fixed model constant (weights, Elo, thresholds, version)
  data_loader.py         downloads + caches the tables; frozen-first loading
  feature_engineering.py play-by-play aggregation, all features, the allowlist
  feature_h2h.py         head-to-head features from earlier meetings
  train.py               ensemble training + saving/loading
  predict.py             the single prediction path (live, backtest, walk-forward)
  evaluate.py            backtest metrics and the live track record
  confidence.py          confidence score (model agreement / completeness / H2H)
  walkforward.py         one walk-forward fold from frozen data
scripts/
  freeze_data.py         builds data/frozen/
  season_guard.py        offseason guard for the scheduled workflows
  merge_backtests.py     merges per-season backtest fragments
  merge_walkforward.py   summarises walk-forward folds; chart
  push_with_retry.sh     bot push with rebase-and-retry
data/
  frozen/                committed historical data (above)
  predictions/           published outputs (the site's data source)
  logs/                  last console log of every workflow stage
docs/                    GitHub Pages site (index.html, walkforward.html, assets/)
```

---

## Glossary

**Betting terms**

- **Spread / spread_line:** the bookmaker's expected margin. In this repo it is always home-oriented and **positive when the home team is favoured** (nflverse convention); the home team covers if home − away > spread.
- **ATS (against the spread):** the model "picks" the side its predicted margin favours relative to the line, in every game. Push = no decision.
- **Over/Under (total):** the bookmaker's line for the combined score; the model picks over or under in every game.
- **Edge game / lean:** model and line differ by **3+ points**. Leans are hidden on the site for LOW/WEAK confidence.
- **−110 odds:** bet 110 to win 100; 52.4% breaks even. **ROI** = profit per unit staked, flat 1-unit bets on edge games.

**Data terms**

- **PBP (play-by-play):** one row per play (nflverse / nflfastR). **EPA:** expected points added by a play. **WEPA:** EPA with plays re-weighted by how predictive they are (garbage time down, normal passes up; after nfelo). **Success rate:** share of plays with positive EPA.
- **NGS (Next Gen Stats):** NFL tracking data (time to throw, separation, rushing yards over expected).
- **Frozen layer:** the committed `data/frozen/` snapshot of finished seasons.

**Modelling terms**

- **Feature / allowlist:** an input column; only allowlisted columns reach the model.
- **Data leakage:** any path by which the predicted game's own outcome reaches the model. **Train/serve skew:** a feature that looks different (here: missing) at prediction time than in training.
- **Rolling window (r4/r8/r16):** mean over the team's previous 4/8/16 played games.
- **Elo:** a running strength rating updated after every game; both teams' pre-game ratings are recorded before either is updated.
- **OOF (out-of-fold):** predictions on data the model was not trained on; the meta-learner is fitted on them.
- **Backtest / walk-forward:** train only on seasons before X, predict X, compare with what happened (walk-forward repeats this for every season since 2013).
- **MAE:** mean absolute error, in points. **Margin error** = MAE of home − away.
- **Confidence (HIGH/MEDIUM/LOW/WEAK):** not a win probability — sub-model agreement (55%), data completeness (30%), head-to-head sample (15%).

---

## Data sources

All free and public: [nflverse](https://github.com/nflverse) release files (play-by-play, weekly player stats, weekly rosters, Next Gen Stats) and the [nflverse/nfldata](https://github.com/nflverse/nfldata) games file (scores, Vegas lines, roof, starting quarterbacks, head coaches).

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · GitHub Actions · GitHub Pages

---

*Built by [@llou77](https://github.com/llou77) · Predictions update every Tuesday and Wednesday night during the NFL season.*

# NFL Score Predictor — 2026 Season

**🏈 Live site:** [llou77.github.io/NFL](https://llou77.github.io/NFL)

---

## What is this?

A machine learning model that predicts the final score of every NFL game before it's played. It outputs a predicted score for both teams — and from those two numbers, it automatically derives:

- Who it thinks will win (and by how much)
- Whether the predicted total is over or under the Vegas line
- A confidence score for each prediction

The site updates automatically every Tuesday during the season with fresh predictions for the upcoming week.

---

## How accurate is it?

The model was tested on the **2025 NFL season** — a full season of games it had never seen before:

| Metric | Result | What it means |
|---|---|---|
| Average score error | **±6.9 points** | The predicted combined score was within ~7 points of the real score, on average |
| Average margin error | **±4.0 points** | The predicted winning margin was within ~4 points of reality, on average |
| Predictive fit (R²) | **0.61** | The model explains 61% of actual score variation — well above random |
| Over/Under prediction rate | **74.7%** | When the model says the total will go over or under the Vegas line, it's been right ~75% of the time |

For reference: a coin flip would give you 50%. Professional sports analysts typically achieve 55–60% on over/under picks.

> **On betting:** The site shows "edge signals" — games where the model's prediction meaningfully disagrees with the Vegas line. These are not betting recommendations. They're the games where the model is most confident it sees something the market may have missed. Whether there's genuine betting edge will only be known after a full season of live predictions.

---

## What does it look at?

About 150 different signals per game, grouped into four categories:

**Recent team performance** (computed over the last 4, 8, and 16 games)
How efficiently a team is moving the ball, stopping the run, creating turnovers, converting third downs, scoring in the red zone, and winning.

**Matchup advantages**
The model directly compares offensive strengths against defensive weaknesses. If Team A has an elite passing attack and Team B gives up a lot of yards through the air, that mismatch is explicitly measured.

**Game context**
Home/away, days of rest (coming off a bye vs. a short week), weather conditions (temperature, wind speed, dome vs. outdoor), primetime games, division rivalries, and referee tendencies.

**Off-season and preseason signals**
Power ratings that update continuously throughout the season. Vegas preseason win totals (a strong signal of team quality before any games are played). Whether the team changed its starting quarterback or head coach. Roster continuity vs. the previous year.

---

## How does the model work?

It uses a "stacked ensemble" — multiple models whose outputs are blended together.

```
Layer 1:  Ridge Regression  +  XGBoost  +  LightGBM
              ↓                   ↓             ↓
Layer 2:          Neural Network (home score + away score heads)
              ↓                   ↓             ↓
Layer 3:             Meta-Learner (optimal blend)
              ↓
          Final score prediction
```

**Layer 1** — Three very different model types each make independent predictions. Linear regression gives a simple baseline. XGBoost and LightGBM are gradient boosting algorithms that capture more complex patterns.

**Layer 2** — A neural network processes the same features and outputs two numbers: a predicted home score and a predicted away score. It's trained to simultaneously minimise errors on the score, the winning margin, and the combined total.

**Layer 3** — A final blending model learns the optimal way to combine all the predictions from layers 1 and 2. It's trained on held-out data (games not used to train the other models) to prevent it from just memorising the training set.

### Why multiple models instead of just one?

Different models are good at different things. Linear regression handles stable, consistent patterns well. Gradient boosting handles non-linear interactions (e.g. "rest advantage matters more in cold weather"). The neural network can capture subtler patterns across many features simultaneously. The blend is more reliable than any single model.

---

## Training setup

The model trains on a rolling window of the **last 3 NFL seasons** (~855 games total). Each week during the season, it retrains from scratch on all available data including the most recent completed games.

Season weights are not fixed — they're tuned every pre-season using Bayesian optimisation (a systematic search algorithm). The most recent season gets the highest weight, and the oldest gets the least. This reflects the reality that NFL teams change significantly year to year.

**Automated pipeline:** Every Tuesday at 10am UTC, a GitHub Actions workflow fetches updated statistics, retrains the model, generates predictions, and publishes them to this site. No manual intervention needed.

---

## Where does the data come from?

All data sources are free and publicly available:

| Source | What it provides |
|---|---|
| [nflverse / nfl_data_py](https://github.com/nflverse) | Schedules, rosters, box scores, play-by-play data |
| nflfastR | Full play-by-play with EPA (efficiency) calculations |
| Next Gen Stats | NFL tracking data — separation, time to throw, etc. |
| Pro Football Reference | Advanced career statistics |
| FTN Data | Manual charting of routes, coverage, and blocking |
| weatherapi.com | Game-day weather conditions |

---

## What are the edge signals on the predictions page?

On each game card, you may see labels like:

- **Total OVER +3.2** — the model predicts the combined score will be 3.2 points higher than the Vegas over/under line
- **Total UNDER −4.8** — the model predicts the combined score will be 4.8 points lower than the line
- **Home value +2.1** — the model thinks the home team is undervalued by ~2 points vs. the Vegas spread
- **Away value −1.8** — the model thinks the away team is undervalued

These only appear when:
1. The model's confidence rating is MEDIUM or higher
2. The disagreement with the Vegas line is large enough to be meaningful (3+ points)

If neither condition is met, no edge signal is shown.

---

## Confidence ratings

Every prediction has a confidence rating: **High / Medium / Low / Weak**

This is not a win probability. It measures how much the model trusts its own prediction given data quality and internal agreement.

| Component | Weight | What it measures |
|---|---|---|
| Sub-model agreement | 40% | How closely all four sub-models agree on the score |
| Data completeness | 35% | How much of the expected input data was available |
| Head-to-head sample size | 25% | How many times these two teams have played historically |

Early in the season (weeks 1–4), confidence tends to be lower because there's less recent game data available. Edge signals are suppressed for predictions with low or insufficient-data confidence.

---

## Known limitations

- **No in-game or injury adjustments:** Predictions are made before kickoff and don't update if a star player is ruled out on game day. Injury report freshness is factored into the confidence score, but not the prediction itself.
- **Rare situations:** The model has never seen a team with a historically bad/good start, so early-season outliers may be poorly predicted until more data accumulates.
- **Vegas lines are hard to beat:** The Vegas spread is set by professional oddsmakers with access to enormous amounts of information. Consistently outperforming it is extremely difficult. The model's edge signal hit rate will be tracked live during the 2026 season.

---

## Tech stack

Python · scikit-learn · XGBoost · LightGBM · PyTorch · Optuna · GitHub Actions · GitHub Pages

---

*Built by [@llou77](https://github.com/llou77) · Updated automatically every Tuesday during the NFL season*

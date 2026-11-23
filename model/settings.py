"""
settings.py — every fixed model constant in one place.

These used to live in model/saved/optimal_weights.json and were partly
"tuned" by an Optuna stage. The 2026-11 audit retired that stage: across the
26 nested walk-forward folds, none of the eight tuned sample weights could be
told apart from a uniform random draw over its search range
(Kolmogorov-Smirnov p = 0.13-0.89), so the tuning was fitting noise. The
values below are plain, monotonic defaults; changing them is a model change
and should be validated with the backtest and walk-forward workflows.
"""

# Bump the major version whenever features, targets or calibration change, so
# the live track record (performance.json) can be split by model version.
MODEL_VERSION = "v2.0"

SEED = 42

# Training sample weights by season, relative to the season being predicted
# (S = that season; S-4 and older fall back to the oldest weight).
SEASON_WEIGHTS = {0: 1.20, 1: 1.00, 2: 0.80, 3: 0.60}
OLDEST_SEASON_WEIGHT = 0.60
GAME_TYPE_WEIGHTS = {"REG": 1.00, "WC": 0.70, "DIV": 0.75, "CON": 0.75, "SB": 0.60}
COVID_SEASON = 2020
COVID_WEIGHT_FACTOR = 0.15   # no fans in 2020: home-field advantage collapsed

# Elo
ELO_START = 1500.0
ELO_K = 20.0
ELO_REGRESSION = 0.67        # share of the rating kept across a season boundary
ELO_MOV_MULT = 2.2

# Matchup gaps: offensive EPA is weighted sqrt(w) and defensive EPA 1/sqrt(w)
OFF_EPA_WEIGHT = 1.6

# Neural-network multi-task loss weights
NN_LOSS_WEIGHTS = {"home": 0.40, "away": 0.40, "total": 0.35, "spread": 0.25}

# Confidence score (see confidence.py)
CONF_WEIGHTS = {"model_agreement": 0.55, "feature_completeness": 0.30, "h2h": 0.15}
CONF_MODEL_DIVISOR = 10.0    # sub-model std (points) at which agreement hits 0
CONF_THRESHOLDS = {"HIGH": 0.74, "MEDIUM": 0.58, "LOW": 0.44}   # below LOW → WEAK

# Edge signals: the model must disagree with the Vegas number by at least
# this many points before the site shows a lean, and the backtest and live
# tracker count an "edge game" with the same rule.
SPREAD_EDGE_PTS = 3.0
TOTAL_EDGE_PTS = 3.0

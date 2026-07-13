"""
confidence.py
=============
Per-game confidence score (0-1) for every prediction.

Key fix: injury_data_freshness is NOT penalized pre-season.
Pre-season predictions are made >7 days before all games, which is expected
and should not drag down confidence. Only in-season same-week predictions
benefit from injury freshness uplift.

Recalibrated thresholds:
  The original thresholds (MEDIUM≥0.65) were too strict given pre-season
  h2h_data_quality defaults (0.2 for non-division). New thresholds are
  calibrated to produce a realistic distribution across the season.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component weights — simulation-calibrated
W_MODEL_AGREEMENT      = 0.55  # most reliable: sub-model std-dev
W_FEATURE_COMPLETENESS = 0.30  # data quality
W_H2H_QUALITY          = 0.15  # historical sample size

FRESHNESS_BONUS_MAX = 0.04

# Recalibrated thresholds — verified against actual pre-season scenarios:
# Division game week 1:      ~0.82 → HIGH ✓
# Conference game week 1:    ~0.77 → HIGH ✓  (was MEDIUM at old 0.78 threshold)
# Cross-conf game week 1:    ~0.72 → MEDIUM ✓
# Cross-conf game week 8:    ~0.77 → HIGH ✓  (was MEDIUM)
LABEL_THRESHOLDS = {
    "HIGH":   0.74,   # was 0.78 — too strict for non-division pre-season games
    "MEDIUM": 0.58,   # was 0.62
    "LOW":    0.44,   # was 0.48
    # below 0.44 → WEAK
}

HIGH_IMPORTANCE_FEATURES = [
    "home_off_epa_per_play_r8", "away_off_epa_per_play_r8",
    "home_def_epa_per_play_r8", "away_def_epa_per_play_r8",
    "home_off_cpoe_r8",         "away_off_cpoe_r8",
    "home_elo_pre_game",        "away_elo_pre_game",
    "home_qb_available",        "away_qb_available",
    "home_off_availability_score","away_off_availability_score",
    "home_off_pass_epa_r8",     "away_off_pass_epa_r8",
    "home_def_pass_epa_r8",     "away_def_pass_epa_r8",
]


def compute_confidence(
    game_row: pd.Series,
    sub_model_predictions: Optional[dict] = None,
    generation_timestamp: Optional[datetime] = None,
    kickoff_timestamp: Optional[datetime] = None,
) -> dict:
    if generation_timestamp is None:
        generation_timestamp = datetime.now(timezone.utc)

    # Load weights dynamically from Bayesian optimizer
    from bayesian_optimizer import load_weights as _lw
    _w = _lw()
    w_model  = float(_w.get("conf_w_model",   W_MODEL_AGREEMENT))
    w_feat   = float(_w.get("conf_w_feat",    W_FEATURE_COMPLETENESS))
    w_h2h    = float(_w.get("conf_w_h2h",     W_H2H_QUALITY))
    div      = float(_w.get("conf_model_div", 10.0))
    t_high   = float(_w.get("conf_high_thr",  LABEL_THRESHOLDS["HIGH"]))
    t_med    = float(_w.get("conf_med_thr",   LABEL_THRESHOLDS["MEDIUM"]))

    # Normalise component weights
    total_w = w_model + w_feat + w_h2h
    if total_w > 0:
        w_model /= total_w; w_feat /= total_w; w_h2h /= total_w

    # Component 1: Model Agreement (uses dynamic divisor)
    model_agreement = _compute_model_agreement(sub_model_predictions, divisor=div)

    # Component 2: Feature Completeness
    feature_completeness = _compute_feature_completeness(game_row)

    # Component 3: H2H Data Quality
    h2h_quality = float(np.clip(game_row.get("h2h_data_confidence", 0.5), 0.0, 1.0))

    # Weighted score with dynamic weights
    score = (
        w_model * model_agreement      +
        w_feat  * feature_completeness +
        w_h2h   * h2h_quality
    )

    # Injury freshness BONUS only
    freshness = _compute_injury_freshness(generation_timestamp, kickoff_timestamp)
    bonus = FRESHNESS_BONUS_MAX * max(0.0, freshness - 0.60)
    score = float(np.clip(score + bonus, 0.0, 1.0))

    # Dynamic thresholds
    if   score >= t_high:                   label = "HIGH"
    elif score >= t_med:                    label = "MEDIUM"
    elif score >= LABEL_THRESHOLDS["LOW"]:  label = "LOW"
    else:                                   label = "WEAK"

    return {
        "confidence_score": round(score, 4),
        "confidence_label": label,
        "confidence_breakdown": {
            "model_agreement":       round(model_agreement,       4),
            "feature_completeness":  round(feature_completeness,  4),
            "h2h_data_quality":      round(h2h_quality,           4),
            "injury_data_freshness": round(freshness,             4),
        },
    }


def compute_confidence_batch(
    game_df: pd.DataFrame,
    all_sub_predictions: Optional[dict] = None,
    generation_timestamp: Optional[datetime] = None,
) -> pd.DataFrame:
    if generation_timestamp is None:
        generation_timestamp = datetime.now(timezone.utc)

    scores, labels, breakdowns = [], [], []

    for _, row in game_df.iterrows():
        game_id  = row.get("game_id")
        sub_preds = all_sub_predictions.get(game_id) if all_sub_predictions else None

        kickoff = None
        gd = row.get("game_date")
        if gd is not None and pd.notna(gd):
            try:
                kickoff = pd.to_datetime(gd).to_pydatetime()
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
            except Exception:
                pass

        result = compute_confidence(row, sub_preds, generation_timestamp, kickoff)
        scores.append(result["confidence_score"])
        labels.append(result["confidence_label"])
        breakdowns.append(result["confidence_breakdown"])

    game_df = game_df.copy()
    game_df["confidence_score"] = scores
    game_df["confidence_label"] = labels
    bd_df = pd.DataFrame(breakdowns, index=game_df.index)
    bd_df.columns = [f"conf_{c}" for c in bd_df.columns]
    return pd.concat([game_df, bd_df], axis=1)


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_model_agreement(sub_preds: Optional[dict], divisor: float = 10.0) -> float:
    if not sub_preds or len(sub_preds) < 2:
        return 0.72

    home_preds = [v[0] for v in sub_preds.values() if v is not None]
    away_preds = [v[1] for v in sub_preds.values() if v is not None]
    if len(home_preds) < 2:
        return 0.72

    avg_std = (np.std(home_preds) + np.std(away_preds)) / 2.0
    # divisor: if models disagree by divisor pts on average → score ≈ 0
    return float(np.clip(1.0 - avg_std / divisor, 0.0, 1.0))


def _compute_feature_completeness(row: pd.Series) -> float:
    total_w = 0.0
    avail_w = 0.0

    for feat in HIGH_IMPORTANCE_FEATURES:
        w = 2.0
        total_w += w
        val = row.get(feat)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            avail_w += w

    for col in row.index:
        if col in HIGH_IMPORTANCE_FEATURES:
            continue
        if col.startswith(("home_", "away_")) and "_r" in col:
            w = 1.0
            total_w += w
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                avail_w += w

    if total_w == 0:
        return 0.6

    penalty = 0.0
    if not row.get("is_dome", 0) and pd.isna(row.get("wind", np.nan)):
        penalty += 0.02

    return float(np.clip(avail_w / total_w - penalty, 0.0, 1.0))


def _compute_injury_freshness(
    generation_time: datetime,
    kickoff_time: Optional[datetime],
) -> float:
    if kickoff_time is None:
        return 0.50   # unknown → neutral

    if generation_time.tzinfo is None:
        generation_time = generation_time.replace(tzinfo=timezone.utc)
    if kickoff_time.tzinfo is None:
        kickoff_time = kickoff_time.replace(tzinfo=timezone.utc)

    hours_before = (kickoff_time - generation_time).total_seconds() / 3600.0

    if hours_before < 0:    return 1.00
    elif hours_before < 24: return 1.00
    elif hours_before < 48: return 0.92
    elif hours_before < 72: return 0.82
    elif hours_before < 120:return 0.68
    elif hours_before < 168:return 0.55
    else:                   return 0.42


def _score_to_label(score: float) -> str:
    if   score >= LABEL_THRESHOLDS["HIGH"]:   return "HIGH"
    elif score >= LABEL_THRESHOLDS["MEDIUM"]: return "MEDIUM"
    elif score >= LABEL_THRESHOLDS["LOW"]:    return "LOW"
    else:                                      return "WEAK"


def should_show_edge(confidence_label: str) -> bool:
    return confidence_label in ("HIGH", "MEDIUM")

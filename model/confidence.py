"""
confidence.py
=============
Computes a per-game confidence score (0-1) for every prediction.

Components:
  1. Model Agreement Score   (40%) — how much sub-models agree
  2. Feature Completeness    (30%) — what fraction of features are available
  3. H2H Data Quality        (15%) — sample size of historical matchup data
  4. Injury Data Freshness   (15%) — how current the injury/roster data is

Output fields added to predictions:
  confidence_score       float [0, 1]
  confidence_label       str   HIGH / MEDIUM / LOW / WEAK
  confidence_breakdown   dict  {model_agreement, feature_completeness,
                                h2h_data_quality, injury_data_freshness}
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component weights
W_MODEL_AGREEMENT      = 0.40
W_FEATURE_COMPLETENESS = 0.30
W_H2H_QUALITY          = 0.15
W_INJURY_FRESHNESS     = 0.15

# Label thresholds
LABEL_THRESHOLDS = {
    "HIGH":   0.80,
    "MEDIUM": 0.65,
    "LOW":    0.50,
    # below 0.50 → WEAK
}

# Features considered "high importance" (penalized more if missing)
HIGH_IMPORTANCE_FEATURES = [
    "home_off_epa_per_play_r8", "away_off_epa_per_play_r8",
    "home_def_epa_per_play_r8", "away_def_epa_per_play_r8",
    "home_off_cpoe_r8",         "away_off_cpoe_r8",
    "home_elo_pre_game",        "away_elo_pre_game",
    "home_qb_available",        "away_qb_available",
    "home_off_availability_score", "away_off_availability_score",
    "home_off_pass_epa_r8",     "away_off_pass_epa_r8",
    "home_def_pass_epa_r8",     "away_def_pass_epa_r8",
]


def compute_confidence(
    game_row: pd.Series,
    sub_model_predictions: Optional[dict] = None,
    generation_timestamp: Optional[datetime] = None,
    kickoff_timestamp: Optional[datetime] = None,
) -> dict:
    """
    Compute confidence score for a single game prediction.

    Parameters
    ----------
    game_row : Series with all feature columns for this game
    sub_model_predictions : dict mapping model_name → (pred_home, pred_away)
                            e.g. {'xgb': (24, 17), 'lgbm': (22, 18), 'nn': (23, 16)}
    generation_timestamp : when the prediction was generated (UTC)
    kickoff_timestamp : when the game kicks off (UTC)

    Returns
    -------
    dict with confidence_score, confidence_label, confidence_breakdown
    """
    if generation_timestamp is None:
        generation_timestamp = datetime.now(timezone.utc)

    # ── Component 1: Model Agreement ─────────────────────────────────────
    model_agreement = _compute_model_agreement(sub_model_predictions)

    # ── Component 2: Feature Completeness ─────────────────────────────────
    feature_completeness = _compute_feature_completeness(game_row)

    # ── Component 3: H2H Data Quality ─────────────────────────────────────
    h2h_quality = float(game_row.get("h2h_data_confidence", 0.5))
    h2h_quality = np.clip(h2h_quality, 0.0, 1.0)

    # ── Component 4: Injury Data Freshness ────────────────────────────────
    injury_freshness = _compute_injury_freshness(
        generation_timestamp, kickoff_timestamp
    )

    # ── Combined score ─────────────────────────────────────────────────────
    score = (
        W_MODEL_AGREEMENT      * model_agreement      +
        W_FEATURE_COMPLETENESS * feature_completeness +
        W_H2H_QUALITY          * h2h_quality          +
        W_INJURY_FRESHNESS     * injury_freshness
    )
    score = float(np.clip(score, 0.0, 1.0))

    label = _score_to_label(score)

    return {
        "confidence_score": round(score, 4),
        "confidence_label": label,
        "confidence_breakdown": {
            "model_agreement":      round(model_agreement, 4),
            "feature_completeness": round(feature_completeness, 4),
            "h2h_data_quality":     round(h2h_quality, 4),
            "injury_data_freshness":round(injury_freshness, 4),
        },
    }


def compute_confidence_batch(
    game_df: pd.DataFrame,
    all_sub_predictions: Optional[dict] = None,
    generation_timestamp: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Apply compute_confidence to an entire DataFrame of games.

    Parameters
    ----------
    game_df : DataFrame with feature columns + kickoff time
    all_sub_predictions : dict mapping game_id → sub_model_predictions dict
    generation_timestamp : when the batch was generated

    Returns
    -------
    game_df with confidence columns appended
    """
    if generation_timestamp is None:
        generation_timestamp = datetime.now(timezone.utc)

    scores, labels, breakdowns = [], [], []

    for _, row in game_df.iterrows():
        game_id = row.get("game_id")
        sub_preds = None
        if all_sub_predictions and game_id in all_sub_predictions:
            sub_preds = all_sub_predictions[game_id]

        # Kickoff time
        kickoff = None
        if "game_date" in row and pd.notna(row["game_date"]):
            try:
                kickoff = pd.to_datetime(row["game_date"]).to_pydatetime()
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

    # Expand breakdown dict into columns
    bd_df = pd.DataFrame(breakdowns, index=game_df.index)
    bd_df.columns = [f"conf_{c}" for c in bd_df.columns]
    game_df = pd.concat([game_df, bd_df], axis=1)

    return game_df


# ── Private helpers ────────────────────────────────────────────────────────────

def _compute_model_agreement(sub_preds: Optional[dict]) -> float:
    """
    Measure how much sub-models agree. Returns [0, 1].
    Based on std-dev of predicted home + away scores across models.
    """
    if not sub_preds or len(sub_preds) < 2:
        return 0.75   # default when only one model is available

    home_preds = [v[0] for v in sub_preds.values() if v is not None]
    away_preds = [v[1] for v in sub_preds.values() if v is not None]

    if len(home_preds) < 2:
        return 0.75

    std_home = np.std(home_preds)
    std_away = np.std(away_preds)
    avg_std  = (std_home + std_away) / 2.0

    # Divisor of 14 ≈ 2 TDs — if models disagree by 2 TDs, score ≈ 0
    agreement = 1.0 - (avg_std / 14.0)
    return float(np.clip(agreement, 0.0, 1.0))


def _compute_feature_completeness(row: pd.Series) -> float:
    """
    Check what fraction of important features are non-null.
    High-importance features have larger penalty when missing.
    """
    total_weight = 0.0
    available_weight = 0.0

    # Check high-importance features (weight = 2)
    for feat in HIGH_IMPORTANCE_FEATURES:
        w = 2.0
        total_weight += w
        val = row.get(feat)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            available_weight += w

    # Check all other numeric features (weight = 1)
    for col in row.index:
        if col in HIGH_IMPORTANCE_FEATURES:
            continue
        if col.startswith(("home_", "away_")) and "_r" in col:
            w = 1.0
            total_weight += w
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                available_weight += w

    if total_weight == 0:
        return 0.5

    # Additional penalties for known missing scenarios
    penalty = 0.0
    if row.get("home_qb_available", 1.0) < 0.5:
        penalty += 0.04   # QB status uncertain
    if row.get("away_qb_available", 1.0) < 0.5:
        penalty += 0.04
    if row.get("wind") is None or (isinstance(row.get("wind"), float) and np.isnan(row.get("wind", np.nan))):
        if not row.get("is_dome", 0):
            penalty += 0.03   # weather unknown for outdoor game

    score = (available_weight / total_weight) - penalty
    return float(np.clip(score, 0.0, 1.0))


def _compute_injury_freshness(
    generation_time: datetime,
    kickoff_time: Optional[datetime],
) -> float:
    """
    Score based on how many hours before kickoff the prediction was generated.
    NFL injury reports: Wed (est), Thu (full), Fri (final), Game day (final).
    """
    if kickoff_time is None:
        return 0.60   # unknown kickoff — medium freshness

    # Ensure timezone awareness
    if generation_time.tzinfo is None:
        generation_time = generation_time.replace(tzinfo=timezone.utc)
    if kickoff_time.tzinfo is None:
        kickoff_time = kickoff_time.replace(tzinfo=timezone.utc)

    hours_before_kickoff = (kickoff_time - generation_time).total_seconds() / 3600.0

    if hours_before_kickoff < 0:
        return 1.0    # game already started / completed — full data available
    elif hours_before_kickoff < 24:
        return 1.00   # same day — final injury report available
    elif hours_before_kickoff < 48:
        return 0.90   # 1-2 days out — final or near-final report
    elif hours_before_kickoff < 72:
        return 0.80   # 2-3 days out — Thursday report available
    elif hours_before_kickoff < 120:
        return 0.65   # 3-5 days out — Wednesday report available
    elif hours_before_kickoff < 168:
        return 0.50   # 5-7 days out — preliminary only
    else:
        return 0.40   # > 1 week out — no injury report yet


def _score_to_label(score: float) -> str:
    if score >= LABEL_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= LABEL_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    elif score >= LABEL_THRESHOLDS["LOW"]:
        return "LOW"
    else:
        return "WEAK"


def should_show_edge(confidence_label: str) -> bool:
    """Return True if edge indicators should be shown for this confidence level."""
    return confidence_label in ("HIGH", "MEDIUM")


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    test_row = pd.Series({
        "h2h_data_confidence": 0.8,
        "home_qb_available": 1.0,
        "away_qb_available": 0.7,
        "home_off_epa_per_play_r8": 0.15,
        "away_off_epa_per_play_r8": -0.05,
        "home_elo_pre_game": 1520.0,
        "away_elo_pre_game": 1490.0,
        "is_dome": 0,
        "wind": 12.0,
    })
    sub_preds = {
        "xgb":  (24, 17),
        "lgbm": (22, 18),
        "nn":   (23, 16),
    }
    now     = datetime.now(timezone.utc)
    kickoff = now + timedelta(hours=30)

    result = compute_confidence(test_row, sub_preds, now, kickoff)
    print(f"Confidence score : {result['confidence_score']:.4f}")
    print(f"Label            : {result['confidence_label']}")
    print(f"Breakdown        : {result['confidence_breakdown']}")

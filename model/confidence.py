"""
confidence.py — how much the model trusts a prediction (0-1). NOT a win
probability.

  model agreement       how closely the four sub-models agree
  feature completeness  share of the model's inputs that are present
  H2H data quality      how many earlier meetings back the H2H features

Labels: HIGH / MEDIUM / LOW / WEAK (thresholds in settings.py). Leans are
suppressed on the site for LOW and WEAK. Whether the tiers actually separate
accurate from inaccurate predictions is tracked live in performance.json
(summary.confidence_breakdown) — through week 10 of 2026 they did not.
"""

import numpy as np
import pandas as pd

import settings

# Inputs that matter most for a game's prediction; missing ones count double.
KEY_FEATURES = [
    "home_off_epa_per_play_r8", "away_off_epa_per_play_r8",
    "home_def_epa_per_play_r8", "away_def_epa_per_play_r8",
    "home_off_pass_epa_r8", "away_off_pass_epa_r8",
    "home_def_pass_epa_r8", "away_def_pass_epa_r8",
    "home_off_cpoe_r8", "away_off_cpoe_r8",
    "home_qb_epa_per_att_r8", "away_qb_epa_per_att_r8",
    "home_elo_pre_game", "away_elo_pre_game", "spread_line", "total_line",
]


def compute_confidence_batch(df: pd.DataFrame, sub_predictions: dict) -> pd.DataFrame:
    """Add confidence_score, confidence_label and conf_* component columns."""
    w = settings.CONF_WEIGHTS
    total_w = sum(w.values())
    rolling_cols = [c for c in df.columns if c.startswith(("home_", "away_"))
                    and c.endswith(("_r4", "_r8", "_r16")) and c not in KEY_FEATURES]
    key_cols = [c for c in KEY_FEATURES if c in df.columns]

    agreement = df["game_id"].map(lambda g: _model_agreement(sub_predictions.get(g)))
    present_key = df[key_cols].notna().sum(axis=1) * 2.0
    present_roll = df[rolling_cols].notna().sum(axis=1) * 1.0
    completeness = (present_key + present_roll) / max(1.0, 2.0 * len(key_cols) + len(rolling_cols))
    h2h = df.get("h2h_data_confidence", pd.Series(0.2, index=df.index)).clip(0, 1)

    score = (w["model_agreement"] * agreement + w["feature_completeness"] * completeness
             + w["h2h"] * h2h) / total_w
    t = settings.CONF_THRESHOLDS
    label = np.select([score >= t["HIGH"], score >= t["MEDIUM"], score >= t["LOW"]],
                      ["HIGH", "MEDIUM", "LOW"], "WEAK")

    df = df.copy()
    df["confidence_score"] = score.round(4)
    df["confidence_label"] = label
    df["conf_model_agreement"] = agreement.round(4)
    df["conf_feature_completeness"] = completeness.round(4)
    df["conf_h2h_data_quality"] = h2h.round(4)
    return df


def _model_agreement(sub: dict | None) -> float:
    """1 − (mean std-dev of the sub-model scores ÷ divisor), clipped to [0, 1]."""
    if not sub or len(sub) < 2:
        return 0.5
    home = [v[0] for v in sub.values()]
    away = [v[1] for v in sub.values()]
    spread = (np.std(home) + np.std(away)) / 2.0
    return float(np.clip(1.0 - spread / settings.CONF_MODEL_DIVISOR, 0.0, 1.0))

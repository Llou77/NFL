"""
evaluate.py — backtests and the live track record.

Definitions used everywhere (backtest, walk-forward, live tracker, site):
  spread_line   home-oriented, POSITIVE when the home team is favoured; the
                home team covers iff (home − away) > spread_line.
  ATS           the model "picks" the side its margin favours relative to
                the line, on EVERY game with a line; push = no decision.
  O/U           same idea for the total, on every game with a total.
  edge game     |model − line| ≥ settings.SPREAD_EDGE_PTS (totals:
                TOTAL_EDGE_PTS); these are the leans shown on the site.
  ROI           flat 1-unit bets at −110 on edge games.
Every model metric is reported next to the same metric for the Vegas line
itself, which is the bar a useful model has to clear.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import settings

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PRED_DIR  = ROOT / "data" / "predictions"
PERF_PATH = PRED_DIR / "performance.json"

# Live predictions published before the 2026-11 audit carry no version tag.
PRE_AUDIT_VERSION = "v1.1"
WIN_PAYOUT = 100.0 / 110.0


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(game_df: pd.DataFrame, test_season: int, n_train_seasons: int = 4) -> dict:
    """Train on the `n_train_seasons` seasons before `test_season` (the same
    window production uses), predict every game of `test_season` through the
    production prediction path, and score it."""
    from feature_engineering import get_feature_columns
    from predict import ensemble_predict
    from train import load_models, train_all

    labeled = game_df[game_df["target_home_score"].notna()]
    train_seasons = sorted(s for s in labeled["season"].unique()
                           if test_season - n_train_seasons <= s < test_season)
    df_train = labeled[labeled["season"].isin(train_seasons)]
    df_test = labeled[labeled["season"] == test_season]
    if len(df_train) < 150 or len(df_test) < 50:
        raise ValueError(f"Backtest {test_season}: not enough games "
                         f"(train={len(df_train)}, test={len(df_test)}).")
    logger.info("Backtest: train %s → test %d (%d / %d games)",
                train_seasons, test_season, len(df_train), len(df_test))

    train_all(df_train, get_feature_columns(game_df), current_season=test_season)
    models = load_models()
    _, _, home, away = ensemble_predict(models, df_test[models["feature_cols"]].to_numpy())
    result = compute_metrics(np.round(home), np.round(away), df_test)
    result.update({"test_season": test_season, "train_seasons": train_seasons,
                   "model_version": models["version"]})
    logger.info("  %d: spread MAE %.2f (Vegas %.2f) | total MAE %.2f (Vegas %.2f) | "
                "ATS %.1f%% | edge ATS %.1f%% (n=%d)", test_season,
                result["mae_spread"], result["vegas_mae_spread"], result["mae_total"],
                result["vegas_mae_total"], 100 * result["ats_pct"],
                100 * result["edge_ats_pct"], result["edge_ats_n"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _record(picks: np.ndarray, outcome: np.ndarray) -> tuple:
    """picks/outcome: +1 / −1 sides (0 = no pick / push). Returns (wins, losses)."""
    decided = (picks != 0) & (outcome != 0)
    wins = int(np.sum(decided & (picks == outcome)))
    return wins, int(np.sum(decided)) - wins


def compute_metrics(pred_home, pred_away, df: pd.DataFrame) -> dict:
    """Accuracy vs actual results, side-by-side with the Vegas line."""
    ph, pa = np.asarray(pred_home, float), np.asarray(pred_away, float)
    ah = df["target_home_score"].to_numpy(float)
    aa = df["target_away_score"].to_numpy(float)
    line = pd.to_numeric(df["spread_line"], errors="coerce").to_numpy(float)
    total = pd.to_numeric(df["total_line"], errors="coerce").to_numpy(float)
    pm, am, pt, at = ph - pa, ah - aa, ph + pa, ah + aa
    hs, ht = ~np.isnan(line), ~np.isnan(total)

    s_edge, t_edge = pm - line, pt - total
    s_pick, s_out = np.sign(np.nan_to_num(s_edge)), np.sign(np.nan_to_num(am - line))
    t_pick, t_out = np.sign(np.nan_to_num(t_edge)), np.sign(np.nan_to_num(at - total))
    s_big = hs & (np.abs(np.nan_to_num(s_edge)) >= settings.SPREAD_EDGE_PTS)
    t_big = ht & (np.abs(np.nan_to_num(t_edge)) >= settings.TOTAL_EDGE_PTS)

    ats_w, ats_l = _record(s_pick[hs], s_out[hs])
    edge_w, edge_l = _record(s_pick[s_big], s_out[s_big])
    ou_w, ou_l = _record(t_pick[ht], t_out[ht])
    eou_w, eou_l = _record(t_pick[t_big], t_out[t_big])
    pct = lambda w, l: round(w / (w + l), 4) if w + l else None  # noqa: E731
    roi = lambda w, l: round((w * WIN_PAYOUT - l) / (w + l), 4) if w + l else None  # noqa: E731

    out = {
        "n_games": int(len(df)),
        "mae_home": round(float(np.mean(np.abs(ph - ah))), 3),
        "mae_away": round(float(np.mean(np.abs(pa - aa))), 3),
        "mae_spread": round(float(np.mean(np.abs(pm - am))), 3),
        "mae_total": round(float(np.mean(np.abs(pt - at))), 3),
        "vegas_mae_spread": round(float(np.mean(np.abs(line[hs] - am[hs]))), 3) if hs.any() else None,
        "vegas_mae_total": round(float(np.mean(np.abs(total[ht] - at[ht]))), 3) if ht.any() else None,
        "model_mae_spread_same_games": round(float(np.mean(np.abs(pm[hs] - am[hs]))), 3) if hs.any() else None,
        "ats_wins": ats_w, "ats_losses": ats_l, "ats_pct": pct(ats_w, ats_l),
        "edge_ats_wins": edge_w, "edge_ats_losses": edge_l, "edge_ats_n": edge_w + edge_l,
        "edge_ats_pct": pct(edge_w, edge_l), "edge_roi_flat_110": roi(edge_w, edge_l),
        "ou_wins": ou_w, "ou_losses": ou_l, "ou_pct": pct(ou_w, ou_l),
        "edge_ou_wins": eou_w, "edge_ou_losses": eou_l, "edge_ou_pct": pct(eou_w, eou_l),
        "pred_spread_std": round(float(np.std(pm)), 2),
        "vegas_spread_std": round(float(np.std(line[hs])), 2) if hs.any() else None,
        "actual_spread_std": round(float(np.std(am)), 2),
        "edge_threshold_pts": settings.SPREAD_EDGE_PTS,
    }
    if "game_type" in df.columns:
        gt = df["game_type"].to_numpy()
        out["by_game_type"] = {
            str(t): {"n": int((gt == t).sum()),
                     "mae_total": round(float(np.mean(np.abs(pt - at)[gt == t])), 3)}
            for t in pd.unique(gt) if (gt == t).sum() >= 5}
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE TRACK RECORD
# ══════════════════════════════════════════════════════════════════════════════

def update_performance_from_latest(game_df: pd.DataFrame) -> Optional[dict]:
    """Match the last PUBLISHED predictions (the committed
    predictions_latest.json, i.e. what was on the site before kickoff) with
    games completed since, and update performance.json. Must run before the
    file is overwritten."""
    latest = PRED_DIR / "predictions_latest.json"
    if not latest.exists():
        return _migrate_if_needed()
    payload = json.loads(latest.read_text())
    preds = pd.DataFrame(payload.get("predictions", []))
    if preds.empty:
        return None
    done = game_df[game_df["target_home_score"].notna()].set_index("game_id")
    preds = preds[preds["game_id"].isin(done.index)]
    if preds.empty:
        logger.info("Performance update: no newly completed predicted games.")
        return _migrate_if_needed()
    default_version = payload.get("model_version") or PRE_AUDIT_VERSION
    logger.info("Performance update: reconciling %d completed games.", len(preds))

    perf = json.loads(PERF_PATH.read_text()) if PERF_PATH.exists() else {"games": []}
    games = {g["game_id"]: g for g in perf["games"]}
    for _, p in preds.iterrows():
        games[p["game_id"]] = {
            "game_id": p["game_id"], "week": int(p["week"]),
            "game_type": p.get("game_type", "REG"),
            "home_team": p["home_team"], "away_team": p["away_team"],
            "predicted_home": int(p["predicted_home_score"]),
            "predicted_away": int(p["predicted_away_score"]),
            "actual_home": int(done.at[p["game_id"], "target_home_score"]),
            "actual_away": int(done.at[p["game_id"], "target_away_score"]),
            "book_spread": _clean(p.get("book_spread")), "book_total": _clean(p.get("book_total")),
            "confidence_label": p.get("confidence_label"),
            "confidence_score": p.get("confidence_score"),
            "model_version": p.get("model_version") or default_version,
        }
    perf = rebuild_performance(list(games.values()))
    PERF_PATH.write_text(json.dumps(perf, indent=2))
    logger.info("Performance updated: %d games tracked", len(perf["games"]))
    return perf


def _migrate_if_needed() -> Optional[dict]:
    """Bring a pre-audit performance.json (no model versions, old result
    definitions) to the current format; the stored predictions are kept."""
    if not PERF_PATH.exists():
        return None
    perf = json.loads(PERF_PATH.read_text())
    if "by_model_version" in perf:
        return None
    perf = rebuild_performance(perf.get("games", []))
    PERF_PATH.write_text(json.dumps(perf, indent=2))
    logger.info("performance.json migrated to the current format (%d games)", len(perf["games"]))
    return perf


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def rebuild_performance(games: list) -> dict:
    """(Re)derive every per-game result and the summary from the stored
    predicted/actual/line numbers, so the definitions stay consistent."""
    for g in games:
        g.setdefault("model_version", PRE_AUDIT_VERSION)
        pm = g["predicted_home"] - g["predicted_away"]
        am = g["actual_home"] - g["actual_away"]
        pt = g["predicted_home"] + g["predicted_away"]
        at = g["actual_home"] + g["actual_away"]
        g["error_home"] = g["predicted_home"] - g["actual_home"]
        g["error_away"] = g["predicted_away"] - g["actual_away"]
        g["error_spread"], g["error_total"] = pm - am, pt - at
        g["ats_result"] = _side_result(pm, am, g.get("book_spread"))
        g["ou_result"] = _side_result(pt, at, g.get("book_total"))
        bs, bt = g.get("book_spread"), g.get("book_total")
        g["spread_edge"] = None if bs is None else pm - bs
        g["total_edge"] = None if bt is None else pt - bt
        g["edge_game"] = g["spread_edge"] is not None and abs(g["spread_edge"]) >= settings.SPREAD_EDGE_PTS
        for stale in ("spread_lean", "total_lean"):   # pre-audit fields (1.5-pt rule)
            g.pop(stale, None)
    games.sort(key=lambda g: (g["week"], g["game_id"]))
    versions = sorted({g["model_version"] for g in games})
    return {
        "games": games,
        "summary": _summary(games),
        "by_model_version": {v: _summary([g for g in games if g["model_version"] == v])
                             for v in versions},
        "definitions": {"edge_threshold_pts": settings.SPREAD_EDGE_PTS,
                        "ats_ou": "model side vs the line at publication time, every game"},
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _side_result(pred: float, actual: float, line) -> Optional[str]:
    if line is None:
        return None
    if actual == line:
        return "PUSH"
    if pred == line:
        return None                      # model exactly on the number: no pick
    return "WIN" if (pred > line) == (actual > line) else "LOSS"


def _summary(games: list) -> dict:
    if not games:
        return {}

    def rec(rows, key):
        w = sum(1 for g in rows if g[key] == "WIN")
        lo = sum(1 for g in rows if g[key] == "LOSS")
        return f"{w}-{lo}", (round(w / (w + lo), 4) if w + lo else None), w, lo

    with_line = [g for g in games if g.get("book_spread") is not None]
    with_total = [g for g in games if g.get("book_total") is not None]
    edge = [g for g in games if g["edge_game"]]
    ats, ats_pct, _, _ = rec(games, "ats_result")
    ou, ou_pct, _, _ = rec(games, "ou_result")
    e_rec, e_pct, ew, el = rec(edge, "ats_result")
    mae = lambda xs: round(float(np.mean(np.abs(xs))), 3) if xs else None  # noqa: E731
    tiers = {}
    for label in ("HIGH", "MEDIUM", "LOW", "WEAK"):
        t = [g for g in games if g.get("confidence_label") == label]
        if t:
            tiers[label] = {"n": len(t), "mae_spread": mae([g["error_spread"] for g in t]),
                            "mae_total": mae([g["error_total"] for g in t]),
                            "ats_pct": rec(t, "ats_result")[1]}
    return {
        "n_games": len(games),
        "mae_home": mae([g["error_home"] for g in games]),
        "mae_away": mae([g["error_away"] for g in games]),
        "mae_spread": mae([g["error_spread"] for g in games]),
        "mae_total": mae([g["error_total"] for g in games]),
        "vegas_mae_spread": mae([g["book_spread"] - (g["actual_home"] - g["actual_away"]) for g in with_line]),
        "vegas_mae_total": mae([g["book_total"] - (g["actual_home"] + g["actual_away"]) for g in with_total]),
        "ats_record": ats, "ats_pct": ats_pct,
        "ou_record": ou, "ou_pct": ou_pct,
        "edge_record": e_rec, "edge_pct": e_pct,
        "edge_roi_flat_110": round((ew * WIN_PAYOUT - el) / (ew + el), 4) if ew + el else None,
        "confidence_breakdown": tiers,
    }

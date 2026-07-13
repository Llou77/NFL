#!/usr/bin/env python3
"""
walkforward.py — deep walk-forward simulation, ONE fold per invocation.

The idea (proposed by the repo owner, hardened here):
  slide a fixed training window across history, at every step tune ONLY on
  data available before the test season, train, predict the (already
  played) test season, and record how well it went — season after season,
  up to the present. The per-fold results tell us the model's real
  out-of-sample skill and how its tuned weights drift over time.

HONESTY RULES built into this script:
  * NESTED tuning: Optuna sees train = [S-W .. S-2], validation = S-1.
    The test season S is never touched before the final evaluation.
  * The fragments produced here must NEVER be used to re-tune the model —
    aggregating walk-forward results and optimising on them would turn the
    backtest into a training set (meta-overfitting).
  * Weight drift is LOGGED and PLOTTED (see scripts/merge_walkforward.py),
    not extrapolated.

Data comes exclusively from the committed data/frozen/ layer — a fold run
needs zero network access.

Usage:
  python model/walkforward.py --test-season 2016 --window 3 [--trials 30]

Output:
  data/predictions/walkforward/wf_w{window}_{season}.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("walkforward")

FROZEN = ROOT / "data" / "frozen"
RAW    = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "predictions" / "walkforward"

# Combined per-season tables the feature code reads via get_table()
PER_SEASON_TABLES = [
    "player_stats_weekly", "rosters", "rosters_weekly", "snap_counts",
    "injuries", "depth_charts", "ftn_charting",
    "pfr_pass", "pfr_rush", "pfr_rec", "pfr_def",
]
# Full-history tables copied as-is
STATIC_TABLES = [
    "schedules", "ngs_passing", "ngs_rushing", "ngs_receiving",
    "officials", "game_lines", "scoring_lines", "win_totals",
    "draft_picks", "draft_values", "combine", "players", "contracts",
    "id_map", "team_desc",
]


def _harmonize_dtypes(frames: list[pd.DataFrame], tbl: str) -> list[pd.DataFrame]:
    """Reconcile schema drift across nflverse eras before concat.

    The same column can be a string in one season's file and a float in
    another's (e.g. jersey_number pre/post 2016) — concat then yields a
    mixed object column that pyarrow refuses to write. Rule:
      * dtype mix involving object/string  → cast to nullable string
      * purely numeric mix (int vs float)  → widen to float64
    """
    from collections import defaultdict
    dtypes: dict[str, set] = defaultdict(set)
    for f in frames:
        for c in f.columns:
            dtypes[c].add(str(f[c].dtype))

    to_string  = sorted(c for c, ds in dtypes.items()
                        if len(ds) > 1 and ("object" in ds or "string" in ds))
    to_float   = sorted(c for c, ds in dtypes.items()
                        if len(ds) > 1 and c not in to_string)
    if to_string or to_float:
        log.info("  %s: harmonizing dtypes (→string: %s, →float64: %s)",
                 tbl, to_string[:6], to_float[:6])
    for f in frames:
        for c in to_string:
            if c in f.columns:
                f[c] = f[c].astype("string")
        for c in to_float:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    return frames


def _write_parquet_safe(df: pd.DataFrame, path: Path, tbl: str) -> None:
    """to_parquet with a last-resort fallback: if pyarrow still refuses a
    mixed object column (heterogeneous types WITHIN one file), stringify
    every object column and retry once."""
    try:
        df.to_parquet(path, index=False)
    except Exception as e:  # noqa: BLE001
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        log.warning("  %s: parquet write failed (%s) — stringifying %d "
                    "object columns and retrying", tbl, e, len(obj_cols))
        for c in obj_cols:
            df[c] = df[c].astype("string")
        df.to_parquet(path, index=False)


def prepare_tables(seasons: list[int]) -> None:
    """Materialise data/raw/ for this fold purely from data/frozen/.

    Older seasons legitimately lack some sources (NGS 2016+, PFR 2018+,
    FTN 2022+, snap counts 2012+) — those tables simply cover fewer
    seasons and the resulting feature columns are NaN, which the tree
    models handle natively.
    """
    RAW.mkdir(parents=True, exist_ok=True)

    for name in STATIC_TABLES:
        src = FROZEN / f"{name}.parquet"
        if src.exists():
            shutil.copy2(src, RAW / f"{name}.parquet")
        else:
            log.warning("static frozen table missing: %s", name)

    for tbl in PER_SEASON_TABLES:
        frames = []
        for s in seasons:
            p = FROZEN / f"{tbl}_{s}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        if frames:
            frames = _harmonize_dtypes(frames, tbl)
            combined = pd.concat(frames, ignore_index=True)
            _write_parquet_safe(combined, RAW / f"{tbl}.parquet", tbl)
        else:
            log.warning("no frozen data for %s in seasons %s "
                        "(source may not exist for this era)", tbl, seasons)

    missing_agg = [s for s in seasons
                   if not (FROZEN / f"pbp_agg_{s}.parquet").exists()]
    if missing_agg:
        log.error("Missing frozen PBP aggregates for %s — run the "
                  "freeze_data workflow with FREEZE_FROM low enough.",
                  missing_agg)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--window", type=int, default=3,
                    help="number of training seasons before the test season")
    ap.add_argument("--trials", type=int, default=30,
                    help="Optuna trials for the nested tuning step")
    args = ap.parse_args()

    S, W = args.test_season, args.window
    train_seasons = list(range(S - W, S))
    seasons = train_seasons + [S]
    log.info("FOLD: test=%d  window=%d  train=%s  trials=%d",
             S, W, train_seasons, args.trials)

    prepare_tables(seasons)

    from feature_engineering import build_all_features, get_feature_columns
    from feature_h2h import build_h2h_features

    game_df = build_all_features(seasons=seasons)
    game_df = build_h2h_features(game_df)

    labeled = game_df[game_df["target_home_score"].notna()
                      & game_df["target_away_score"].notna()].copy()
    df_train = labeled[labeled["season"].isin(train_seasons)].copy()
    df_test  = labeled[labeled["season"] == S].copy()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"wf_w{W}_{S}.json"

    if len(df_train) < 150 or len(df_test) < 50:
        log.warning("Insufficient data (train=%d test=%d) — writing skip "
                    "fragment.", len(df_train), len(df_test))
        out_path.write_text(json.dumps({f"{S}_w{W}": {
            "test_season": S, "window": W, "skipped": True,
            "n_train": len(df_train), "n_test": len(df_test)}}, indent=2))
        return

    feature_cols = get_feature_columns(game_df)

    # ── nested tuning (test season untouched) ────────────────────────────
    from bayesian_optimizer import run_bayesian_optimization, DEFAULT_WEIGHTS
    val_season = train_seasons[-1]
    tune_train = df_train[df_train["season"] < val_season]
    tune_val   = df_train[df_train["season"] == val_season]
    if len(tune_train) >= 150 and len(tune_val) >= 100 and args.trials > 0:
        weights = run_bayesian_optimization(tune_train, tune_val,
                                            feature_cols,
                                            n_trials=args.trials)
    else:
        weights = dict(DEFAULT_WEIGHTS)
        log.warning("Nested tuning skipped (train=%d val=%d trials=%d) — "
                    "research-default weights.", len(tune_train),
                    len(tune_val), args.trials)

    # ── train on the window, evaluate on the held-out season ─────────────
    from train import train_all, load_models
    from predict import ensemble_predict
    from evaluate import compute_metrics

    train_all(df_train, feature_cols, weights, current_season=S, run_cv=False)
    models = load_models()
    model_cols = models.get("feature_cols", feature_cols)
    X_test = df_test.reindex(columns=model_cols,
                             fill_value=np.nan)[model_cols].values
    _, _, raw_h, raw_a = ensemble_predict(models, X_test,
                                          apply_calibration=True)

    res = compute_metrics(
        np.round(raw_h).astype(float), np.round(raw_a).astype(float),
        df_test["target_home_score"].values.astype(float),
        df_test["target_away_score"].values.astype(float),
        df_test,
    )
    res.update({
        "test_season":   S,
        "window":        W,
        "train_seasons": train_seasons,
        "n_train":       len(df_train),
        "n_test":        len(df_test),
        "n_features":    len(model_cols),
        "trials":        args.trials,
        # 2020 had no fans (HFA collapsed) — flag folds it contaminates
        "covid_affected": (S == 2020) or (2020 in train_seasons),
        "tuned_weights": {k: round(float(v), 4) for k, v in weights.items()},
    })

    out_path.write_text(json.dumps({f"{S}_w{W}": res}, indent=2,
                                   default=str))
    log.info("FOLD DONE %d (w=%d): MAE_spread=%.2f MAE_total=%.2f "
             "ATS=%.1f%% OU=%.1f%% edgeATS=%s → %s",
             S, W, res.get("mae_spread", float("nan")),
             res.get("mae_total", float("nan")),
             res.get("ats_pct", 0) * 100, res.get("ou_pct", 0) * 100,
             f"{res['edge_ats_pct']*100:.1f}%" if res.get("edge_ats_pct") is not None else "n/a",
             out_path)


if __name__ == "__main__":
    main()

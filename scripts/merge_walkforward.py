#!/usr/bin/env python3
"""
merge_walkforward.py — combine walk-forward fold fragments, summarise, plot.

Inputs : data/predictions/walkforward/wf_w{window}_{season}.json fragments
Outputs: data/predictions/walkforward_results.json  (+ copy in docs/assets/)
         docs/assets/walkforward_weights.png        (weight drift + metrics)

The plot LOGS the tuned-weight evolution across folds; deliberately no
trend extrapolation — forecasting a model's own parameter drift from ~12
noisy points adds variance, not signal. Use the chart to SPOT regime
shifts (e.g. 2020 COVID), then model the underlying league covariate as a
feature instead.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "data" / "predictions" / "walkforward"
OUT  = ROOT / "data" / "predictions" / "walkforward_results.json"
DOCS = ROOT / "docs" / "assets"

TUNED_KEYS = ["w_oldest", "w_middle", "w_recent", "w_current",
              "wt_wc", "wt_div", "wt_con", "wt_sb"]


def main() -> None:
    merged = {}
    if OUT.exists():
        try:
            merged = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            merged = {}
    # The previous run's aggregate block must not be treated as a fold row
    # (this exact mistake once crashed the summary step mid-script).
    merged.pop("_summary", None)

    fragments = sorted(FRAG.glob("wf_w*_*.json")) if FRAG.exists() else []
    if not fragments:
        print("no walkforward fragments found", file=sys.stderr)
        sys.exit(1)

    for p in fragments:
        try:
            merged.update(json.loads(p.read_text()))
        except json.JSONDecodeError as e:
            print(f"skipping corrupt fragment {p.name}: {e}", file=sys.stderr)

    OUT.write_text(json.dumps(merged, indent=2, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, DOCS / "walkforward_results.json")

    # ── console summary + per-window aggregates ───────────────────────────
    rows = [v for k, v in merged.items()
            if k != "_summary" and isinstance(v, dict) and not v.get("skipped")]
    rows.sort(key=lambda r: (r["window"], r["test_season"]))
    print(f"{'season':>6} {'win':>3} {'MAEsp':>6} {'MAEtot':>6} "
          f"{'ATS%':>6} {'OU%':>6} {'edgeATS%':>8} {'ROI':>7} {'covid':>5}")
    for r in rows:
        e = r.get("edge_ats_pct")
        print(f"{r['test_season']:>6} {r['window']:>3} "
              f"{r.get('mae_spread', float('nan')):>6.2f} "
              f"{r.get('mae_total', float('nan')):>6.2f} "
              f"{r.get('ats_pct', 0)*100:>6.1f} "
              f"{r.get('ou_pct', 0)*100:>6.1f} "
              f"{(e*100 if e is not None else float('nan')):>8.1f} "
              f"{(r.get('edge_roi_flat_110') if r.get('edge_roi_flat_110') is not None else float('nan')):>7.3f} "
              f"{'x' if r.get('covid_affected') else '':>5}")

    summary = {}
    for w in sorted({r["window"] for r in rows}):
        sel = [r for r in rows if r["window"] == w]
        clean = [r for r in sel if not r.get("covid_affected")]

        def _avg(rs, key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary[f"window_{w}"] = {
            "n_folds":            len(sel),
            "mean_mae_spread":    _avg(sel, "mae_spread"),
            "mean_ats_pct":       _avg(sel, "ats_pct"),
            "mean_edge_ats_pct":  _avg(sel, "edge_ats_pct"),
            "mean_edge_roi":      _avg(sel, "edge_roi_flat_110"),
            "mean_ats_pct_excl_covid": _avg(clean, "ats_pct"),
            "mean_edge_ats_pct_excl_covid": _avg(clean, "edge_ats_pct"),
        }
    # ── tuned-weight stability across folds ───────────────────────────────
    # The "stable trend" idea (repo owner): where independent per-fold
    # tunings keep agreeing on a value (low CV), that value is a durable
    # property of the problem → candidate for pinning/narrowing the search.
    # Where they disagree wildly, tuning is mostly chasing noise → also a
    # candidate for fixing, for the opposite reason.
    import statistics as _st
    tuned_rows = [r for r in rows if r.get("tuned_weights")]
    stability = {}
    for key in TUNED_KEYS:
        vals = [r["tuned_weights"][key] for r in tuned_rows
                if key in r.get("tuned_weights", {})]
        if len(vals) >= 4:
            m = _st.mean(vals)
            cv = (_st.stdev(vals) / abs(m)) if m else float("inf")
            stability[key] = {
                "median": round(_st.median(vals), 3),
                "cv_pct": round(100 * cv, 1),
                "class": ("stable" if cv < 0.20 else
                          "medium" if cv < 0.40 else "noisy"),
            }
    summary["weight_stability"] = stability

    merged["_summary"] = summary
    OUT.write_text(json.dumps(merged, indent=2, default=str))
    shutil.copy2(OUT, DOCS / "walkforward_results.json")
    print("\nsummary:", json.dumps(summary, indent=2))

    # ── plot (optional — needs matplotlib) ────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    # (1) tuned weight drift — window 3 folds
    ax = axes[0]
    w3 = [r for r in rows if r["window"] == 3 and r.get("tuned_weights")]
    xs = [r["test_season"] for r in w3]
    for key in TUNED_KEYS:
        ys = [r["tuned_weights"].get(key) for r in w3]
        ax.plot(xs, ys, marker="o", markersize=3, label=key)
    ax.axvspan(2019.5, 2021.5, alpha=0.12, color="red",
               label="COVID-affected")
    ax.set_title("Tuned sample-weight drift across folds (window=3) — "
                 "logged, NOT extrapolated")
    ax.set_ylabel("weight value")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # (2) spread MAE per fold
    ax = axes[1]
    for w, style in [(3, "o-"), (4, "s--")]:
        sel = [r for r in rows if r["window"] == w]
        ax.plot([r["test_season"] for r in sel],
                [r.get("mae_spread") for r in sel], style,
                markersize=4, label=f"window={w}")
    ax.axhline(10.2, color="gray", ls=":", label="≈ Vegas closing (10.2)")
    ax.set_title("Spread MAE by test season (lower = better)")
    ax.set_ylabel("points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) ATS / edge ATS per fold
    ax = axes[2]
    for w, style in [(3, "o-"), (4, "s--")]:
        sel = [r for r in rows if r["window"] == w]
        ax.plot([r["test_season"] for r in sel],
                [r.get("ats_pct", 0) * 100 for r in sel], style,
                markersize=4, label=f"ATS window={w}")
        es = [(r["test_season"], r["edge_ats_pct"] * 100)
              for r in sel if r.get("edge_ats_pct") is not None]
        if es:
            ax.plot([x for x, _ in es], [y for _, y in es], style,
                    alpha=0.45, markersize=3, label=f"edge ATS window={w}")
    ax.axhline(52.4, color="red", ls=":", label="break-even (52.4%)")
    ax.axhline(50.0, color="gray", ls=":", alpha=0.5)
    ax.set_title("ATS hit rate by test season")
    ax.set_ylabel("%")
    ax.set_xlabel("test season")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png = DOCS / "walkforward_weights.png"
    fig.savefig(png, dpi=110)
    print(f"plot → {png}")


if __name__ == "__main__":
    main()

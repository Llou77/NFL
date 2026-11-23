#!/usr/bin/env python3
"""
merge_walkforward.py — combine walk-forward fold fragments, summarise, plot.

Inputs : data/predictions/walkforward/wf_w{window}_{season}.json
Outputs: data/predictions/walkforward_results.json (+ copy in docs/assets/)
         docs/assets/walkforward_chart.png  (model vs Vegas error, ATS per season)

Only the folds of the CURRENT run are kept: results produced by older model
code must not be averaged together with new ones.
"""

import json
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "data" / "predictions" / "walkforward"
OUT  = ROOT / "data" / "predictions" / "walkforward_results.json"
DOCS = ROOT / "docs" / "assets"


def _pct(x):
    """Percentage for printing/plotting; NaN (a gap) when there were no games."""
    return float("nan") if x is None else 100 * x


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(st.mean(vals), 4) if vals else None


def main() -> None:
    fragments = sorted(FRAG.glob("wf_w*_*.json")) if FRAG.exists() else []
    if not fragments:
        sys.exit("no walk-forward fragments found")
    merged = {}
    for p in fragments:
        try:
            merged.update(json.loads(p.read_text()))
        except json.JSONDecodeError as e:
            print(f"skipping corrupt fragment {p.name}: {e}", file=sys.stderr)

    rows = sorted(merged.values(), key=lambda r: (r["window"], r["test_season"]))
    print(f"{'season':>6} {'win':>3} {'MAEsp':>6} {'Vegas':>6} {'MAEtot':>6} {'Vegas':>6} "
          f"{'ATS%':>5} {'edge%':>5} {'n':>4}")
    for r in rows:
        print(f"{r['test_season']:>6} {r['window']:>3} {r['mae_spread']:>6.2f} "
              f"{r['vegas_mae_spread'] or float('nan'):>6.2f} {r['mae_total']:>6.2f} "
              f"{r['vegas_mae_total'] or float('nan'):>6.2f} {_pct(r['ats_pct']):>5.1f} "
              f"{_pct(r['edge_ats_pct']):>5.1f} {r['edge_ats_n']:>4}")

    summary = {}
    for w in sorted({r["window"] for r in rows}):
        sel = [r for r in rows if r["window"] == w]
        clean = [r for r in sel if not r.get("covid_affected")]
        ew = sum(r["edge_ats_wins"] for r in sel)
        el = sum(r["edge_ats_losses"] for r in sel)
        summary[f"window_{w}"] = {
            "n_folds": len(sel),
            "seasons": [r["test_season"] for r in sel],
            "mean_mae_spread": _mean(sel, "mae_spread"),
            "mean_vegas_mae_spread": _mean(sel, "vegas_mae_spread"),
            "mean_mae_total": _mean(sel, "mae_total"),
            "mean_vegas_mae_total": _mean(sel, "vegas_mae_total"),
            "mean_ats_pct": _mean(sel, "ats_pct"),
            "mean_ats_pct_excl_covid": _mean(clean, "ats_pct"),
            "pooled_edge_ats_pct": round(ew / (ew + el), 4) if ew + el else None,
            "pooled_edge_n": ew + el,
            "pooled_edge_roi_flat_110": round((ew * 100 / 110 - el) / (ew + el), 4) if ew + el else None,
            "folds_beating_vegas_spread": sum(1 for r in sel if r["mae_spread"] < (r["vegas_mae_spread"] or 0)),
        }
    merged["_summary"] = summary
    OUT.write_text(json.dumps(merged, indent=2, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, DOCS / "walkforward_results.json")
    print("\nsummary:", json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot", file=sys.stderr)
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for w in sorted({r["window"] for r in rows}):
        sel = [r for r in rows if r["window"] == w]
        xs = [r["test_season"] for r in sel]
        axes[0].plot(xs, [r["mae_spread"] for r in sel], "o-", label=f"model (window {w})")
        axes[1].plot(xs, [_pct(r["ats_pct"]) for r in sel], "o-", label=f"ATS, all games (window {w})")
        axes[1].plot(xs, [_pct(r["edge_ats_pct"]) for r in sel], "s--", alpha=0.6,
                     label=f"ATS, edge games (window {w})")
    vegas = {r["test_season"]: r["vegas_mae_spread"] for r in rows}
    axes[0].plot(sorted(vegas), [vegas[s] for s in sorted(vegas)], "k:", label="Vegas line")
    axes[0].set_title("Margin (spread) error per held-out season — lower is better")
    axes[0].set_ylabel("mean absolute error (pts)")
    axes[1].axhline(52.4, color="red", ls=":", label="break-even at -110 (52.4%)")
    axes[1].axhline(50, color="gray", ls=":", alpha=0.5)
    axes[1].set_title("Against-the-spread hit rate per held-out season")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("test season")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "walkforward_chart.png", dpi=110)
    print(f"plot → {DOCS / 'walkforward_chart.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
season_guard.py — decide whether a scheduled pipeline run has anything to do.

During the offseason there are no games to predict and no new results to
learn from, yet the weekly cron used to spend 15+ minutes downloading data
and retraining for nothing. This guard is the first step of the scheduled
workflows: it prints "run" or "skip" on stdout (reasons go to stderr), and
the workflow skips every expensive step on "skip".

Deliberately STDLIB-ONLY so it can run before `pip install`
(a skipped run costs ~30 seconds total).

Decision rule: RUN if any NFL game is scheduled in the window
[today-3 days, today+10 days] — i.e. there is either a fresh result to
reconcile or an upcoming game to predict. Otherwise SKIP.
Fail-open: if the schedule cannot be fetched, print "run" (never let the
guard itself break the pipeline).

Usage: python3 scripts/season_guard.py [--force]
"""

import csv
import io
import sys
import urllib.request
from datetime import date, timedelta

SCHEDULE_URL = "http://www.habitatring.com/games.csv"
LOOKBACK_DAYS  = 3
LOOKAHEAD_DAYS = 10


def main() -> None:
    if "--force" in sys.argv:
        print("run")
        print("guard: --force given, skipping schedule check", file=sys.stderr)
        return

    try:
        raw = urllib.request.urlopen(SCHEDULE_URL, timeout=30).read()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    except Exception as e:  # noqa: BLE001 — fail-open by design
        print("run")
        print(f"guard: schedule fetch failed ({e}) — failing open", file=sys.stderr)
        return

    today = date.today()
    lo = today - timedelta(days=LOOKBACK_DAYS)
    hi = today + timedelta(days=LOOKAHEAD_DAYS)

    n_window = 0
    for r in rows:
        gd = (r.get("gameday") or "").strip()
        try:
            y, m, d = (int(x) for x in gd.split("-"))
            g = date(y, m, d)
        except (ValueError, AttributeError):
            continue
        if lo <= g <= hi:
            n_window += 1

    if n_window:
        print("run")
        print(f"guard: {n_window} game(s) in [{lo} … {hi}] — season active",
              file=sys.stderr)
    else:
        print("skip")
        print(f"guard: no games in [{lo} … {hi}] — offseason, nothing to do",
              file=sys.stderr)


if __name__ == "__main__":
    main()

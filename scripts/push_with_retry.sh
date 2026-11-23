#!/usr/bin/env bash
# Push the bot's commit(s) to main, rebasing over concurrent bot commits.
# Exits non-zero if the push never lands, so the job shows red instead of
# silently dropping the commit (the old inline retry loop could end green).
set -u
for attempt in 1 2 3 4 5; do
  if git push origin HEAD:main; then
    exit 0
  fi
  echo "push rejected (attempt ${attempt}) — rebasing on origin/main" >&2
  sleep $((attempt * 3))
  if ! git pull --rebase origin main; then
    git rebase --abort 2>/dev/null
    echo "rebase failed — giving up" >&2
    exit 1
  fi
done
echo "push failed after 5 attempts" >&2
exit 1

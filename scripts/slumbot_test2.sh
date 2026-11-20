#!/usr/bin/env bash
# Pre-registered Slumbot test 2 runner. Plays until 20,000 hands are logged.
# A crash (network or protocol) restarts the client in the same log with a new action seed,
# at most MAX_RESTARTS times; every restart is recorded in the restart log for disclosure.
set -u
cd "$(dirname "$0")/.."
CKPT=runs/buckets200-42/iteration-1348732.json.gz
LOG=results/slumbot/test2-20000.jsonl
RESTARTS=results/slumbot/test2-restarts.log
MAX_RESTARTS=10
# Resuming after a pause continues the attempt (and seed) numbering instead of reusing seeds.
FIRST=$( [ -f "$RESTARTS" ] && grep -c ' start attempt=' "$RESTARTS" || echo 0)
for attempt in $(seq "$FIRST" "$MAX_RESTARTS"); do
  seed=$((20 + attempt))
  echo "$(date -Is) start attempt=$attempt seed=$seed logged=$( [ -f "$LOG" ] && wc -l < "$LOG" || echo 0)" >> "$RESTARTS"
  .venv/bin/python -m cortado.slumbot "$CKPT" --hands 20000 --seed "$seed" --log "$LOG" --resume \
       > results/slumbot/test2-summary-attempt$attempt.json
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$(date -Is) finished attempt=$attempt" >> "$RESTARTS"
    cp results/slumbot/test2-summary-attempt$attempt.json results/slumbot/test2-20000-summary.json
    exit 0
  fi
  echo "$(date -Is) crashed attempt=$attempt exit=$code logged=$(wc -l < "$LOG")" >> "$RESTARTS"
  sleep 120
done
echo "$(date -Is) gave up after $MAX_RESTARTS restarts" >> "$RESTARTS"
exit 1

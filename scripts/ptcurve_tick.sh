#!/usr/bin/env bash
# Hourly learning-curve step for the parallel training runs (invoked by a systemd user timer).
cd "$(dirname "$0")/.."
exec 9> runs/.ptcurve.lock
flock -n 9 || exit 0
for s in 100 200; do
  [ -d runs/ptrain-v3-sizes-$s ] || continue
  echo "$(date -Is) tick $s" >> runs/ptcurve-$s.log
  .venv/bin/python -m doppio.ptcurve --run-dir runs/ptrain-v3-sizes-$s --out-dir results/curve-v3-sizes-$s \
    --service doppio-ptrain-$s >> runs/ptcurve-$s.log 2>&1
done

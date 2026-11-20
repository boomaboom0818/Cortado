#!/usr/bin/env bash
# A/B on development deals (seed 700000): plain blueprint vs real-history re-solving, same checkpoint, same paired deals.
cd "$(dirname "$0")/.."
CK=${CK:-runs/ptrain-v3-sizes-100/iteration-7780000.json.gz}
OUT=${OUT:-results/rhresolve-ab-100}
OPP="nit tag min-raiser third-pot random-size overbet"
for agent in blueprint rhresolve; do
  .venv/bin/python -m cortado.fullbench $CK --agent $agent --pairs 1500 --seed 700000 --workers ${W:-4} --blocks 8 \
    --opponents $OPP --aivat --out $OUT/$agent.json > $OUT/$agent.log 2>&1
done

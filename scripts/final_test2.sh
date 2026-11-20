#!/usr/bin/env bash
# Final test 2, exactly as pre-registered in EVALUATION_PROTOCOL.md. Each step skips if its output exists.
cd "$(dirname "$0")/.."
OUT=results/final-test-2; mkdir -p $OUT
py=.venv/bin/python
[ -s $OUT/C_vs_A.json ] || $py -m cortado.headsup runs/final-2/C-100bb.json.gz models/cortado-a-100bb.json.gz --pairs 30000 --blocks 10 --workers 5 --seed 300000000 --out $OUT/C_vs_A.json > $OUT/C_vs_A.log 2>&1
[ -s $OUT/D_vs_B.json ] || $py -m cortado.headsup runs/final-2/D-200bb.json.gz models/cortado-b-200bb.json.gz --pairs 30000 --blocks 10 --workers 5 --seed 300000000 --out $OUT/D_vs_B.json > $OUT/D_vs_B.log 2>&1
for c in C-100bb D-200bb; do n=${c%%-*}
  [ -s $OUT/$n.json ] || $py -m cortado.fullbench runs/final-2/$c.json.gz --seed 200000000 --pairs 30000 --blocks 10 --workers 5 \
    --lbr-pairs 3000 --lbr-hands 60 --aivat --out $OUT/$n.json > $OUT/$n.log 2>&1
done

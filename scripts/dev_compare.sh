#!/usr/bin/env bash
# Development replication on a fresh dev seed block (800000): frozen v1 models vs v3 snapshots, identical deals.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet cortado-rhab-200; do sleep 30; done
OUT=results/dev-compare-1
OPP="nit tag min-raiser third-pot random-size overbet"
run() { # name checkpoint
  [ -s $OUT/$1.json ] && return
  .venv/bin/python -m cortado.fullbench $2 --pairs 10000 --seed 800000 --workers 8 --blocks 8 \
    --opponents $OPP --aivat --out $OUT/$1.json > $OUT/$1.log 2>&1
}
run A-100bb-v1          models/cortado-a-100bb.json.gz
run v3-100bb-3180000    runs/ptrain-v3-sizes-100/iteration-3180000.json.gz
run v3-100bb-7780000    runs/ptrain-v3-sizes-100/iteration-7780000.json.gz
run B-200bb-v1          models/cortado-b-200bb.json.gz
run v3-200bb-4780000    runs/ptrain-v3-sizes-200/iteration-4780000.json.gz
run v3-200bb-7220000    runs/ptrain-v3-sizes-200/iteration-7220000.json.gz

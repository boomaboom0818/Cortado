#!/usr/bin/env bash
# Development head-to-head: v3 snapshots vs frozen v1 models (seed 850000, 10,000 paired deals each).
cd "$(dirname "$0")/.."
OUT=results/headsup-1
m() { [ -s $OUT/$1.json ] || .venv/bin/python -m cortado.headsup $2 $3 --pairs 10000 --blocks 6 --workers 3 --seed 850000 --out $OUT/$1.json > $OUT/$1.log 2>&1; }
m v3-100-3180000_vs_A runs/ptrain-v3-sizes-100/iteration-3180000.json.gz models/cortado-a-100bb.json.gz
m v3-200-4780000_vs_B runs/ptrain-v3-sizes-200/iteration-4780000.json.gz models/cortado-b-200bb.json.gz
m v3-200-7220000_vs_B runs/ptrain-v3-sizes-200/iteration-7220000.json.gz models/cortado-b-200bb.json.gz

# Models

Two frozen candidates are included, chosen on development data and then tested once on reserved deals
(`results/final-test-1/REPORT.md`, rules in `EVALUATION_PROTOCOL.md`). Both are strategy tables from
external-sampling MCCFR with the `buckets-v1` card abstraction. Neither is shown to be strong against a strong
opponent, and neither is a GTO or near-GTO strategy.

| | **Model A** | **Model B** |
|---|---|---|
| File | `cortado-a-100bb.json.gz` | `cortado-b-200bb.json.gz` |
| Stack depth | 100 BB (blinds 0.5 / 1 BB) | 200 BB (blinds 0.5 / 1 BB) |
| Training | 525,987 iterations, seed 42 | 1,348,732 iterations, seed 42 |
| SHA-256 | `64e5057b...be559` | `3024ae99...b239` |
| Use it when | you play 100 BB | you play 200 BB (Slumbot's depth) |
| Beats the nine scripted pool bots? | 8 of 9; not shown against the nit (+5, CI -4 to +15) | **9 of 9**, worst case +72 vs the nit (CI +57 to +87) |
| LBR attacker | no profitable exploit found (+43) | no profitable exploit found (+85) |
| Slumbot (hosted, 200 BB) | not played | 2,683 hands: -32 bb/100, CI -100 to +37, **inconclusive** |

Numbers are AIVAT-corrected bb/100 on 30,000 paired deals per bot (reserved seed block 100,000,000). A and B play
different games, so their scores are not comparable. Full SHA-256 values are in `SHA256SUMS`.

## What both share

* **Card abstraction `buckets-v1`:** 169 lossless preflop classes; 30 equity buckets per postflop street; imperfect recall.
* **Betting:** trained on the FCPA abstraction (fold, call, pot, all-in). `FullActionAgent` plays the full no-limit game:
  opponent bet sizes are mapped to the nearest trained size, and our choices map back to legal chip amounts.
* **Coverage:** the table answers about 93-100% of decisions; the rest fall back to an equity heuristic.
* **Known weakness:** very large or unusual bet sizes (overbets) can knock the agent off its replayed history,
  and it then plays the heuristic. Training longer did not improve the results (`results/curve-v1-*/REPORT.md`).

## Use

```python
from cortado.policies import Policy
from cortado.fullaction import FullActionAgent

agent = FullActionAgent(Policy('checkpoint', 'models/cortado-b-200bb.json.gz'))
# agent.act(observation, rng) -> engine action id
```

```sh
.venv/bin/python -m cortado.fullbench models/cortado-a-100bb.json.gz --pairs 3000 --aivat --seed 700000 --out dev.json
.venv/bin/python -m cortado.slumbot models/cortado-b-200bb.json.gz --hands 200 --log slumbot.jsonl   # contacts slumbot.com
```

Use development seeds (650000-899999) for any tuning. The reserved block (100,000,000 and up) has been used by the final test.

## Other model options in this repo (not frozen candidates)

| Option | Where | Status |
|---|---|---|
| `exact-v1` (full cards, FCPA, 100 BB) | `results/holdem-10k.json.gz`, 10k iterations | Legacy baseline. The table answers only about 47% of decisions, so it often plays uniformly. |
| `suits-v1` (suit-symmetric cards) | disk-backed campaign, `results/bot-campaign/` | 524k / 565k iterations per seed. Large SQLite tables (3.8-4.2 GB) are not committed. About 53% of decisions fell back to uniform against a calling bot. Beats weak controls on development data only. |
| `buckets-v2` (adds board texture) | config `configs/train-bucketsv2-*.json` | At equal compute it did worse than v1 on development data. Trained checkpoints are not committed. |
| Re-solving agent (`--agent resolve`) | `cortado/resolve.py`, `results/resolve/REPORT.md` | Real-time river re-solver. No measured benefit and high latency, so it is off by default. |
| Baseline policies | `uniform`, `call`, `aggressive` (`cortado.cli`) | Controls for sanity checks only. |
| Scripted pool (`pool-v1`) and LBR | `cortado/pool.py`, `cortado/lbr.py` | Opponents we wrote for evaluation. Not models to deploy. |

Retrain with the configs in `configs/` (for example `train-buckets-42.json` for 100 BB, `train-buckets200-42.json`
for 200 BB). Training and evaluation commands are in the root README.

# Models

**C (100 BB) and D (200 BB) are the current frozen models.** They replaced A and B after final test 2
(`results/final-test-2/REPORT.md`), under a promotion rule written down before the test (`EVALUATION_PROTOCOL.md`).
A and B stay here as earlier versions. All four are strategy tables from external-sampling MCCFR self-play.
None is shown to be strong against a strong opponent, and none is a GTO or near-GTO strategy.

| | **C (main, 100 BB)** | **D (main, 200 BB)** | A (earlier, 100 BB) | B (earlier, 200 BB) |
|---|---|---|---|---|
| File | `cortado-c-100bb.json.gz` | `cortado-d-200bb.json.gz` | `cortado-a-100bb.json.gz` | `cortado-b-200bb.json.gz` |
| Abstraction | `sizes-v1` + `buckets-v3` | `sizes-v1` + `buckets-v3` | FCPA + `buckets-v1` | FCPA + `buckets-v1` |
| Training | 3.18M iterations, seed 42 (parallel) | 7.22M iterations, seed 42 (parallel) | 525,987 iterations, seed 42 | 1,348,732 iterations, seed 42 |
| Needs bucket tables? | **yes**, see below | **yes**, see below | no | no |
| Head to head vs earlier model | beats A: +30 bb/100 [+23, +38] | beats B: +33 bb/100 [+22, +45] | | |
| Scripted pool bots beaten (of 9) | **9** (worst: nit +22 [+12, +31]) | **9** (worst: nit +69 [+54, +84]) | 8 (not the nit) | 9 |
| LBR attacker | no profitable exploit found (+101) | no profitable exploit found (+132) | none found (+43) | none found (+85) |
| Slumbot | not played | not played | not played | 2,683 hands, inconclusive |

Numbers are AIVAT-corrected bb/100 on 30,000 paired deals per opponent, on reserved deals (C/D: final test 2;
A/B: final test 1). The pool numbers of the two tests are on different deals and are not directly comparable; the
head-to-head is the direct comparison. Full SHA-256 values are in `SHA256SUMS`.

## C and D: what changed from A and B

* **Bet sizes (`sizes-v1`):** trained with several pot fractions per street (including a 2x-pot overbet) instead of only
  pot and all-in, so most real bet sizes need no translation. Off-tree sizes are translated between the two nearest trained sizes.
* **Card buckets (`buckets-v3`):** 50 buckets per postflop street from exact enumeration over every distinct board
  (E[HS²] on flop and turn, hand strength on the river), instead of 30 sampled equity buckets.
* Real-history re-solving was tried for C/D-type models and dropped: it was worse on development data
  (`results/rhresolve-ab-*/REPORT.md`).

## The bucket tables (C and D only)

C and D look up each hand's postflop bucket in precomputed tables: for every distinct flop, turn and river board
(up to suit symmetry), which of the 50 buckets each two-card holding belongs to. The tables are about 190 MB
(the river table alone is 178 MB, above GitHub's 100 MB file limit), so they are not in git. Build them once:

```sh
.venv/bin/python -m cortado.buckets3 build     # about 15 minutes on 12 cores; writes cortado/data/buckets-v3/
.venv/bin/python -m cortado.buckets3 digest    # must print 4573773f525cd6c43bc4d88868834440847357e8a2006d7a704d28ee7416ac3a
```

The build is deterministic. Loading C or D checks the tables' digest against the one stored in the model and refuses
to play with different tables. A and B use a small bucket file that is in git and need nothing extra.

## Use

```python
from cortado.policies import Policy
from cortado.fullaction import FullActionAgent

agent = FullActionAgent(Policy('checkpoint', 'models/cortado-d-200bb.json.gz'))
# agent.act(observation, rng) -> engine action id
```

```sh
.venv/bin/python -m cortado.fullbench models/cortado-c-100bb.json.gz --pairs 3000 --aivat --seed 700000 --out dev.json
.venv/bin/python -m cortado.headsup models/cortado-d-200bb.json.gz models/cortado-b-200bb.json.gz --pairs 3000 --seed 850000
```

Use development seeds (650000-899999) for any tuning. Seed bases 100,000,000, 200,000,000 and 300,000,000 have been
used by final tests 1 and 2; a new final test needs a fresh block and a new pre-registration.

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

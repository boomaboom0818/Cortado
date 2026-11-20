# Head-to-head: v3 (sizes-v1 + buckets-v3) vs frozen v1 models

`scripts/headsup.sh` / `python -m cortado.headsup`: full no-limit game, 9,996 paired deals per match (each deal played
twice with seats swapped), development seed 850000, AIVAT weights 1,1,1,1 using the v3 model's strategy.
Positive = v3 wins. Development evaluation only; not a pre-registered final test.

| Match | Raw bb/100 [95% CI] | AIVAT bb/100 [95% CI] | Off-history decisions (v3 / frozen) |
|---|---|---|---|
| v3 100 BB 3.18M vs A | +32.6 [+10.6, +54.5] | **+29.0 [+16.4, +41.6]** | 28 / 518 |
| v3 200 BB 4.78M vs B | +30.6 [−1.0, +62.2] | +19.1 [−1.0, +39.2] | 15 / 546 |
| v3 200 BB 7.22M vs B | +44.1 [+15.1, +73.1] | **+38.6 [+19.6, +57.6]** | 16 / 532 |

- v3 wins every match on point estimate; clearly at 100 BB (3.18M) and at 200 BB with 7.22M; 4.78M vs B is borderline.
- The frozen models fall off their replayed history (equity-heuristic fallback) about 1.4% of decisions against v3,
  v3 about 0.04%: v3's extra bet sizes are off-tree for the v1 models.
- At 200 BB the head-to-head favours 7.22M while the pool benchmark (`results/dev-compare-1`) favoured 4.78M against
  overbets; three matches, intervals not corrected for multiplicity.

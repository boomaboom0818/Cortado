# Learning curve: buckets-v1, 100 BB (seed 42)

The run was driven by `cortado.curve`, using a stopping rule fixed before it started:

- **Composite score:** the mean AIVAT-corrected bb/100 against six strategic pool-v1 bots (nit, tag, min-raiser, third-pot, random-size, overbet).
- **Benchmark:** 3,000 paired deals, development seed 700000, with the same deals at every point.
- **Stop:** when two consecutive cycles each improve the composite by less than 10 bb/100.
- **LBR:** reported, but kept out of the composite.

**Result:** stopped by the flat rule after 2 cycles.

| Iterations | Composite | Δ vs previous (95% CI) | nit | tag | min-raiser | third-pot | random-size | overbet | LBR (95% CI) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 525,987 (baseline) | 148.3 | – | 26 | 99 | 200 | 120 | 197 | 249 | 56 [−43, 155] |
| 1,327,539 | 146.3 | −2.1 [−17.6, 13.5] | 23 | 137 | 175 | 118 | 182 | 242 | 80 [−25, 186] |
| 1,817,355 | 146.1 | −0.1 [−9.4, 9.1] | 30 | 140 | 175 | 108 | 190 | 234 | 54 [−48, 156] |

The final checkpoint changed by −2.2 bb/100 against the baseline (95% CI −18.9 to +14.5, paired). Nearly 3.5 times as much training produced no measurable gain. The table covered 98% of decisions at every point.

## Interpretation

At 100 BB, buckets-v1 is saturated at about 0.5M iterations against this benchmark. The limits are now the abstraction and the evaluation, not training compute:

- **Card abstraction:** 30 equity buckets per street, with imperfect recall.
- **Betting abstraction:** FCPA betting, with full-action play handled by translation.

More iterations of the same setup are not a good use of compute. Every LBR interval includes small or zero values, so exploitability is not measured to be low. These are development results only, not a final test. They do not support claims of strong or GTO play.

Large checkpoints remain in the ignored `runs/buckets-42/`.

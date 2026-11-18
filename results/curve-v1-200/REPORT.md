# Learning curve: buckets-v1, 200 BB (seed 42)

The protocol is the same as in `results/curve-v1-100/REPORT.md`:

- **Composite score:** the mean AIVAT-corrected bb/100 against six strategic pool-v1 bots.
- **Benchmark:** 3,000 paired deals, development seed 700000.
- **Stop:** when two consecutive cycles each improve the composite by less than 10 bb/100.

**Result:** stopped by the flat rule after 3 cycles.

| Iterations | Composite | Δ vs previous (95% CI) | nit | tag | min-raiser | third-pot | random-size | overbet | LBR (95% CI) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 527,245 (baseline) | 262.3 | – | 18 | 216 | 375 | 298 | 367 | 301 | 102 [−72, 275] |
| 1,348,732 | 278.0 | +15.7 [−12.7, 44.1] | 80 | 237 | 368 | 298 | 341 | 344 | 101 [−98, 299] |
| 1,830,242 | 273.9 | −4.1 [−21.7, 13.4] | 85 | 232 | 342 | 314 | 321 | 349 | 204 [−15, 423] |
| 2,327,893 | 260.0 | −13.9 [−28.8, 0.9] | 68 | 222 | 347 | 281 | 311 | 331 | 179 [−63, 421] |

The final checkpoint changed by −2.4 bb/100 against the baseline (95% CI −34.7 to +29.9, paired). The 1.35M checkpoint had the highest point estimate, +15.7 over the baseline, but its interval includes zero.

The clearest per-opponent change was against the nit. At the baseline the 95% CI was −39 to +75. After 1.35M iterations the CIs were 36 to 124, 40 to 130, and 27 to 108, all clearly positive. Against the bots that play loosely, the scores drifted slightly down.

The table covered about 98% of decisions at every point.

## Interpretation

The 200 BB curve is flat within the noise from about 1.3M iterations on. This run alone cannot show whether the later downward drift is real or noise. It reuses development deals, and picking the 1.35M checkpoint by its score would be selection on this same data.

The conclusion matches the 100 BB run: buckets-v1 has saturated against this benchmark. Further gains need a better abstraction or better evaluation, not more iterations.

None of this shows strong or GTO play, or low exploitability. The LBR intervals are wide and include values near zero.

Large checkpoints remain in the ignored `runs/buckets200-42/`.

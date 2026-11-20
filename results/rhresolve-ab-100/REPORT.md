# Real-history river re-solving (`--agent rhresolve`): development A/B, retired

**Decision (user, 2026-11-19): the re-solving line of work is dropped.** No 200 BB re-solving run, no further tuning. The code stays in git history only as an experiment; no model or README recommendation uses it.

## What was run
- Checkpoint: `runs/ptrain-v3-sizes-100/iteration-7780000.json.gz` (buckets-v3, sizes-v1, 100 BB).
- Development deals, seed 700000, 1,496 independent paired pairs per opponent, AIVAT, six scripted pool-v1 bots.
- Plain blueprint agent vs. `rhresolve` on the identical deals. `scripts/rhresolve_ab.sh`. LBR not run for the re-solver.

## Result (bb/100, AIVAT mean; difference = re-solver − blueprint)

| Opponent | Blueprint | Re-solver | Difference | Decisions re-solved |
|---|---|---|---|---|
| nit | +30.7 | +27.0 | −3.7 | 3.8% |
| tag | +180.8 | +130.9 | −49.9 | 5.7% |
| min-raiser | +203.5 | +163.1 | −40.4 | 9.8% |
| third-pot | +204.0 | +144.7 | −59.3 | 9.7% |
| random-size | +204.9 | +200.9 | −4.0 | 6.9% |
| overbet | +251.4 | +294.0 | +42.6 | 5.8% |

- Lower point estimate against five opponents (clearly lower against three, see paired intervals below), higher only against the overbet bot.
- Latency: median 0 ms; 95th percentile 0.9–4.6 s; slowest decision about 9 s (blueprint 95th percentile under 1 ms). Wall time 77 min vs about 5 min.
- Paired 95% intervals on the difference (`python -m cortado.compare`, clustered by deal; added after the first write-up): nit −3.7 [−27.0, +19.6]; tag −49.9 [−79.3, −20.4]; min-raiser −40.4 [−79.6, −1.2]; third-pot −59.3 [−103.3, −15.3]; random-size −4.0 [−39.2, +31.1]; overbet +42.6 [+6.3, +78.9]. Worse against three opponents, better against overbet, undecided against nit and random-size; intervals are not corrected for six comparisons.
- Cause not tested (a possible explanation is that range estimates are worse than the blueprint's own strategy against on-tree opponents).

## Conclusion
No evidence that history re-solving improves play overall; it costs seconds per decision. Not adopted.

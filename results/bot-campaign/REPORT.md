# Bounded CPU campaign completed

All four authorized train/evaluate cycles completed. Each worker stopped at its 30-minute training budget in each cycle, without a memory or table-disk stop. Both exit codes were zero. No campaign or worker processes remained at final inspection. The recurring 30-minute monitor was disabled; no additional campaign was launched and no resource limits were raised.

Final checkpoints:

- Seed 42: 523,923 iterations; sampled worker peak RSS in the final batch 88.92 MiB; working table 3,841,667,072 bytes before final snapshot.
- Seed 43: 565,390 iterations; sampled worker peak RSS in the final batch 88.93 MiB; working table 4,248,010,752 bytes before final snapshot, close to the 4 GiB threshold (4,294,967,296 bytes).

Each cycle evaluated each seed against call, aggressive, uniform and the historical 10k exact-card checkpoint with 2,000 independent paired deals per matchup. The same development seeds were reused across cycles. The 128,000 hands executed across all cycle evaluations are not 128,000 independent unique test hands. Intervals in the individual reports treat paired deals as the independent sampling unit; they are not corrected for multiple comparisons. This is development evaluation, not an untouched final test.

Both final candidates have positive individual 95% intervals against all four weak controls. That does not establish competitive strength, GTO play or low full-Hold'em exploitability. Uniform fallback remains substantial: against call, the final candidate miss fractions are about 53.2% (seed 42) and 52.5% (seed 43). No winner between seeds is selected from these reused, noisy development matchups. The suits-v1 representation preserves information but generalizes poorly; FCPA still restricts bet sizing.

Operational conclusion: bounded-cache disk storage supported the complete sustained train/evaluate campaign within the worker memory envelope. Disk growth, strategic coverage, independent strong opponents and full-action differential rules validation remain the next research gates. Seed 43 is near its authorized table-size cap; automatic continuation is inappropriate.

The large model files remain in ignored `runs/bot-campaign/seed-*/latest.sqlite`, with iteration-numbered snapshots alongside them. `checkpoint-identities.json` records SHA-256 hashes verified against the cycle-4 evaluation identities. This results directory preserves all cycle reports, campaign configuration/status, final metadata and metrics. No AWS resources were used.

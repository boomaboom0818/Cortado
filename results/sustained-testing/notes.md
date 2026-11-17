# Sustained CPU run: resource outcome

Two independent seeds resumed from 22,001 iterations with streaming checkpoint serialization, retaining the 1 GiB soft peak-RSS limit. Seed 42 reached 44,896 iterations; seed 43 reached 44,407. Both stopped with memory_budget and exit code 0. Both latest and iteration-numbered checkpoints were saved. Launcher wall time was 323.74 seconds, including bootstrap and worker save overhead. Detailed worker metrics are in training.json; checkpoint identities in sources.json.

Training is stopped. This is an operational resource result, not evidence of competitive strength. The representation still uses suit symmetry rather than strategic card buckets. Per-worker table growth is now the limiting concern. No automatic memory-limit increase or repeated resume is justified. Next work should compact the table/checkpoint loader or validate a strategic abstraction before another sustained run.

Development evaluation uses four weak controls, including the historical 10k checkpoint, 2,000 independent paired deals per matchup, seed 880000 and separate seed blocks per opponent. These are reused development deals and not untouched final testing. Do not select or claim strength based solely on these matchups. Individual uncertainty intervals use paired-deal clusters and are not simultaneous multiple-comparison intervals.

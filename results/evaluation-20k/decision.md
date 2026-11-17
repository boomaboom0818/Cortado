# Development evaluation and next-run decision

Evaluated the immutable 20,000-iteration seed 42 and 43 checkpoints on 2026-11-17. SHA-256 identities are in sources.json. Each candidate played four development opponents, with 5,000 independent paired deals (10,000 hands) per matchup: 80,000 hands overall. Seats are swapped within a pair; intervals use the pair as the independent sampling unit. The historical opponent is results/holdem-10k.json.gz. Results are in development.json. These seeds are development seeds, not untouched final-test data.

Both candidates have positive individual 95% intervals against these four controls. The controls are weak; this does not establish competitive strength or low exploitability. Confidence intervals are individual, not corrected for multiple comparisons. Differences between candidates do not justify discarding either seed. Against call, about 54% of decisions miss the table and use uniform fallback; this is a substantial unresolved coverage issue.

Decision: continue both seeds for one additional bounded batch from 20,000 toward 30,000 iterations, subject to the unchanged 120-second/worker limit and 1 GiB soft peak-RSS limit. Keep independent histories and retain the 20k checkpoints. Do not raise limits or loop indefinitely. Memory growth and fallback coverage require attention before larger runs. CLI: python -m doppio.parallel --config runs/cpu-pilot-config.json --run-dir runs/cpu-pilot --seeds 42 43 --resume.

## Actual next-run outcome

Both workers exited successfully after 35.83 seconds of launcher wall time. Each reached 22,001 iterations and stopped with `memory_budget`. The recorded process peak RSS was 1,067.1 MiB for seed 42 and 1,081.5 MiB for seed 43; 644,603 and 653,285 information sets respectively. These are high-water marks, including prior serialization. The 22,000 periodic save can raise this high-water mark and cause the next iteration to stop, even if current live memory has fallen. The configured threshold is soft, not a hard allocation cap. Each final latest checkpoint and iteration-numbered snapshot exists; report in next-run.json.

Training is stopped. Do not automatically resume again under the same conditions. Next engineering action: reduce table/checkpoint memory overhead or validate a coarser strategic card representation, then rerun a bounded resource pilot. Simply increasing CPU workers does not solve per-worker memory or fallback coverage. Existing 20k evaluation remains associated only with those checkpoint identities; the 22,001 checkpoints have not been evaluated here.

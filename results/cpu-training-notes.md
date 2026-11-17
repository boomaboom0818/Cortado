# Parallel CPU experiments

CPU runs use independent processes, not shared-table threads. The existing solver uses process-global NumPy RNG and Python traversals. Parallel independent runs preserve the validated algorithm and provide separate candidate strategies. This is not parallel traversal aggregation for one model.

`cpu-concurrency.json` contains the reproducible 1/2/4-worker sweep, 1,000 iterations per worker. The sweep used the i7-9750H host, not the user's 9950X, with one BLAS/OpenMP thread per process. Initialization, serialization and disk writes are included in launcher wall time. The pilot is short and not a stable prediction of overnight speed; thermal throttling and table growth can change it. Summed worker peak RSS is not a time-aligned total or a hard memory bound.

`cpu-development-smoke.json` exercises the completed-worker evaluator against call/aggressive/uniform/a historical checkpoint, using development deals and 100 pairs per matchup. This is a software verification run with insufficient evidence for strength conclusions.

Use two workers and a 1 GiB per-worker soft threshold initially on this 15 GiB desktop host. The table remains large and grows with training, and checkpoint conversion creates additional allocations. Do not launch a large worker count on the 9950X until its RAM is measured and this sweep is repeated there. No AWS spending is required. A hard resource cap is not implemented by the Python soft limits.

Sustained launch and resume commands are in README.md. Ctrl-C/SIGTERM are tested to forward shutdown and save complete iterations. Worker nonzero exits are reported; launcher does not silently restart a failed seed. Histories stay isolated and cannot be merged by averaging regret tables. Human mode remains deferred.

The remaining competitive-training gates are strategic/draw-aware abstraction, stronger independent opponents, independent rules verification, and untouched final evaluation. Running longer does not resolve these automatically. This work makes the current baseline operational for bounded sustained experiments, not proven competitive play.

The longer two-worker pilot completed 10,000 iterations per seed (42 and 43), 20,000 total, in 77.47 seconds of launcher wall time. Both exit codes were zero; periodic checkpoints at 2,000-iteration intervals succeeded. Worker-reported peak RSS before final serialization was 332.0 and 346.0 MiB, with 290,020 and 303,631 information sets. These are bounded pilots, not a forecast of memory saturation or evidence of playing strength. Full report: cpu-sustained-pilot.json. Total automated tests: 25 passed.

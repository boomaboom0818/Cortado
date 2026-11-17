# Sustained CPU training readiness

This is an operational readiness gate for reproducible research training, not a certification of competitive poker strength.

## Decision and resource envelope

Use OpenSpiel external-sampling MCCFR, FCPA actions, suits-v1 information sets and two independent CPU worker processes. Each seed has its own exact float64 strategy table. The table now lives in SQLite with a 4,096-row Python cache and an 8 MiB SQLite page-cache target per connection. Historical JSON snapshots remain supported. No neural rewrite or lossy card-bucket change is required to fix the memory stop, and no regret tables are merged.

The host seen here is i7-9750H with 15 GiB RAM; this is not the user's 9950X. No AWS resources are used. Defaults: two seeds, 30-minute maximum per training invocation, 1 GiB soft live-RSS threshold per worker, 4 GiB soft working-table limit, 2 GiB minimum free disk, and a checkpoint every 2,000 iterations. Linux `/proc` supplies live RSS. Sampled peak and OS high-water are recorded separately because a forked launcher's memory can contaminate the inherited OS high-water value. Transient allocations between samples can be missed. These are soft limits, not kernel-enforced caps.

SQLite snapshots are self-contained `.sqlite` files containing metadata, RNG state and table rows. Backup uses a new temporary database, fsync and atomic replacement; partial saves never replace the last successful snapshot. Working databases are private to a trainer and can be discarded after a crash; only complete checkpoint files resume an iteration exactly. In-place updates are flushed on LRU eviction. Cache eviction order never changes arithmetic, sampling or action semantics. Checkpoint tables are opened read-only by policies. Files use schema-3 model metadata plus storage_schema=1. Old `.json.gz` checkpoints remain readable. A legacy-to-disk bootstrap can temporarily consume substantial memory in the launcher while decoding the old JSON; that is a one-time migration, not worker memory growth.

## Gates

- Reference equality: disk and memory tables/RNGs match on Kuhn, Leduc and Hold'em with split-run resume.
- Eviction correctness: Hold'em traversals with a one-row cache match the reference table exactly.
- Recovery: invalid saves preserve the last checkpoint; hard exit from a separate working database leaves the snapshot unchanged; SIGTERM forwarding is covered by the existing launcher tests.
- Compatibility: game/specification, representation, runtime and seed checks remain enforced.
- Resource behavior: resumed large-table pilots must finish without repeated memory stops; reported memory includes live RSS, sampled peak and diagnostic OS high-water.
- Workflow: the campaign must complete two short train/evaluate/resume cycles; resource stops must prevent further cycles.

## Launch and supervision

After the readiness pilot passes:

```sh
.venv/bin/python -m doppio.campaign --config configs/train-disk.json --run-dir runs/bot-campaign --bootstrap-from runs/disk-readiness --cycles 4 --pairs 2000
.venv/bin/python -m doppio.status --run-dir runs/bot-campaign
```

At most four train/evaluate cycles run, giving at most two hours of training wall budget plus initialization, checkpointing and evaluation. Each cycle saves training and development-evaluation reports, immutable iteration snapshots, current checkpoints and campaign status. Healthy completed/time-budget cycles continue automatically. Memory, disk, signal or worker failures halt escalation. A 4 GiB table stop is a research scaling gate, not a reason to raise the limit automatically. A terminal campaign returns its saved state rather than restarting. The 30-minute T3 checks inspect worker and campaign processes, checkpoint ages, resource use, logs and outcomes. They require T3 and this machine to stay available.

Development evaluations use 2,000 paired deals per candidate/opponent against call, aggressive, uniform and the historical 10k exact-card checkpoint. They reuse development seeds and are not a final untouched test set. Reports include checkpoint hashes and actual iteration counts. Confidence intervals cluster by paired deals. Do not select a model solely because it beats these weak controls; full Hold'em exploitability is not measured.

## Remaining research limits

Moving the table to disk fixes RAM growth, not strategic coverage. Suit symmetry alone generalizes poorly; misses still fall back to uniform. The strategy still uses a restricted betting abstraction. A strategic card abstraction, stronger independent opponents, and rigorous full-action differential rules validation remain separate research requirements. Sustained experiments should measure coverage and performance before claiming stronger play. Disk I/O and storage growth may eventually become limiting; the bounded campaign stops rather than running indefinitely.

## Recorded gate results

The automated suite passes 40 tests. The migrated two-worker disk pilot completed an additional 3,000 iterations per seed after correcting the inherited-high-water guard. The subsequent readiness campaign completed two train/evaluate/resume cycles of 5,000 iterations per worker per cycle, retaining checkpoints and development reports. No memory or disk stop occurred in those cycles. Recorded metrics and reports are in `results/disk-readiness/`; large database artifacts remain in ignored `runs/`. The readiness evaluation uses only 100 paired deals per matchup as a workflow check, so its winnings estimates are too uncertain for strength conclusions. The full campaign uses 2,000 pairs instead.

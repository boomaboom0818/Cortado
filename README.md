# Cortado heads-up poker research pipeline

Working research prototype, **not a competitive poker bot yet**. Exclusively two players, 100 BB reset stacks, 0.5/1 blinds, no rake. Seat 0 is button/small blind, acts first preflop; seat 1 acts first thereafter. Engine chips are half a BB. Evaluation swaps policies between these seats on repeated deals.

## Models

The frozen models are **C** (100 BB) and **D** (200 BB) in `models/`: bet-size abstraction `sizes-v1` with exact
`buckets-v3` card buckets. On reserved deals each beats its predecessor head to head (C vs A +30 bb/100, D vs B +33)
and beats all nine scripted pool bots, and LBR found no profitable exploit. Their card-bucket tables ship compressed
in the repo and unpack automatically on first use. The earlier models A and B are kept.
Strength against strong opponents is unproven. See `models/README.md` for options, usage and limits, and
`results/final-test-2/REPORT.md` for the numbers.

## Install and run

Python 3.12 is pinned for binary package reproducibility; `uv` can install it.

```sh
uv venv --python 3.12
uv pip install -r requirements.lock
.venv/bin/pytest -q
.venv/bin/python -m cortado.cli train --game kuhn --iterations 10000 --seed 42 --checkpoint results/kuhn.json
.venv/bin/python -m cortado.cli train --game holdem --iterations 100000 --seed 42 --checkpoint holdem-100k.json
.venv/bin/python -m cortado.cli evaluate --game holdem --policy checkpoint --checkpoint holdem-100k.json --opponent call --pairs 10000 --seed 100000 --mode bot
```

Repeat evaluation with `--opponent uniform`, `--opponent aggressive`, and `--opponent checkpoint --opponent-checkpoint <historical.json>`. Use disjoint evaluation seeds and freeze evaluation opponents before tuning. Training uses self-play, not these scripted opponents. Bot mode (`--mode bot`, the default) is the active target. Human mode is deferred and `--mode human` now exits with an explicit error. The JSON files in `configs/` document the profiles defined in `cortado/profiles.py`; they are descriptive exports, not runtime overrides. No adaptation is enabled. External platform integration is intentionally absent.

## Architecture and API

- `game.py`: trusted OpenSpiel engine and frozen `Observation` boundary. `Policy.act(observation, numpy_rng)` returns a legal integer action. No state reference, opponent cards, deck, or simulator handle crosses this boundary. This is an API isolation boundary, not a sandbox for malicious Python plugins.
- `policies.py`: average checkpoint strategy plus call, aggressive and uniform evaluation controls. Unseen information sets explicitly fall back to uniform and are counted.
- `training.py`: OpenSpiel external-sampling MCCFR, validated first on Kuhn. The solver legitimately traverses hidden states internally; deployed policies cannot see them.
- `opponents.py`: blocker-aware probability distribution over two-card holdings and conservative Bayesian likelihood updates. It is not yet wired to betting decisions. Action likelihood calibration and opponent persistence remain work.
- `evaluation.py`: independent paired-deal clusters with seat swaps, Student-t confidence interval over cluster means. Latencies cover policy calls across both players, excluding engine and observation construction. Cost is wall time on local CPU, no monetary estimate.

Action IDs in Hold'em: 0 fold, 1 check/call, 2 pot-size bet/raise, 3 all-in. Only listed legal actions may be returned. This FCPA action abstraction omits most legal bet sizes; the public API currently supports this restricted game only. No card abstraction is used, so table coverage scales extremely poorly. All-in legality and settlement are delegated to ACPC. Arbitrary raises, explicit short-all-in reopening and constructed tie regression fixtures still need independent validation before production use.

Legacy checkpoints are schema-1 JSON with engine version, game specification, seed, iteration count, runtime and information-set keys mapping to `[regrets, cumulative_average_weights]`. Schema-1 files are inference-only. New schema-2 files include RNG state and support exact solver restoration via `--resume`; saves use atomic replacement. Never substitute Kuhn checkpoints for Hold'em. Engine versions and action abstraction must remain fixed.

## Actual verification

The initial six tests covered: button order/fold payouts, postflop order, randomized legal trajectories and zero-sum stack bounds, Kuhn convergence, range normalization/blockers, and deterministic paired evaluation.

Committed reports and checkpoints are in `results/`. Seed 42, 10,000 Kuhn iterations yielded exact exploitability 0.0076424254 chips/game in approximately 1.87 CPU-wall seconds. This is OpenSpiel's exact small-game best-response calculation, not Hold'em exploitability.

The ten-iteration Hold'em checkpoint is a smoke test only: 384 information sets, 0.0204 seconds training. Evaluation versus call with seed 100000 used 100 independent pairs (200 hands): -43.75 BB/100, 95% interval [-694.35, 606.85]. There were 418 checkpoint misses, so this essentially evaluates fallback play. No competitive-strength conclusion follows. Timing is machine-specific (12 logical CPUs, 15 GiB RAM available host; single-process training). Full Hold'em exploitability is not measured.

## Staged plan and remaining requirements

1. Completed: trusted engine wrapper, isolated policy API, small-game CFR validation, schema-versioned snapshots, local paired evaluation.
2. Next correctness gate: full-action adapter and independent reference-engine differential tests for minimum raises, short all-ins, board ties and payouts; stronger hidden-information invariance tests; checkpoint compatibility and resume support.
3. Scaling gate: implement and validate suit canonicalization and equity/strategic card buckets, preserve relevant action history, then train MCCFR or Deep CFR with held-out validation. The present exact-card table is not an efficient competitive training representation. Select accelerated PyTorch/JAX training only after profiling and algorithm experiments; no runtime preference dictated the current solver choice.
4. Strength gate: budget measured training runs with increasing traversal counts, multiple seeds, historical checkpoints and stronger independent agents. No defensible fixed iteration count or compute budget for competitive strength can be inferred from this smoke test. Measure coverage, learning curves and confidence intervals before allocating larger CPU/GPU runs.
5. Adaptation/search gate: fit calibrated public-action likelihoods, update opponent hand ranges, evaluate conservative exploitation and real-time subgame solving under latency budgets. No heuristic policy blend constitutes a safety guarantee. Human and bot modes must share these improvements without weakening human mode.
6. Later: validated configurable stacks, rake accounting, JSON transport for the explicit API. No LLM betting engine.

## Dependencies and licensing

OpenSpiel 2.0.2 supplies both CFR and ACPC-backed universal poker: https://github.com/google-deepmind/open_spiel (Apache-2.0 project; bundled ACPC has its own notices under `open_spiel/games/universal_poker/acpc`). Preserve dependency notices when distributing binaries. Engine source: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/universal_poker/universal_poker.cc . PokerKit (MIT, https://github.com/uoftcprg/pokerkit/blob/main/LICENSE) was considered as an independent rules oracle, but is not a runtime dependency. This repository adds no separate license grant yet.

Branches: `main` is the initialization baseline; implementation is on `develop`.


## Correctness and training follow-up

The current suite has 13 passing tests, adding full-action raise bounds, short-all-in responses, a board royal-flush split pot, hidden-opponent-card invariance, checkpoint compatibility rejection, exact split-run resumption, and suit relabeling invariance.

`game(full_actions=True)` and evaluation `--full-actions` expose every legal integer-chip raise (half-BB granularity). In this encoding, 0 is fold, 1 is call/check, and raise IDs are ACPC total-commitment amounts. Use engine action descriptions; do not interpret these as FCPA IDs. FCPA checkpoints are rejected for this game. Full-action CFR training remains disabled because its branching factor is impractical. This is not arbitrary sub-chip sizing. Independent differential rules validation remains outstanding.

```sh
.venv/bin/python -m cortado.cli evaluate --game holdem --full-actions --pairs 100
.venv/bin/python -m cortado.cli train --game holdem --iterations 1000 --resume results/holdem-10k.json.gz --checkpoint resumed.json
.venv/bin/python -m cortado.benchmark results/holdem-10k.json.gz --historical results/holdem.json --pairs 2000
```

The committed 10,000-iteration run took 33.75 seconds of training-loop wall time and produced 320,170 information sets. Its compressed checkpoint and the four-opponent, 16,000-hand benchmark report are committed. Serialization, loading, and validation costs are not included in training-loop or decision latency measurements. Benchmark intervals are individual 95% intervals, not simultaneous multiple-comparison guarantees. These controls are weak, and substantial fallback usage persists; competitive strength remains unproven.

The versioned suit-relabeling adapter in `abstraction.py` is now integrated into both training and inference; see the sustained-training instructions below. Equity buckets, calibrated action likelihoods, adaptive betting and decision-time search are still pending. The next research milestone is stronger card abstraction and learning curves across multiple seeds. The future human profile will reuse the foundation without reducing difficulty.


## Bot-first development targets

There is one codebase and a shared poker/CFR foundation. The profiles identify opponent populations, not easy/hard settings. We have not trained two distinct models.

- **Bot target (active):** self-play foundation first; frozen bot versions and historical checkpoints are the evaluation population. Future modeling should infer a distribution over opponent strategies and hands from public actions, conditioned on bot version where legitimately known. Exploitation must earn its place through held-out evidence. Generalization to unseen bot versions is the key test.
- **Human target (deferred):** initialize from the validated foundation later, then model individual/session tendencies, uncertainty, sparse observations and behavior changes. Evaluate on held-out opponents and sessions. Do not assume all humans share tendencies, lower difficulty, or optimize entertainment.

New training checkpoints and CLI evaluation reports record a `profile` identifier. Existing checkpoints predate this metadata and are treated as bot-foundation artifacts for resumption. Opponent data for the future human target must use a separate namespace and must not contaminate frozen bot evaluation sets. No opponent-history persistence is implemented yet.

Bot-first milestones: train multiple reproducible seeds using the validated adapter; improve coverage and benchmark against stronger independent bots; only then enable calibrated exploitation. Human-specific adaptation comes after these gates. Both profiles maximize long-run expected winnings, and neither uses an LLM as a betting engine.


## Sustained training readiness

The pipeline is ready for **bounded, restartable bot training experiments**. It is not a claim that a complete competitive Hold'em training run fits this host. The FCPA action restriction remains; full-action evaluation support does not imply full-action training support.

```sh
# Install and run correctness gates first.
uv venv --python 3.12
uv pip install -r requirements.lock
.venv/bin/pytest -q

# One bounded training invocation; writes under ignored runs/.
.venv/bin/python -m cortado.run --config configs/train-bot.json --run-dir runs/bot-seed42
# Continue from exactly the last completed saved iteration.
.venv/bin/python -m cortado.run --config configs/train-bot.json --run-dir runs/bot-seed42 --resume

# Evaluate separately from training; never tune on the frozen final test suite.
.venv/bin/python -m cortado.benchmark runs/bot-seed42/latest.json.gz --historical results/holdem-10k.json.gz --pairs 10000
```

`configs/train-bot.json` specifies 100,000 **additional** iterations per invocation, a one-hour wall budget, a 4 GiB peak-RSS soft threshold and a checkpoint every 1,000 iterations. Whichever stopping condition occurs first ends the invocation. Resume uses the immutable run configuration and the checkpoint RNG, not a new seed. Copy the config and use independent run directories for seeds 43 and 44; results from independent runs are not merged into one CFR table. Start sequentially until measured RAM requirements justify concurrency.

Each run directory contains its immutable config, `latest.json.gz`, append-only metrics, summary, and an iteration-numbered historical checkpoint after each invocation. An advisory lock prevents simultaneous trainers in that directory. Writes use compression, fsync and atomic replacement. SIGINT/SIGTERM finish the current iteration before saving; SIGKILL, crashes and power loss can lose work since the last checkpoint. A completed save remains resumable. Direct `cortado.cli train` is a lower-level interface without the run-directory lock; prefer `cortado.run` for long runs.

Budgets are **soft iteration-boundary checks**, not hard allocation limits. Peak RSS is the process lifetime high-water mark on Linux. Serialization copies the table and needs extra RAM; reserve at least comparable free memory above the table footprint. Wall budgets can be exceeded by an iteration or checkpoint save. The 1 GiB disk preflight is only a minimum, not a forecast; monitor disk use and historical snapshot growth. Logs record peak RSS, wall time, checkpoint size, iteration count, representation and runtime versions. Checkpoint serialization time is included in subsequent cumulative wall measurements but not the snapshot that is being saved. Exact schema-3 resume requires matching Python/NumPy versions; pinned engine compatibility is also checked.

New schema-3 checkpoints explicitly distinguish `exact-v1` from `suits-v1`; legacy checkpoints remain exact. `suits-v1` chooses a canonical suit labeling, retaining ranks, card order, public history and betting information; ordered legal action IDs are part of the key. This is a lossless symmetry reduction, not equity bucketing, and gives at most a 24-fold reduction. It does not forget public streets or invent access to hidden cards. Changing representation requires a fresh run. Deployment automatically uses the checkpoint's representation. Tests verify split-run equality, inference lookup, information isolation, save failure recovery, signal handling and resource stops.

The readiness pilot uses seed 123, 1,000 suit-abstracted iterations and four held-out opponent evaluations. Reports are in `results/readiness-*.json`; the local pilot checkpoint is in ignored `runs/readiness/`. Reproduce with:

```sh
.venv/bin/python -m cortado.cli train --game holdem --iterations 1000 --seed 123 --representation suits-v1 --checkpoint runs/readiness/latest.json.gz --checkpoint-every 250 --max-seconds 60 --max-rss-mb 1024 --metrics runs/readiness/metrics.jsonl
.venv/bin/python -m cortado.benchmark runs/readiness/latest.json.gz --historical results/holdem-10k.json.gz --pairs 500
```

Before committing large compute: measure coverage, RAM growth and iterations/second at several run lengths and seeds; evaluate against stronger independent agents; select and validate a strategic/equity abstraction or neural CFR implementation if table growth is unacceptable. Those research choices remain necessary for competitive-scale training. No honest total training-hours estimate follows from this pilot, and no full Hold'em exploitability claim is made. Opponent adaptation remains disabled pending calibration.

## Parallel CPU training

The CPU launcher runs independent MCCFR seeds in separate processes. Each worker has a separate regret table, RNG, run directory, log, and checkpoint. This uses multiple cores without sharing Python RNG state or concurrently modifying a CFR table. It increases aggregate experiment throughput; it does not divide one strategy's learning time by the worker count. Do not average regret tables from independent runs.

```sh
# Conservative starting point for this 15 GiB host: two independent workers.
.venv/bin/python -m cortado.parallel --config configs/train-cpu.json --run-dir runs/cpu-training --seeds 42 43
# Continue each seed for another bounded invocation.
.venv/bin/python -m cortado.parallel --config configs/train-cpu.json --run-dir runs/cpu-training --seeds 42 43 --resume
# Reproduce the concurrency sweep, using a new directory each time.
.venv/bin/python -m cortado.cpu_experiment --run-dir runs/cpu-sweep-new --iterations 1000
```

The supplied CPU config uses a one-hour maximum per worker, 100,000 additional iterations, a 1 GiB **soft** peak-RSS threshold per worker, and a save every 1,000 iterations. Serialization and loading can exceed that memory threshold. Use two workers initially, keep the desktop's memory reserve, and inspect logs before adding workers. Set BLAS/OpenMP threads to one per worker (the launcher does this). On the 9950X, first rerun the sweep and measure available RAM; this workspace is a different computer.

SIGINT/SIGTERM to the launcher forwards SIGTERM to running workers, which finish their current iteration and save. The launcher waits for them and records exit codes. A hard kill cannot guarantee the latest work. Resume requires an identical experiment manifest and an existing checkpoint for every seed. Errors are visible in `seed-*.log` and through a nonzero launcher exit. The outer directory lock prevents duplicate launches. Each worker also retains the original run lock. The launcher launches one bounded invocation and exits; it does not silently restart forever.

Evaluate candidates using development seeds and a frozen opponent pool. `cortado.cpu_experiment` evaluates its two-worker candidates against the calling control with development seed 880000; this is a pipeline/coverage diagnostic, not evidence of competitive play. The table still grows rapidly and unseen states fall back to uniform actions. Suit abstraction is not a strategic card bucket. Sustained runs are operationally supported, but competitive-scale representation, independent opponent integration and final untouched evaluation remain research tasks.

Parallel launcher tests cover independent worker state, resume, config rejection, and SIGTERM forwarding with readable completed checkpoints. Leduc validation is now available through `--game leduc` and reports exact small-game exploitability.

To evaluate every completed worker against the same development pool (use separate final-test seeds later):

```sh
.venv/bin/python -m cortado.evaluate_workers --run-dir runs/cpu-training --historical results/holdem-10k.json.gz --pairs 2000
```

Run evaluation while trainers are stopped to avoid loading large tables during checkpoint serialization. The paired deals are shared across candidates for comparable conditions; each report's uncertainty remains computed over independent deal pairs. Do not select a candidate solely from one noisy matchup.

## Sustained testing with 30-minute checks

Checkpoint serialization now converts NumPy arrays one row at a time instead of duplicating the entire strategy table. Resume releases JSON rows while converting them. The schema and exact strategy/RNG values remain compatible; unit tests cover save/load and exact resumption. JSON loading and table growth still consume substantial memory.

A fresh experiment can continue existing checkpoints under new explicit resource settings without modifying the original experiment manifest:

```sh
.venv/bin/python -m cortado.parallel --config configs/train-sustained.json --run-dir runs/cpu-sustained --seeds 42 43 --bootstrap-from runs/cpu-pilot
.venv/bin/python -m cortado.status --run-dir runs/cpu-sustained
```

The sustained config permits one 30-minute training invocation per worker, up to one million additional iterations, with the existing 1 GiB soft memory limit. It does not raise the resource cap or automatically restart after memory stops. The lightweight status command reports process RSS, checkpoint age, latest saved metrics and disk space without loading strategy tables. `summary` can be from an earlier invocation; active processes and timestamps must be checked too.

The requested 30-minute checks are scheduled through T3 separately from the Python trainer. They require this computer and T3 to stay available. A scheduled check should inspect health/logs, stop on resource or correctness failures, evaluate completed checkpoints sequentially, and record results. Never repeatedly resume a memory-stopped run without an explicit engineering change. Continuous wall-clock observation is not a hard memory cap; trainer resource checks remain iteration-boundary checks.

## Current sustained-run path: disk-backed CFR

Use the bounded CPU campaign in [TRAINING_READINESS.md](TRAINING_READINESS.md). It replaces the RAM-growing path for new long runs while retaining the existing checkpoints and exact training algorithm. The new optional run-config field `storage: "sqlite"` selects self-contained `latest.sqlite` and iteration snapshots; omit it for legacy JSON/in-memory behavior. Training and policy inference both support this format. Each worker uses a bounded row cache and its own working database. The campaign performs development evaluation between training batches and stops on resource failures instead of automatically restarting them.

`configs/train-disk.json` provides two-worker-compatible limits for this host. `cortado.campaign` is capped at four cycles; use the documented command and a fresh run directory. Do not substitute `.sqlite` filenames with gzip files or move active working databases between workers. The 1 GiB soft memory guard now uses live Linux RSS, with sampled peak and inherited OS high-water reported separately.


## Strength gates: coverage, full-action play, adversaries, external benchmark

This section supersedes older statements above about uniform fallback and the FCPA-only public API.

- **Card abstraction (`abstraction.py`, `buckets.py`, `cards.py`)**: `buckets-v1` keeps 169 lossless preflop classes and replaces postflop cards with 30 equity buckets per street (equity vs a random hand, deterministic seeded Monte Carlo, versioned edges in `cortado/data/buckets-v1.json`, digest checked at load). `buckets-v2` adds a public board-texture class (flush potential x paired board). Imperfect recall: earlier postflop buckets are forgotten. The table drops from billions of potential keys to about 43.5k (v1), so trained-table coverage is about 100% instead of about 47%.
- **Full-action play (`fullaction.py`)**: `FullActionAgent` plays any FCPA checkpoint in the unrestricted game. Opponent bet sizes are mapped by pseudo-harmonic translation (seeded per own cards and public prefix); our abstract choices map to concrete legal chip amounts. Stateless and isolated: it rebuilds an abstract replay from the observation with dummy opponent cards. Known limits: translation is not subgame solving; real and abstract pots drift; mapped all-ins can desynchronise the replay, after which an equity heuristic plays (counted as `desync_heuristic`).
- **Configurable stacks**: `--stack-bb` / `stack_bb` in training configs, recorded in checkpoints and enforced at evaluation.
- **Opponents (`pool.py`, `lbr.py`)**: `pool-v1` is nine frozen diagnostic scripts (nit, tag, lag, station, maniac, min-raiser, third-pot, overbet, random-size) with real-chip sizing. `LBR` is a Local Best Response adversary using the candidate's exact strategy. Neither is an independent strong agent.
- **Benchmark (`fullbench.py`)**: parallel paired-deal blocks, per-opponent t-intervals, candidate-only latency, decision-source fractions, and optional `--aivat`.
- **Variance reduction (`aivat.py`)**: chance-event and own-action control variates. Unbiasedness checked exactly for chance terms (unit test) and empirically: on 8,000 paired deals, mean correction -18.8 bb/100, SE 18.1. Variance ratio about 0.39 against `tag`, so about 2.6x fewer deals are needed for the same interval. Weights were tuned on one seed range and validated on another.
- **Slumbot (`slumbot.py`)**: client for the hosted 200 BB API (wire format from Slumbot's published `sample_api.py`). It replays every finished hand in our engine with Slumbot's revealed cards and checks the payoff, and it reports Slumbot's `baseline_winnings`-adjusted estimate.

```sh
.venv/bin/python -m cortado.fullbench runs/buckets-42/latest.json.gz --pairs 3000 --lbr-pairs 300 --aivat --seed 700000 --out dev.json
.venv/bin/python -m cortado.slumbot runs/buckets200-42/latest.json.gz --hands 200 --log slumbot.jsonl   # contacts slumbot.com
```

Results and their limits are in `results/strength-gates/REPORT.md`, `results/slumbot/` and `EVALUATION_PROTOCOL.md`. Competitive strength remains unproven.

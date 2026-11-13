# Doppio heads-up poker research pipeline

Working research prototype, **not a competitive poker bot yet**. Exclusively two players, 100 BB reset stacks, 0.5/1 blinds, no rake. Seat 0 is button/small blind, acts first preflop; seat 1 acts first thereafter. Engine chips are half a BB. Evaluation swaps policies between these seats on repeated deals.

## Install and run

Python 3.12 is pinned for binary package reproducibility; `uv` can install it.

```sh
uv venv --python 3.12
uv pip install -r requirements.lock
.venv/bin/pytest -q
.venv/bin/python -m doppio.cli train --game kuhn --iterations 10000 --seed 42 --checkpoint results/kuhn.json
.venv/bin/python -m doppio.cli train --game holdem --iterations 100000 --seed 42 --checkpoint holdem-100k.json
.venv/bin/python -m doppio.cli evaluate --game holdem --policy checkpoint --checkpoint holdem-100k.json --opponent call --pairs 10000 --seed 100000 --mode bot
```

Repeat evaluation with `--opponent uniform`, `--opponent aggressive`, and `--opponent checkpoint --opponent-checkpoint <historical.json>`. Use disjoint evaluation seeds and freeze evaluation opponents before tuning. Training uses self-play, not these scripted opponents. Bot mode (`--mode bot`, the default) is the active target. Human mode is deferred and `--mode human` now exits with an explicit error. The JSON files in `configs/` document the profiles defined in `doppio/profiles.py`; they are descriptive exports, not runtime overrides. No adaptation is enabled. External platform integration is intentionally absent.

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
.venv/bin/python -m doppio.cli evaluate --game holdem --full-actions --pairs 100
.venv/bin/python -m doppio.cli train --game holdem --iterations 1000 --resume results/holdem-10k.json.gz --checkpoint resumed.json
.venv/bin/python -m doppio.benchmark results/holdem-10k.json.gz --historical results/holdem.json --pairs 2000
```

The committed 10,000-iteration run took 33.75 seconds of training-loop wall time and produced 320,170 information sets. Its compressed checkpoint and the four-opponent, 16,000-hand benchmark report are committed. Serialization, loading, and validation costs are not included in training-loop or decision latency measurements. Benchmark intervals are individual 95% intervals, not simultaneous multiple-comparison guarantees. These controls are weak, and substantial fallback usage persists; competitive strength remains unproven.

A tested suit-relabeling primitive is available in `abstraction.py`, but is deliberately not integrated into the solver: canonical keys must be used consistently in both training and deployment, with a new checkpoint identity and convergence tests. Equity buckets, calibrated action likelihoods, adaptive betting and decision-time search are still pending. The next engineering milestone is an abstract-game adapter with validated information-set/action consistency, followed by learning curves across multiple seeds. The future human profile will reuse the foundation without reducing difficulty.


## Bot-first development targets

There is one codebase and a shared poker/CFR foundation. The profiles identify opponent populations, not easy/hard settings. We have not trained two distinct models.

- **Bot target (active):** self-play foundation first; frozen bot versions and historical checkpoints are the evaluation population. Future modeling should infer a distribution over opponent strategies and hands from public actions, conditioned on bot version where legitimately known. Exploitation must earn its place through held-out evidence. Generalization to unseen bot versions is the key test.
- **Human target (deferred):** initialize from the validated foundation later, then model individual/session tendencies, uncertainty, sparse observations and behavior changes. Evaluate on held-out opponents and sessions. Do not assume all humans share tendencies, lower difficulty, or optimize entertainment.

New training checkpoints and CLI evaluation reports record a `profile` identifier. Existing checkpoints predate this metadata and are treated as bot-foundation artifacts for resumption. Opponent data for the future human target must use a separate namespace and must not contaminate frozen bot evaluation sets. No opponent-history persistence is implemented yet.

Bot-first milestones: integrate and validate the abstraction adapter; train multiple reproducible seeds; improve coverage and benchmark against stronger independent bots; only then enable calibrated exploitation. Human-specific adaptation comes after these gates. Both profiles maximize long-run expected winnings, and neither uses an LLM as a betting engine.

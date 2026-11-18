# Strength-evidence protocol (draft; nothing here has been run as a final test)

Purpose: decide, before looking at results, what would justify a competitive-strength claim. Anything run
under `--seed` blocks listed as *development* below may be used to choose among candidates and must never
be reported as final evidence.

## Gates

| Gate | Evidence required | Status |
|---|---|---|
| G1 Unrestricted sizes | Candidate plays the full-action game through `FullActionAgent`; zero illegal actions in fuzzing; desync/fallback rate reported per opponent; sizing-probe opponents and LBR included | Implemented; results in `results/strength-gates/` |
| G2 Coverage | Fraction of decisions answered from the trained table (not a fallback) reported per opponent | Implemented |
| G3 Independent opponents | At least one externally maintained, independently trained agent at matching stack/blinds/rake | Slumbot client working (200 BB); 200-hand pilot only, inconclusive |
| G4 Frozen test | Candidate checkpoint hash, adapter version, opponent versions, seed block, pair count and acceptance threshold written down before the run | Template below |
| G5 Adversarial search | LBR (implemented) plus trained best-response agents against the frozen candidate on fresh deals | LBR only |
| G6 Variance | AIVAT or equivalent so the required sample is feasible | Local AIVAT (variance ratio ~0.39); Slumbot baseline adjustment available |

## Seed blocks

* Development: `--seed 650000`–`899999` (pool, LBR, historical checkpoint comparisons).
* Frozen final: `--seed 5000000` upward. Reserved; unused as of this file's writing.

## Frozen-test template (fill in before running, commit, then run once)

```
candidate checkpoint sha256:
adapter version (git commit):
pool version: pool-v1        external opponents + versions:
stack / blinds / rake:       100 BB or 200 BB / 0.5-1 / none
seed block:                  pairs per opponent:
acceptance rule:             lower 95% bound of mean bb/100 over the named pool > threshold
multiplicity:                per-opponent intervals are NOT simultaneous; use Bonferroni over k opponents if a pooled claim is made
stopping rule:               fixed pair count; no optional stopping
```

## Sample-size reality check

Per-deal (paired) standard deviation is roughly 25-30 big blinds in these matchups. A 95% half-width of
w bb/100 needs about (1.96 * 27 / (w/100))^2 deals: w = 20 needs about 70,000 deals; w = 5 needs about
1.1 million. Plan sample sizes from the pilot standard deviation of the actual matchup, and expect to
need variance reduction (G6) for tight bounds.

## External opponents

All publicly reachable strong HUNL benchmarks found play 200 big blinds: Slumbot, the GTO Wizard benchmark
service, and OpenHoldem-hosted baselines. `doppio/slumbot.py` plays Slumbot through its public API (no login
required; anonymous sessions do not appear on leaderboards). Every finished hand is replayed in our engine
with Slumbot's revealed cards; all 200 pilot hands matched. Slumbot is not a duplicate opponent: hands are
independent and unpaired, but the API returns `baseline_winnings` (Slumbot playing our seat with the same
cards), which gives an unbiased lower-variance difference estimator. Be polite with request volume.

## Pre-registration: Slumbot test 1 (committed before any hand was played)

```
candidate checkpoint:        runs/buckets200-42/iteration-527245.json.gz  (buckets-v1, stack_bb=200, 527,245 iterations)
candidate sha256:            a7f80e86251b8c48ffc42567bffac974686080220ba197809d594dc4ceec8084
selection:                   best of the three 200 BB candidates against the development pool on
                             nit/tag/min-raiser/third-pot/random-size (results/strength-gates/dev-*.json)
adapter / client version:    git 5f10a90 (FullActionAgent heuristic fallback; doppio/slumbot.py)
opponent:                    hosted Slumbot API, anonymous session, version not disclosed by the provider
hands:                       1,000 (even; fixed, no optional stopping); action seed 3
primary metric:              baseline-adjusted bb/100 (winnings - baseline_winnings) with 95% t-interval
secondary:                   raw bb/100 with 95% t-interval; protocol checks; decision sources
claim rule:                  "beats Slumbot at 200 BB (this sample)" only if the primary lower bound > 0;
                             "loses" only if the primary upper bound < 0; otherwise inconclusive
```

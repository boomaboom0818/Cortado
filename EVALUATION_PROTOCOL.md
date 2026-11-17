# Strength-evidence protocol (draft; nothing here has been run as a final test)

Purpose: decide, before looking at results, what would justify a competitive-strength claim. Anything run
under `--seed` blocks listed as *development* below may be used to choose among candidates and must never
be reported as final evidence.

## Gates

| Gate | Evidence required | Status |
|---|---|---|
| G1 Unrestricted sizes | Candidate plays the full-action game through `FullActionAgent`; zero illegal actions in fuzzing; desync/fallback rate reported per opponent; sizing-probe opponents and LBR included | Implemented; results in `results/strength-gates/` |
| G2 Coverage | Fraction of decisions answered from the trained table (not a fallback) reported per opponent | Implemented |
| G3 Independent opponents | At least one externally maintained, independently trained agent at matching stack/blinds/rake | **Not met.** See "External opponents" |
| G4 Frozen test | Candidate checkpoint hash, adapter version, opponent versions, seed block, pair count and acceptance threshold written down before the run | Template below |
| G5 Adversarial search | LBR (implemented) plus trained best-response agents against the frozen candidate on fresh deals | LBR only |
| G6 Variance | AIVAT or equivalent so the required sample is feasible | Not implemented |

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

## External opponents (what exists, what blocks us)

All publicly reachable strong HUNL benchmarks found play 200 big blinds (blinds 50/100, 20,000 chips):
Slumbot (hosted API; source at github.com/ericgjackson/slumbot2019 under MIT but its competition strategy
files are not in the repository), the GTO Wizard benchmark service, and OpenHoldem-hosted baselines.
The pipeline therefore now supports `stack_bb` end to end (training, checkpoints, adapter, LBR, pool), so
a 200 BB candidate can be trained and tested at matching stacks. Remaining blockers: a client for the
hosted API (needs the exact wire format verified against the provider's current documentation, and
permission to send hands to a third-party service), and checking each service's usage terms and rake/ante
settings. No hosted opponent has been contacted.

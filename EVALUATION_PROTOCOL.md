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

## Pre-registration: final test 1 (reserved deals) and Slumbot test 2 (committed before any test hand)

Written on 2026-11-18, after the train-until-flat curves (`results/curve-v1-*/`) and before any hand below was played.

**Seed-block correction.** `doppio.fullbench` derives block seeds as base + 1,000,000 × opponent index + 1,000 × block.
Development runs from base 700000 therefore already used block seeds up to about 9.7M, including values inside the
range previously marked as reserved (≥5,000,000). That range is no longer untouched. The final block is moved to
**base 100,000,000**; no earlier run used a base above 1,000,000, so no earlier block seed falls in 100,000,000–110,000,000.

**Candidate selection (development data only).** For each stack, take the curve checkpoint with the highest development
composite score (seed 700000, six strategic pool bots, AIVAT). The scores are in `results/curve-v1-*/curve.jsonl`.
The differences between checkpoints are within noise, so the selection carries winner's-curse bias on the
development numbers only; the final numbers are on fresh deals.

```
candidate A (100 BB):        runs/buckets-42/iteration-525987.json.gz  (buckets-v1, stack_bb=100, 525,987 iterations)
  sha256:                    64e5057bad184d773675f6f8150fd0d1c26d495bcd11c5089e8e69dc808be559
candidate B (200 BB):        runs/buckets200-42/iteration-1348732.json.gz  (buckets-v1, stack_bb=200, 1,348,732 iterations)
  sha256:                    3024ae99937737e2371040c06a8c143a4ae85488ebb1f209cef36c810b09b239
adapter / harness version:   this commit (FullActionAgent, heuristic fallback, blueprint agent, no re-solving)
```

### Final test 1: reserved local deals (both candidates)

```
command:        python -m doppio.fullbench <ckpt> --seed 100000000 --pairs 30000 --blocks 10 --workers 5
                  --lbr-pairs 3000 --lbr-hands 60 --aivat --out results/final-test-1/<A|B>.json
opponents:      pool-v1, all nine (nit, tag, lag, station, maniac, min-raiser, third-pot, overbet, random-size) + LBR (sizes .5/1/2 pot)
stack/blinds:   A at 100 BB, B at 200 BB; blinds 0.5/1 BB; no rake
pairs:          30,000 paired deals (60,000 hands) per pool opponent; 3,000 pairs per LBR run. Fixed; no optional stopping
primary metric: AIVAT-corrected bb/100 with 95% t-intervals clustered by paired deal (weights 1,1,1,1, fixed earlier on dev seeds)
secondary:      raw bb/100; decision sources (table coverage); latency
claim rules:    per-opponent "beats <bot>" only if the 95% lower bound > 0; "loses to <bot>" only if the upper bound < 0
                pooled "beats every pool-v1 bot": every opponent's Bonferroni-adjusted lower bound (alpha 0.05/9) > 0
                LBR: "LBR finds no profitable exploit (this sample)" only if the lower bound > 0. LBR is a weak
                approximate best response; no result here bounds true exploitability or supports a GTO claim
run once:       each candidate is run once. A crash may be rerun with the identical command; it gives identical deals
```

### Slumbot test 2 (candidate B)

```
runner:         scripts/slumbot_test2.sh  (hosted Slumbot API, anonymous sessions; provider version not disclosed)
hands:          20,000 logged hands (fixed; no optional stopping); action seed 20, then 21, 22 ... on each restart
restarts:       crash or network failure restarts in the same log, at most 10 times; every restart is listed in
                results/slumbot/test2-restarts.log. Hands lost mid-hand are logged as abandoned
primary metric: baseline-adjusted bb/100 (winnings - baseline_winnings) over completed hands, 95% t-interval
secondary:      raw bb/100; abandoned hands also scored as 0 (sensitivity check); engine payoff checks; decision sources
claim rule:     "beats Slumbot at 200 BB (this sample)" only if the primary lower bound > 0; "loses to Slumbot"
                only if the upper bound < 0; otherwise inconclusive
also reported:  the pooled estimate with test 1 (972 hands, candidate 527,245) is NOT the primary result,
                because the candidate differs
```

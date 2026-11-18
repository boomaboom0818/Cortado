# Slumbot results (hosted API, 200 BB, blinds 50/100, no rake)

## Pilot (200 hands, before pre-registration)
Candidate `buckets200-42/iteration-154840` (buckets-v1, 155k iterations). Raw -242 bb/100 [-596, 111];
baseline-adjusted +32 [-351, 414]. Inconclusive; it established that the protocol works (200/200 engine checks).

## Test 1 (pre-registered in EVALUATION_PROTOCOL.md before any hand)
Candidate `runs/buckets200-42/iteration-527245.json.gz` (buckets-v1, 527k iterations),
sha256 a7f80e86251b8c48ffc42567bffac974686080220ba197809d594dc4ceec8084, chosen on development data only.

| Metric | Value |
|---|---|
| Hands requested / logged | 1,000 / 1,000 (+1 lost in the crash, see below) |
| Completed and scored | 972 |
| **Primary: baseline-adjusted bb/100** | **-8.4, 95% CI [-98.1, +81.2]** |
| Raw bb/100 | +109.7, 95% CI [-28.8, +248.2] |
| Correlation(winnings, Slumbot-in-our-seat baseline) | 0.82 |
| Engine payoff checks against Slumbot's reported winnings | 972 / 972 exact |
| Opponent bets needing rounding to our chip grid | 0 |
| Decisions from trained table | 100% |

**Outcome under the pre-registered rule: inconclusive.** The interval contains zero, so we can't say the bot beats or loses to Slumbot at this sample.
The positive raw number is mostly card luck. The baseline (Slumbot playing our seat with the same cards) also won,
and the adjusted estimate is near zero.

### Deviations and data handling (disclosed)
* **Hand 255:** Slumbot's server dropped the TLS connection mid-hand; the client then crashed (traceback in
  `test1-interruption-traceback.txt`). That hand was never logged. The client was hardened (no blind resend of
  an action whose reply was lost; such hands are logged as abandoned) and the run resumed in the same log up to
  the pre-registered 1,000 entries. The continuation used a new anonymous session and action seed 4 instead of 3.
  The checkpoint, rules and metric were unchanged. The stop was caused by the network, not by results.
* **28 hands abandoned** after further server-side connection drops (18 after our first decision, 10 after 2–4).
  Slumbot's own `session_total` / `session_baseline_total` show it voided every one (0 winnings, 0 baseline), so they
  carry no score. Including them as zeros gives -8.3 [-96.4, +79.8], essentially unchanged.

### What would be needed for a decisive result
The adjusted standard deviation per hand is about 14 BB. A ±20 bb/100 interval needs roughly 19,000 hands;
±10 needs about 75,000. At the observed ~2.2 s per hand (including the 0.25 s politeness delay), 19,000 hands
is about 12 hours of sequential requests. Ask before running that volume against a free service.

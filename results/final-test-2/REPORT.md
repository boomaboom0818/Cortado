# Final test 2: v3 candidates on reserved deals (pre-registered in EVALUATION_PROTOCOL.md, commit c296240)

C = v3 100 BB (3.18M iterations), D = v3 200 BB (7.22M iterations); v3 = `sizes-v1` bet sizes + `buckets-v3` cards.
Pool/LBR base seed 200,000,000; head-to-head base seed 300,000,000 (both untouched before this test). 30,000 paired
deals per pool bot and per match, 3,000 per LBR run. AIVAT-corrected bb/100, 95% t-intervals clustered by paired deal.
Bonferroni lower bound uses alpha 0.05 / 9. Each run once (`scripts/final_test2.sh`), no reruns.

## Head to head against the frozen models

| Match | AIVAT bb/100 [95% CI] | Raw bb/100 [95% CI] |
|---|---|---|
| C vs A (100 BB) | **+30.1 [+22.7, +37.6]** | +41.8 [+29.2, +54.3] |
| D vs B (200 BB) | **+33.4 [+22.3, +44.6]** | +38.2 [+21.5, +55.0] |

## Pool-v1 bots and LBR

| Opponent | C, 100 BB | D, 200 BB |
|---|---|---|
| nit | +21.8 [+12.4, +31.1] | +69.0 [+54.3, +83.7] |
| tag | +128.7 [+116.1, +141.3] | +249.9 [+229.1, +270.8] |
| lag | +286.7 [+270.7, +302.7] | +438.3 [+411.0, +465.6] |
| station | +411.8 [+395.3, +428.4] | +497.4 [+473.4, +521.4] |
| maniac | +520.4 [+501.2, +539.6] | +904.7 [+867.2, +942.3] |
| min-raiser | +177.5 [+163.5, +191.6] | +305.5 [+281.7, +329.4] |
| third-pot | +179.0 [+164.6, +193.3] | +260.4 [+236.9, +283.9] |
| overbet | +294.5 [+278.4, +310.6] | +452.9 [+424.2, +481.7] |
| random-size | +267.0 [+251.3, +282.7] | +442.6 [+415.8, +469.3] |
| LBR | +101.1 [+68.1, +134.2] | +132.4 [+72.0, +192.8] |
| Bonferroni lower bound, worst bot | +8.5 (nit) | +48.2 (nit) |

## Outcome under the pre-registered rules
* **C:** beats A head to head and beats all nine pool-v1 bots (pooled Bonferroni claim holds; A beat eight).
  LBR found no profitable exploit (lower bound +68). **Meets the promotion rule.**
* **D:** beats B head to head and beats all nine pool-v1 bots (pooled claim holds, worst case +48 vs the nit; B beat nine).
  LBR found no profitable exploit (lower bound +72). **Meets the promotion rule.**
* Trained-table coverage: 98.7%-100% of decisions (lowest vs min-raiser); the rest is the heuristic fallback.
  95th-percentile decision latency under 1 ms.

## Limits
* Pool numbers are on different deals from final test 1, so per-bot differences against A/B are not paired. A and B
  appear to win more against station and maniac; that comparison is not a test. The head-to-head is the direct comparison.
* Pool-v1 bots are scripts we wrote, and A/B are our own earlier models. Beating them is not evidence of strength
  against strong opponents. LBR is a weak approximate best response; nothing here bounds exploitability or supports a GTO claim.
* Raw data: C.json, D.json, C_vs_A.json, D_vs_B.json in this directory.
